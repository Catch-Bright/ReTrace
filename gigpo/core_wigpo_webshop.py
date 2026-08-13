# Copyright 2025 verl-agent (WiGPO) contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic dynamic (event-driven) WiGPO segment computation.

WebShop and ALFWorld milestones are detected *inside the environment worker*
during rollout (that is the only place with cheap access to the goal and, for
WebShop, the product catalog). Each step therefore carries:

    - ``milestone_achieved`` (bool):  True on steps that reach a new milestone
    - ``num_milestones``      (int):  total number of milestones M for this
                                      trajectory (may differ per trajectory)
    - ``step_idx``            (int):  in-trajectory step index (optional)

Given those flags, this module turns them into per-step ``segment_id``s: a
step's segment id is the number of milestones already achieved *before* it in
the same trajectory. Steps between two consecutive milestone achievements thus
share one segment; a step that achieves a milestone is the LAST step of its
segment (the next step starts a new segment). The heavy lifting (subgroup
construction + advantage normalization) is reused unchanged from ``core_wigpo``
via ``compute_wigpo_outcome_advantage``.

Unlike the older WebShop-only helper, ``num_milestones`` is read from the batch
per trajectory, so M may differ between trajectories (e.g. ALFWorld task types).
"""

import numpy as np

from verl import DataProto

# Backwards-compatible default when a trajectory carries no explicit M.
DEFAULT_NUM_MILESTONES = 4


def compute_batch_segment_meta_dynamic(batch: DataProto,
                                       num_milestone_key: str = "num_milestones"):
    """
    Derive per-step segment metadata from ``milestone_achieved`` flags.

    IMPORTANT: call this while step order within each trajectory is intact
    (i.e. before ``balance_batch`` reorders the batch).

    Requires ``batch.non_tensor_batch`` to contain:
        - ``traj_uid``:            unique trajectory identifier per step
        - ``milestone_achieved``:  bool flag, True on steps that reach a new milestone
        - ``num_milestones``:      int, total milestone count M for this trajectory
                                   (optional; if absent, M defaults to
                                   ``DEFAULT_NUM_MILESTONES`` for every trajectory)
        - ``step_idx`` (optional): in-trajectory step index, used to order steps

    Returns a tuple aligned to batch order (all shape ``(bsz,)``):
        segment_ids:      int32, 0 before the first milestone, 1 after the first, ...
        milestone_flags:  bool,  the ``milestone_achieved`` flags (True on achieving steps).
        num_milestones:   int32, M per step (per-trajectory, from the batch).
    """
    ntb = batch.non_tensor_batch
    traj_uids = ntb["traj_uid"]
    milestone_achieved = ntb.get("milestone_achieved", None)
    step_idx = ntb.get("step_idx", None)
    per_step_M = ntb.get(num_milestone_key, None)

    bsz = len(traj_uids)
    segment_ids = np.zeros(bsz, dtype=np.int32)
    milestone_flags = np.zeros(bsz, dtype=bool)
    num_milestones = np.full(bsz, DEFAULT_NUM_MILESTONES, dtype=np.int32)

    if per_step_M is not None:
        per_step_M = np.asarray(per_step_M)
        if per_step_M.dtype == object or per_step_M.dtype.kind == "O":
            per_step_M = per_step_M.astype(np.int32)
        if per_step_M.shape[0] == bsz:
            num_milestones[:] = per_step_M

    if milestone_achieved is None:
        print("[WiGPO-dynamic] Warning: milestone_achieved missing; all steps -> segment 0.")
        return segment_ids, milestone_flags, num_milestones

    for uid in np.unique(traj_uids):
        traj_pos = np.where(traj_uids == uid)[0]  # positions in the batch

        # Order steps within the trajectory (robust even if already ordered).
        if step_idx is not None:
            order = np.argsort([step_idx[p] for p in traj_pos], kind="stable")
        else:
            order = np.arange(len(traj_pos))
        ordered_pos = traj_pos[order]

        # segment_id = number of milestones achieved strictly before this step.
        seg = 0
        for p in ordered_pos:
            segment_ids[p] = seg
            if bool(milestone_achieved[p]):
                milestone_flags[p] = True
                seg += 1

    return segment_ids, milestone_flags, num_milestones


def compute_batch_segment_ids_dynamic(batch: DataProto) -> np.ndarray:
    """Thin wrapper returning only segment_ids (see ``compute_batch_segment_meta_dynamic``)."""
    segment_ids, _flags, _num = compute_batch_segment_meta_dynamic(batch)
    return segment_ids


# -----------------------------------------------------------------------------
# WebShop-compatible wrappers (backwards compatibility).
#
# WebShop emits per-step ``milestone_achieved`` flags with M=4 (SEARCH -> DETAIL
# -> OPTIONS -> PURCHASE) but historically carried no per-step ``num_milestones``
# field, so these wrappers keep the default M and old name so the trainer's
# ``compute_batch_segment_meta_webshop`` import keeps working unchanged.
# -----------------------------------------------------------------------------

WEBSHOP_NUM_MILESTONES = 4


def compute_batch_segment_meta_webshop(batch: DataProto):
    """WebShop-backward-compatible wrapper over the generic dynamic helper.

    Equivalent to ``compute_batch_segment_meta_dynamic`` but guarantees
    ``num_milestones`` defaults to WebShop's M=4 when the batch carries none.
    """
    return compute_batch_segment_meta_dynamic(batch, num_milestone_key="num_milestones")


def compute_batch_segment_ids_webshop(batch: DataProto) -> np.ndarray:
    """Thin wrapper returning only segment_ids (WebShop name)."""
    segment_ids, _flags, _num = compute_batch_segment_meta_webshop(batch)
    return segment_ids

