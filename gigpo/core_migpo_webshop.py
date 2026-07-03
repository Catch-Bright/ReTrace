# Copyright 2025 verl-agent (MiGPO) contributors.
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
WebShop-specific MiGPO segment computation.

Unlike ALFWorld (static action-anchor matching against a milestone JSON), WebShop
milestones are detected *inside the environment worker* during rollout, because
that is the only place with cheap access to the goal and product catalog. Each
step therefore already carries a boolean ``milestone_achieved`` flag (see
``WebshopWorker.step``). The four WebShop milestones are, in order:

    SEARCH -> DETAIL -> OPTIONS -> PURCHASE

Given those flags, this module only has to turn them into per-step ``segment_id``s:
a step's segment id is the number of milestones already achieved *before* it in
the same trajectory. Steps between two consecutive milestone achievements thus
share one segment, exactly mirroring the ALFWorld segment semantics. The heavy
lifting (subgroup construction + advantage normalization) is reused unchanged
from ``core_migpo`` via ``compute_migpo_outcome_advantage``.
"""

import numpy as np

from verl import DataProto


# WebShop defines four sequential milestones: SEARCH -> DETAIL -> OPTIONS -> PURCHASE.
WEBSHOP_NUM_MILESTONES = 4


def compute_batch_segment_ids_webshop(batch: DataProto) -> np.ndarray:
    """Thin wrapper returning only segment_ids (see ``compute_batch_segment_meta_webshop``)."""
    segment_ids, _flags, _num = compute_batch_segment_meta_webshop(batch)
    return segment_ids


def compute_batch_segment_meta_webshop(batch: DataProto):
    """
    Derive per-step segment metadata for a batch of WebShop trajectories from the
    ``milestone_achieved`` flags produced during rollout.

    IMPORTANT: call this while step order within each trajectory is intact
    (i.e. before ``balance_batch`` reorders the batch).

    Requires ``batch.non_tensor_batch`` to contain:
        - ``traj_uid``:            unique trajectory identifier per step
        - ``milestone_achieved``:  bool flag, True on steps that reach a new milestone
        - ``step_idx`` (optional): in-trajectory step index, used to order steps

    Returns a tuple aligned to batch order (all shape ``(bsz,)``):
        segment_ids:      int32, 0 before the first milestone, 1 after the first, ...
        milestone_flags:  bool,  the ``milestone_achieved`` flags (True on achieving steps).
        num_milestones:   int32, WEBSHOP_NUM_MILESTONES for every step (fixed M=4).
    """
    ntb = batch.non_tensor_batch
    traj_uids = ntb["traj_uid"]
    milestone_achieved = ntb.get("milestone_achieved", None)
    step_idx = ntb.get("step_idx", None)

    bsz = len(traj_uids)
    segment_ids = np.zeros(bsz, dtype=np.int32)
    milestone_flags = np.zeros(bsz, dtype=bool)
    num_milestones = np.full(bsz, WEBSHOP_NUM_MILESTONES, dtype=np.int32)

    if milestone_achieved is None:
        print("[MiGPO-WebShop Warning] milestone_achieved missing; all steps -> segment 0.")
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
