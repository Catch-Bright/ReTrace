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
Measure how much of GiGPO's credit assignment WiGPO could repair.

This module is pure instrumentation: it runs on a plain GiGPO batch
(``algorithm.adv_estimator=gigpo``), touches neither rewards nor advantages,
and only reports three ratios per training iteration:

  1. ``step_group_mixed_stage_ratio`` -- of all GiGPO anchor-state step groups
     (Eq. 6: same episode group ``uid`` + same anchor observation), the fraction
     whose member steps do NOT all sit in the same subtask stage. In those
     groups GiGPO normalizes step returns across *different* subtask phases;
     WiGPO splits them by segment, so this is the share of groups WiGPO changes.

  2. ``fail_traj_ratio`` -- the fraction of trajectories that failed
     (``won == 0``), i.e. the trajectories whose steps all get a zero episode
     reward under GiGPO.

  3. ``fail_step_subtask_done_ratio`` -- among the steps of those failed
     trajectories, the fraction whose OWN subtask stage was actually completed
     (the milestone that closes the step's segment was reached later in the same
     trajectory). These steps made real, verifiable progress that GiGPO cannot
     credit; WiGPO's subtask-shaped step reward does.

Subtask stages come from the same facts-based milestone detector WiGPO uses:
``env.alfworld.use_milestone_detector=True`` makes each env worker emit a
per-step ``milestone_achieved`` flag (plus the task's milestone count ``M``),
and ``compute_batch_segment_meta_dynamic`` turns those flags into per-step
segment ids. The numbers therefore describe exactly the segments WiGPO would
have built for these same rollouts.

IMPORTANT: call this while step order within each trajectory is intact (before
``balance_batch`` reorders the batch). Calling it before ``adjust_batch`` is
also preferred -- ``adjust_batch(mode="copy")`` duplicates a few steps to make
the batch size divisible, and those duplicates would inflate the step counts.
They do not change the group count or a group's purity (a duplicate joins its
original's group with the same segment id), so the group ratios are unaffected
either way.
"""

import json
import os
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from gigpo.core_gigpo import build_step_group
from gigpo.core_wigpo_webshop import compute_batch_segment_meta_dynamic

# Prefix for the metric keys returned to the trainer (-> wandb "wigpo_diag/*").
DIAG_PREFIX = "wigpo_diag"


def compute_wigpo_repair_stats(batch,
                               enable_similarity: bool = False,
                               similarity_thresh: float = 0.95,
                               prefix: str = DIAG_PREFIX) -> Dict[str, float]:
    """
    Compute the WiGPO-repairable ratios for one GiGPO training batch.

    Args:
        batch: DataProto (or anything exposing ``non_tensor_batch``) holding one
            iteration of rollouts. Needs ``uid``, ``traj_uid``, ``anchor_obs``,
            ``milestone_achieved`` and (ideally) ``traj_won``; ``step_idx`` and
            ``num_milestones`` are used when present.
        enable_similarity: mirror of ``algorithm.gigpo.enable_similarity`` so the
            step groups measured here are the ones GiGPO actually builds.
        similarity_thresh: mirror of ``algorithm.gigpo.similarity_thresh``.
        prefix: metric key prefix.

    Returns:
        Dict of metric name -> float. Empty when the batch carries no milestone
        flags (i.e. the detector is disabled), so the caller can stay agnostic.
    """
    ntb = batch.non_tensor_batch

    if "milestone_achieved" not in ntb:
        print("[WiGPO-diag] 'milestone_achieved' is missing from the batch; set "
              "env.alfworld.use_milestone_detector=True to collect subtask stats.")
        return {}

    segment_ids, milestone_flags, num_milestones = compute_batch_segment_meta_dynamic(batch)

    stats: Dict[str, float] = {}
    stats.update(_step_group_stage_stats(ntb, segment_ids, enable_similarity, similarity_thresh))
    stats.update(_failed_traj_stats(ntb, segment_ids, milestone_flags, num_milestones))
    return {f"{prefix}/{key}": value for key, value in stats.items()}


def _step_group_stage_stats(ntb,
                            segment_ids: np.ndarray,
                            enable_similarity: bool,
                            similarity_thresh: float) -> Dict[str, float]:
    """Metric 1: share of GiGPO step groups that mix subtask stages."""
    anchor_obs = ntb.get("anchor_obs", None)
    uid = ntb.get("uid", None)
    if anchor_obs is None or uid is None:
        print("[WiGPO-diag] anchor_obs/uid missing; skipping step-group stats.")
        return {}

    # Same grouping GiGPO uses for its step-level advantage (Eq. 6).
    step_group_uids = build_step_group(
        np.asarray(anchor_obs, dtype=object),
        np.asarray(uid, dtype=object),
        enable_similarity,
        similarity_thresh,
    )

    members = defaultdict(list)
    for i, group_uid in enumerate(step_group_uids):
        members[group_uid].append(i)

    num_groups = len(members)
    num_groups_ge2 = 0   # groups that actually compare >= 2 steps
    num_mixed = 0        # groups whose steps span >= 2 subtask stages
    steps_in_mixed = 0

    for indices in members.values():
        if len(indices) > 1:
            num_groups_ge2 += 1
        if len({int(segment_ids[i]) for i in indices}) > 1:
            num_mixed += 1
            steps_in_mixed += len(indices)

    num_steps = len(segment_ids)
    return {
        # The headline ratio: mixed-stage groups / all step groups.
        "step_group_mixed_stage_ratio": _ratio(num_mixed, num_groups),
        # Same numerator over groups of size >= 2 (singletons are trivially pure
        # and only dilute the headline ratio).
        "step_group_mixed_stage_ratio_size_ge2": _ratio(num_mixed, num_groups_ge2),
        # Share of steps that sit in a mixed-stage group.
        "step_in_mixed_group_ratio": _ratio(steps_in_mixed, num_steps),
        "num_step_groups": float(num_groups),
        "num_step_groups_size_ge2": float(num_groups_ge2),
        "num_step_groups_mixed_stage": float(num_mixed),
        "num_steps": float(num_steps),
    }


def _failed_traj_stats(ntb,
                       segment_ids: np.ndarray,
                       milestone_flags: np.ndarray,
                       num_milestones: np.ndarray) -> Dict[str, float]:
    """Metrics 2 and 3: failure rate, and unrewarded-but-real progress in failures."""
    traj_uids = np.asarray(ntb["traj_uid"], dtype=object)
    won = _per_step_won(ntb)
    if won is None:
        return {}

    num_traj = 0
    num_fail = 0
    num_fail_with_progress = 0
    fail_steps = 0
    fail_steps_stage_done = 0
    fail_progress = []  # achieved/M per failed trajectory

    for uid in np.unique(traj_uids):
        positions = np.where(traj_uids == uid)[0]
        num_traj += 1
        if bool(won[positions[0]]):
            continue

        num_fail += 1
        # Milestones this trajectory actually reached, and the task's total M.
        achieved = int(np.asarray(milestone_flags)[positions].sum())
        total_milestones = max(int(num_milestones[positions[0]]), 1)

        # A step's own subtask stage is completed iff its segment id is below the
        # number of milestones achieved: segments 0..achieved-1 were each closed
        # by a milestone, segment `achieved` is the trailing unfinished one.
        fail_steps += len(positions)
        fail_steps_stage_done += int(np.sum(np.asarray(segment_ids)[positions] < achieved))

        fail_progress.append(min(achieved, total_milestones) / total_milestones)
        if achieved > 0:
            num_fail_with_progress += 1

    return {
        "fail_traj_ratio": _ratio(num_fail, num_traj),
        # The headline ratio: steps of failed trajectories whose own subtask
        # stage was completed, over all steps of failed trajectories.
        "fail_step_subtask_done_ratio": _ratio(fail_steps_stage_done, fail_steps),
        # Failed trajectories that completed at least one milestone.
        "fail_traj_with_progress_ratio": _ratio(num_fail_with_progress, num_fail),
        # Mean milestone completion (achieved/M) over failed trajectories.
        "fail_traj_milestone_progress_mean": float(np.mean(fail_progress)) if fail_progress else 0.0,
        "num_traj": float(num_traj),
        "num_fail_traj": float(num_fail),
        "num_fail_steps": float(fail_steps),
        "num_fail_steps_subtask_done": float(fail_steps_stage_done),
    }


def _per_step_won(ntb) -> Optional[np.ndarray]:
    """Per-step trajectory outcome as a bool array (True = the episode was won)."""
    won = ntb.get("traj_won", None)
    if won is not None:
        return np.array([float(w) > 0.5 for w in won], dtype=bool)

    episode_rewards = ntb.get("episode_rewards", None)
    if episode_rewards is None:
        print("[WiGPO-diag] neither 'traj_won' nor 'episode_rewards' is in the batch; "
              "skipping failure stats.")
        return None

    # Fallback: ALFWorld's text reward is 10 * won, so any positive episode
    # reward means the task was solved.
    print("[WiGPO-diag] 'traj_won' missing; treating episode_rewards > 0 as success.")
    return np.array([float(r) > 0.0 for r in episode_rewards], dtype=bool)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def log_wigpo_repair_stats(stats: Dict[str, float],
                           global_step: int,
                           path: Optional[str] = None,
                           prefix: str = DIAG_PREFIX) -> None:
    """
    Print the ratios and append one JSON line per iteration to ``path``.

    The JSONL file keeps the exact per-iteration numbers (5, 10, 15, ...) next to
    each other, which is easier to read back than scraping them out of wandb.
    Metric keys are stored without ``prefix``.
    """
    if not stats:
        return

    plain = {key.split("/", 1)[-1] if key.startswith(prefix + "/") else key: value
             for key, value in stats.items()}

    print(f"[WiGPO-diag] iteration {global_step}: "
          f"mixed-stage step groups {plain.get('step_group_mixed_stage_ratio', 0.0):.2%} "
          f"({int(plain.get('num_step_groups_mixed_stage', 0))}/"
          f"{int(plain.get('num_step_groups', 0))}), "
          f"failed trajectories {plain.get('fail_traj_ratio', 0.0):.2%} "
          f"({int(plain.get('num_fail_traj', 0))}/{int(plain.get('num_traj', 0))}), "
          f"failed steps with their subtask done "
          f"{plain.get('fail_step_subtask_done_ratio', 0.0):.2%} "
          f"({int(plain.get('num_fail_steps_subtask_done', 0))}/"
          f"{int(plain.get('num_fail_steps', 0))})")

    if not path:
        return
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"iteration": int(global_step), **plain}) + "\n")
    except OSError as e:
        print(f"[WiGPO-diag] could not append to {path}: {e}")
