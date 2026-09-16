#!/usr/bin/env python3
"""Tests for the WiGPO-repairable ratios measured on a plain GiGPO batch."""

import numpy as np

from gigpo.wigpo_diagnostics import compute_wigpo_repair_stats


class _FakeBatch:
    """Minimal DataProto-like container holding a non_tensor_batch dict."""

    def __init__(self, ntb):
        self.non_tensor_batch = ntb


def _batch(traj_won=None, episode_rewards=None):
    """One episode group with two 3-step trajectories over the same task.

    Trajectory A reaches its first milestone on step 1 -> segments [0, 0, 1].
    Trajectory B reaches nothing                       -> segments [0, 0, 0].
    Both see the same three anchor observations, so the three step groups are
    {A0,B0}, {A1,B1} and {A2,B2}; only the last one mixes subtask stages
    (A is past the first milestone there, B is not).
    """
    ntb = {
        "uid": np.array(["g"] * 6, dtype=object),
        "traj_uid": np.array(["A", "A", "A", "B", "B", "B"], dtype=object),
        "step_idx": np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "anchor_obs": np.array(["o1", "o2", "o3", "o1", "o2", "o3"], dtype=object),
        "milestone_achieved": np.array(
            [False, True, False, False, False, False], dtype=bool),
        "num_milestones": np.array([2, 2, 2, 2, 2, 2], dtype=np.int32),
    }
    if traj_won is not None:
        ntb["traj_won"] = np.array(traj_won, dtype=object)
    if episode_rewards is not None:
        ntb["episode_rewards"] = np.array(episode_rewards, dtype=object)
    return _FakeBatch(ntb)


def test_repair_stats_mixed_groups_and_failed_steps():
    # A failed, B won.
    stats = compute_wigpo_repair_stats(_batch(traj_won=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))

    # 1 of the 3 step groups spans two subtask stages; all groups hold 2 steps.
    assert stats["wigpo_diag/num_step_groups"] == 3.0
    assert stats["wigpo_diag/num_step_groups_size_ge2"] == 3.0
    assert stats["wigpo_diag/num_step_groups_mixed_stage"] == 1.0
    assert stats["wigpo_diag/step_group_mixed_stage_ratio"] == 1 / 3
    assert stats["wigpo_diag/step_group_mixed_stage_ratio_size_ge2"] == 1 / 3
    # The 2 steps of that group, out of 6 steps.
    assert stats["wigpo_diag/step_in_mixed_group_ratio"] == 2 / 6

    # 1 of 2 trajectories failed.
    assert stats["wigpo_diag/fail_traj_ratio"] == 0.5
    assert stats["wigpo_diag/num_fail_traj"] == 1.0
    assert stats["wigpo_diag/num_fail_steps"] == 3.0
    # A's steps 0 and 1 sit in segment 0, which A did complete; step 2 sits in the
    # trailing unfinished segment 1.
    assert stats["wigpo_diag/num_fail_steps_subtask_done"] == 2.0
    assert stats["wigpo_diag/fail_step_subtask_done_ratio"] == 2 / 3
    # A completed 1 of its M=2 milestones.
    assert stats["wigpo_diag/fail_traj_with_progress_ratio"] == 1.0
    assert stats["wigpo_diag/fail_traj_milestone_progress_mean"] == 0.5


def test_repair_stats_all_trajectories_won():
    stats = compute_wigpo_repair_stats(_batch(traj_won=[1.0] * 6))

    assert stats["wigpo_diag/fail_traj_ratio"] == 0.0
    assert stats["wigpo_diag/num_fail_steps"] == 0.0
    # No failed steps -> the ratio degrades to 0 instead of dividing by zero.
    assert stats["wigpo_diag/fail_step_subtask_done_ratio"] == 0.0
    assert stats["wigpo_diag/fail_traj_with_progress_ratio"] == 0.0
    assert stats["wigpo_diag/fail_traj_milestone_progress_mean"] == 0.0
    # Step-group purity is independent of the outcome.
    assert stats["wigpo_diag/step_group_mixed_stage_ratio"] == 1 / 3


def test_repair_stats_falls_back_to_episode_rewards():
    """Without traj_won, a positive episode reward counts as a win (ALFWorld)."""
    stats = compute_wigpo_repair_stats(
        _batch(episode_rewards=[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]))

    assert stats["wigpo_diag/fail_traj_ratio"] == 0.5
    assert stats["wigpo_diag/fail_step_subtask_done_ratio"] == 2 / 3


def test_repair_stats_without_milestone_flags_is_empty():
    """Detector disabled -> no stats, and no exception."""
    batch = _batch(traj_won=[0.0] * 6)
    del batch.non_tensor_batch["milestone_achieved"]

    assert compute_wigpo_repair_stats(batch) == {}


def test_repair_stats_robust_to_shuffled_batch_order():
    """Grouping and segments must not depend on the batch row order."""
    ordered = compute_wigpo_repair_stats(_batch(traj_won=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))

    batch = _batch(traj_won=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    order = [4, 0, 5, 2, 3, 1]
    batch.non_tensor_batch = {k: v[order] for k, v in batch.non_tensor_batch.items()}
    shuffled = compute_wigpo_repair_stats(batch)

    assert ordered == shuffled
