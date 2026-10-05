import copy

import numpy as np
import torch

from gigpo.core_wigpo import compute_subtask_shaped_step_rewards
from gigpo.core_wigpo_webshop import compute_batch_segment_meta_dynamic
from gigpo.sokoban_milestone_detector import SokobanMilestoneDetector


def _corridor(player_col, box_col, goal_col=5):
    room_fixed = np.zeros((3, 7), dtype=np.int8)
    room_fixed[1, 1:6] = 1
    room_fixed[1, goal_col] = 2
    room_state = room_fixed.copy()
    room_state[1, player_col] = 5
    room_state[1, box_col] = 3 if box_col == goal_col else 4
    return room_fixed, room_state


def test_halfway_and_win_fire_once():
    fixed, initial = _corridor(player_col=1, box_col=3)
    detector = SokobanMilestoneDetector(fixed, initial)

    assert detector.initial_distance == 3
    assert detector.num_milestones == 2
    assert not detector.process(_corridor(2, 3)[1], won=False)
    assert detector.process(_corridor(3, 4)[1], won=False)
    assert detector.phase == 1
    assert not detector.process(_corridor(3, 4)[1], won=False)
    assert detector.process(_corridor(4, 5)[1], won=True)
    assert detector.phase == 2
    assert not detector.process(_corridor(4, 5)[1], won=True)


def test_copied_detector_keeps_phase_without_sharing_state():
    fixed, initial = _corridor(player_col=1, box_col=3)
    detector = SokobanMilestoneDetector(fixed, initial)
    halfway = _corridor(3, 4)[1]
    assert detector.process(halfway, won=False)

    branch = copy.deepcopy(detector)
    assert branch.phase == 1
    assert branch.num_milestones == detector.num_milestones
    assert branch.initial_distance == detector.initial_distance
    assert branch._distance_cache is not detector._distance_cache
    assert branch.process(_corridor(4, 5)[1], won=True)
    assert branch.phase == 2
    assert detector.phase == 1


def test_one_step_room_has_only_terminal_milestone():
    fixed, initial = _corridor(player_col=2, box_col=3, goal_col=4)
    detector = SokobanMilestoneDetector(fixed, initial)

    assert detector.initial_distance == 1
    assert detector.num_milestones == 1
    assert detector.process(_corridor(3, 4, goal_col=4)[1], won=True)


def test_deadlocked_box_does_not_trigger_halfway():
    fixed, initial = _corridor(player_col=1, box_col=3)
    detector = SokobanMilestoneDetector(fixed, initial)
    deadlocked = _corridor(player_col=2, box_col=1)[1]

    assert detector.distance(deadlocked) is None
    assert not detector.process(deadlocked, won=False)
    assert detector.phase == 0


def test_milestones_feed_split_reward_without_environment_step_rewards():
    class Batch:
        non_tensor_batch = {
            "traj_uid": np.array(["room"] * 3, dtype=object),
            "step_idx": np.arange(3),
            "milestone_achieved": np.array([False, True, True]),
            "num_milestones": np.array([2, 2, 2]),
            "rewards": np.array([-0.1, -0.1, 10.9]),
        }
        batch = {"input_ids": torch.zeros((3, 1), dtype=torch.long)}

    batch = Batch()
    segments, flags, counts = compute_batch_segment_meta_dynamic(batch)
    shaped = compute_subtask_shaped_step_rewards(
        batch, segments, flags, counts,
        gamma=0.95, beta=0.5, r_max=10.0,
        reward_mode="reverse_decay_from_step_split",
    )

    assert segments.tolist() == [0, 0, 1]
    assert torch.allclose(shaped, torch.tensor([5 * (0.95 + 0.95**2),
                                                5 * (1 + 0.95), 5.0]))


def test_failed_room_keeps_completed_halfway_credit():
    class Batch:
        non_tensor_batch = {
            "traj_uid": np.array(["room"] * 3, dtype=object),
            "step_idx": np.arange(3),
            "milestone_achieved": np.array([False, True, False]),
            "num_milestones": np.array([2, 2, 2]),
        }
        batch = {"input_ids": torch.zeros((3, 1), dtype=torch.long)}

    batch = Batch()
    segments, flags, counts = compute_batch_segment_meta_dynamic(batch)
    shaped = compute_subtask_shaped_step_rewards(
        batch, segments, flags, counts,
        gamma=0.95, beta=0.5, r_max=10.0,
        reward_mode="reverse_decay_from_step_split",
    )

    assert segments.tolist() == [0, 0, 1]
    assert torch.allclose(shaped, torch.tensor([4.75, 5.0, 0.0]))
