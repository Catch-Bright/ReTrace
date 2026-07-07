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
Core functions for WiGPO (Milestone-Guided / seGmented GiGPO).

WiGPO *refines* GiGPO's anchor-state (obs) grouping with an additional
subtask-segment partition, and adds a third advantage term:

    adv = episode_adv
          + step_advantage_w     * step_adv          # GiGPO obs-cluster group
          + subgroup_advantage_w  * subgroup_adv       # (obs-cluster n segment) group

Segments are derived by sequentially matching each rollout step's projected
action against per-trial expert action anchors (milestones) stored in a JSON
file. The steps between two consecutive matched anchors form one segment; every
step is assigned a ``segment_id`` (an integer local to its trajectory).

Both step_adv and subgroup_adv normalize the *same* GiGPO ``step_rewards``
(discounted returns); the only difference is the grouping granularity.

This module reuses the numerically-tested GiGPO primitives
(``episode_norm_reward``, ``build_step_group``, ``step_norm_reward``) from
``core_gigpo`` so that WiGPO with ``subgroup_advantage_w == 0`` is *identical*
to GiGPO.
"""

import json
import os
import uuid
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np
import torch

from verl import DataProto

from gigpo import core_gigpo


# ---------------------------------------------------------- #
# --------------- Milestone data loading ------------------- #
# ---------------------------------------------------------- #

class MilestoneLoader:
    """
    Singleton loader that maps trial_id -> ordered list of milestone actions.

    JSON format:
    {
        "trajectories": [
            {"id": "<task>_<trial>", "task_type": "...", "actions": ["...", ...]},
            ...
        ]
    }
    """

    _instance: Optional["MilestoneLoader"] = None
    _milestones: Dict[str, List[str]] = {}
    _loaded: bool = False
    _loaded_path: Optional[str] = None

    def __new__(cls, json_path: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, json_path: str = None):
        # Treat empty string the same as None (e.g. an unset Hydra/CLI override
        # or a shell variable that failed to capture a path lands here as "").
        if not json_path:
            json_path = None

        # Reload if a different path is requested; otherwise load once.
        if self._loaded and (json_path is None or json_path == self._loaded_path):
            return

        if json_path is None:
            json_path = os.environ.get("WIGPO_MILESTONE_PATH", None) or None
        if json_path is None:
            # Fall back to the milestone JSON shipped inside the gigpo package.
            # __file__ is this core_wigpo.py, so dirname gives us the gigpo/ directory.
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alfworld_milestones.json")
            if os.path.exists(default_path):
                json_path = default_path

        if json_path is None or not os.path.exists(json_path):
            print(f"[MilestoneLoader] Warning: milestone JSON not found at {json_path}")
            self._milestones = {}
            self._loaded = True
            self._loaded_path = json_path
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        self._milestones = {
            item["id"]: item["actions"]
            for item in data.get("trajectories", [])
        }
        self._loaded = True
        self._loaded_path = json_path
        print(f"[MilestoneLoader] Loaded {len(self._milestones)} milestone trajectories from {json_path}")

    def get_milestones(self, trial_id: str) -> Optional[List[str]]:
        return self._milestones.get(trial_id, None)

    def has_milestones(self, trial_id: str) -> bool:
        return trial_id in self._milestones

    @property
    def num_trajectories(self) -> int:
        return len(self._milestones)

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._milestones = {}
        cls._loaded = False
        cls._loaded_path = None


def get_milestone_loader(json_path: str = None) -> MilestoneLoader:
    return MilestoneLoader(json_path)


def gamefile_to_trial_id(gamefile: Optional[str]) -> Optional[str]:
    """
    Convert an ALFWorld gamefile path to the milestone JSON ``id``.

    Path layout:
        .../json_2.1.1/<split>/<task_folder>/<trial_folder>/game.tw-pddl
    JSON id:
        "<task_folder>_<trial_folder>"

    The gamefile may point at the directory or at the ``game.tw-pddl`` file.
    """
    if gamefile is None:
        return None
    path = str(gamefile)
    # Strip a trailing filename (e.g. game.tw-pddl) if present.
    base = os.path.basename(path)
    if base.startswith("game") or base.endswith(".tw-pddl") or base.endswith(".pddl"):
        path = os.path.dirname(path)
    path = path.rstrip("/")
    parts = path.split("/")
    if len(parts) < 2:
        return None
    task_folder = parts[-2]
    trial_folder = parts[-1]
    return f"{task_folder}_{trial_folder}"


# ---------------------------------------------------------- #
# --------------- Segment matching ------------------------- #
# ---------------------------------------------------------- #

def are_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check whether two action strings are similar enough (ratio >= threshold)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def match_milestones(actions: List[str],
                     milestones: List[str],
                     threshold: float = 0.85) -> List[int]:
    """
    Sequentially match trajectory actions against the ordered milestone list.

    For each milestone m_k, find the first not-yet-consumed action a_t with
    Sim(a_t, m_k) >= threshold.

    Returns a list of matched action indices per milestone; -1 for unmatched.
    """
    k = 0
    match_indices: List[int] = []
    for t, action in enumerate(actions):
        if k >= len(milestones):
            break
        if are_similar(action, milestones[k], threshold):
            match_indices.append(t)
            k += 1
    while len(match_indices) < len(milestones):
        match_indices.append(-1)
    return match_indices


def compute_segment_ids(traj_length: int, match_indices: List[int]) -> np.ndarray:
    """
    Assign a segment_id to every step of one trajectory.

    Segment k spans (prev_matched_anchor, current_matched_anchor], i.e. the steps
    following the previous milestone up to and including the step that matched the
    current milestone. Steps after the last matched milestone (failed / trailing
    attempts) share the next segment id.

    Args:
        traj_length: number of steps in the trajectory.
        match_indices: output of ``match_milestones`` (action index per milestone,
            -1 if unmatched).

    Returns:
        segment_ids: np.ndarray[int32] of shape (traj_length,).
    """
    segment_ids = np.zeros(traj_length, dtype=np.int32)
    prev_end = -1
    seg = 0
    for match_idx in match_indices:
        if match_idx == -1:
            # No more matched anchors: remaining steps form the final segment.
            for t in range(prev_end + 1, traj_length):
                segment_ids[t] = seg
            prev_end = traj_length - 1
            break
        for t in range(prev_end + 1, match_idx + 1):
            segment_ids[t] = seg
        prev_end = match_idx
        seg += 1
    # Any steps beyond the last matched anchor (all milestones matched but the
    # trajectory kept going) go into a trailing segment.
    if prev_end < traj_length - 1:
        for t in range(prev_end + 1, traj_length):
            segment_ids[t] = seg
    return segment_ids


def compute_batch_segment_ids(batch: DataProto,
                              threshold: float = 0.85,
                              milestone_path: str = None) -> np.ndarray:
    """
    Compute per-step segment_ids for a whole batch (thin wrapper).

    Kept for backward compatibility; see ``compute_batch_segment_meta`` for the
    version that also returns milestone flags and per-task milestone counts.
    """
    segment_ids, _flags, _num = compute_batch_segment_meta(batch, threshold, milestone_path)
    return segment_ids


def compute_batch_segment_meta(batch: DataProto,
                               threshold: float = 0.85,
                               milestone_path: str = None):
    """
    Compute per-step segment metadata for a whole batch (ALFWorld / static
    action-anchor matching).

    IMPORTANT: call this while step order within each trajectory is intact
    (i.e. before ``balance_batch`` reorders the batch).

    Requires ``batch.non_tensor_batch`` to contain: ``traj_uid``, ``actions``,
    ``trial_id``, and ``step_idx``.

    Returns a tuple aligned to batch order (all shape ``(bsz,)``):
        segment_ids:      int32, subtask/segment id local to each trajectory.
        milestone_flags:  bool,  True on the step that *achieves* a milestone
                          (i.e. the last step of a completed segment).
        num_milestones:   int32, total number of milestones defined for the task
                          the step belongs to (denominator ``M`` for r_sub).
    """
    loader = get_milestone_loader(milestone_path)

    ntb = batch.non_tensor_batch
    traj_uids = ntb["traj_uid"]
    actions = ntb.get("actions", None)
    trial_ids = ntb.get("trial_id", None)
    step_idx = ntb.get("step_idx", None)

    bsz = len(traj_uids)
    segment_ids = np.zeros(bsz, dtype=np.int32)
    milestone_flags = np.zeros(bsz, dtype=bool)
    num_milestones = np.zeros(bsz, dtype=np.int32)

    if actions is None or trial_ids is None:
        print("[WiGPO Warning] actions/trial_id missing; all steps -> segment 0.")
        return segment_ids, milestone_flags, num_milestones

    for uid in np.unique(traj_uids):
        traj_pos = np.where(traj_uids == uid)[0]  # positions in the batch

        # Order steps within the trajectory (robust even if already ordered).
        if step_idx is not None:
            order = np.argsort([step_idx[p] for p in traj_pos], kind="stable")
        else:
            order = np.arange(len(traj_pos))
        ordered_pos = traj_pos[order]

        trial_id = trial_ids[ordered_pos[0]]
        milestones = loader.get_milestones(trial_id) if trial_id is not None else None
        if not milestones:
            # No milestone data (e.g. validation split): single segment, M=1.
            for p in ordered_pos:
                segment_ids[p] = 0
                num_milestones[p] = 1
            continue

        M = len(milestones)
        traj_actions = [actions[p] for p in ordered_pos]
        match_indices = match_milestones(traj_actions, milestones, threshold)
        seg = compute_segment_ids(len(traj_actions), match_indices)
        matched_set = {mi for mi in match_indices if mi != -1}
        for local_i, p in enumerate(ordered_pos):
            segment_ids[p] = seg[local_i]
            num_milestones[p] = M
            if local_i in matched_set:
                milestone_flags[p] = True

    return segment_ids, milestone_flags, num_milestones


def compute_subtask_shaped_step_rewards(batch: DataProto,
                                        segment_ids: np.ndarray,
                                        milestone_flags: np.ndarray,
                                        num_milestones: np.ndarray,
                                        gamma: float,
                                        beta: float,
                                        r_max: float = 10.0,
                                        reward_mode: str = 'cumulative') -> torch.Tensor:
    """
    Compute the WiGPO subtask-shaped per-step reward that *replaces* GiGPO's
    discounted-return ``step_rewards``.

    Two reward modes are supported:

    **cumulative** (default): Credits own subtask + all later completed subtasks.
        reward(i) = [s < m] * r_sub * gamma^(e_s - i)
                  + sum_{j=s+1}^{m-1} beta^(j-s) * r_sub * gamma^(L_j)
        where r_sub = r_max / M, and each later subtask j contributes with
        geometric distance decay beta^(j-s) and phase-length discount gamma^(L_j).

    **reverse_decay** (BEACON-style): Only credits own subtask with reverse-decay.
        reward(i) = [s < m] * gamma^(e_s - i) * r_max
        Each completed segment gets reverse-decay from its achieving step backward.
        Simpler, no cross-segment accumulation, unified [0, r_max] scale.

    **reverse_decay_cumulative** (BEACON temporal extension): On top of the own
        subtask's reverse-decay term, add the terminal reward of every *subsequent*
        completed milestone, each discounted by the number of timesteps from THIS
        segment's achieving step e_s to that later milestone's achieving step e_j:
            reward(i) = [s completed] * gamma^(e_s - i) * r_max
                      + sum_{j>s, j completed} gamma^(e_j - e_s) * r_max
        The own term still decays with the step position i (as in ``reverse_decay``),
        while each later-milestone bonus is constant within a segment (measured from
        e_s), so every step in segment s is credited for the milestones it enables.

    **reverse_decay_from_step** (per-step temporal extension): Same set of credited
        milestones as ``reverse_decay_cumulative`` (own + every later completed one),
        but EVERY milestone's discount is measured from THIS step i rather than from
        the segment's achieving step e_s:
            reward(i) = [s completed] * gamma^(e_s - i) * r_max
                      + sum_{j>s, j completed} gamma^(e_j - i) * r_max
        Equivalently, reward(i) = sum over each completed milestone k>=s of
        gamma^(e_k - i) * r_max. Every reachable future milestone is discounted by
        its own distance to step i, so a step is credited more for the milestones it
        is temporally closer to. Unlike ``reverse_decay_cumulative``, the later-
        milestone terms are NOT constant within a segment.

    **reverse_decay_cumulative_split** (milestone-split variant of cumulative):
        Same logic as ``reverse_decay_cumulative``, but each milestone contributes
        r_max/M instead of r_max (where M is the total number of milestones for this
        task, e.g., WebShop M=4 → each milestone worth 2.5). Terminal reward r_max is
        evenly split across M milestones:
            reward(i) = [s completed] * gamma^(e_s - i) * (r_max/M)
                      + sum_{j>s, j completed} gamma^(e_j - e_s) * (r_max/M)
        Completing all M milestones gives max reward = r_max.

    **reverse_decay_from_step_split** (milestone-split variant of from_step):
        Same logic as ``reverse_decay_from_step``, but each milestone contributes
        r_max/M instead of r_max:
            reward(i) = sum_{k>=s, k completed} gamma^(e_k - i) * (r_max/M)
        Completing all M milestones gives max reward = r_max.

    Args:
        batch: DataProto with traj_uid, step_idx in non_tensor_batch.
        segment_ids: per-step segment id (int32).
        milestone_flags: per-step milestone achievement flag (bool).
        num_milestones: per-step total milestone count M (int32).
        gamma: time discount factor.
        beta: geometric distance decay for later subtasks (cumulative mode only).
        r_max: max task reward (default 10.0, matches WebShop/ALFWorld).
        reward_mode: 'cumulative' (default), 'reverse_decay' (BEACON),
            'reverse_decay_cumulative' (BEACON temporal extension),
            'reverse_decay_from_step' (per-step temporal extension),
            'reverse_decay_cumulative_split' (cumulative with milestone-split reward),
            or 'reverse_decay_from_step_split' (from_step with milestone-split reward).

    Returns:
        step_rewards: torch.Tensor of shape (bsz,), aligned to batch order.
    """
    if reward_mode not in ('cumulative', 'reverse_decay',
                           'reverse_decay_cumulative', 'reverse_decay_from_step',
                           'reverse_decay_cumulative_split', 'reverse_decay_from_step_split'):
        raise ValueError(
            "reward_mode must be 'cumulative', 'reverse_decay', "
            "'reverse_decay_cumulative', 'reverse_decay_from_step', "
            "'reverse_decay_cumulative_split', or 'reverse_decay_from_step_split', "
            f"got {reward_mode}")

    ntb = batch.non_tensor_batch
    traj_uids = ntb["traj_uid"]
    step_idx = ntb.get("step_idx", None)

    bsz = len(traj_uids)
    step_rewards = np.zeros(bsz, dtype=np.float32)

    for uid in np.unique(traj_uids):
        traj_pos = np.where(traj_uids == uid)[0]

        if step_idx is not None:
            order = np.argsort([step_idx[p] for p in traj_pos], kind="stable")
        else:
            order = np.arange(len(traj_pos))
        ordered_pos = traj_pos[order]
        T = len(ordered_pos)

        seg = np.array([int(segment_ids[p]) for p in ordered_pos])
        flag = np.array([bool(milestone_flags[p]) for p in ordered_pos])
        M = int(num_milestones[ordered_pos[0]]) if num_milestones[ordered_pos[0]] > 0 else 1

        # Segment lengths L[k] and completion end index e[k] (local).
        seg_len: Dict[int, int] = defaultdict(int)
        for k in seg:
            seg_len[int(k)] += 1
        end_idx: Dict[int, int] = {}
        for i in range(T):
            if flag[i]:
                end_idx[int(seg[i])] = i

        if reward_mode == 'reverse_decay':
            # BEACON-style: each segment gets reverse-decay from achieving step
            for i in range(T):
                s = int(seg[i])
                if s in end_idx:
                    # Completed segment: reverse-decay from end
                    step_rewards[ordered_pos[i]] = (gamma ** (end_idx[s] - i)) * r_max
                else:
                    # Incomplete segment: 0
                    step_rewards[ordered_pos[i]] = 0.0

        elif reward_mode == 'reverse_decay_cumulative':
            # BEACON temporal extension: own-milestone reverse-decay term PLUS the
            # terminal reward of every later completed milestone, each discounted by
            # the timestep gap from THIS segment's achieving step e_s to that later
            # milestone's achieving step e_j.
            completed_segs = sorted(end_idx.keys())
            for i in range(T):
                s = int(seg[i])
                val = 0.0
                if s in end_idx:
                    e_s = end_idx[s]
                    # Own milestone: reverse-decay from its achieving step.
                    val += (gamma ** (e_s - i)) * r_max
                    # Later completed milestones: discount from e_s to e_j.
                    for j in completed_segs:
                        if j > s:
                            e_j = end_idx[j]
                            val += (gamma ** (e_j - e_s)) * r_max
                step_rewards[ordered_pos[i]] = val

        elif reward_mode == 'reverse_decay_from_step':
            # Per-step temporal extension: own milestone + every later completed
            # milestone, but EVERY discount is measured from THIS step i (not e_s).
            # reward(i) = sum_{k>=s, k completed} gamma^(e_k - i) * r_max
            completed_segs = sorted(end_idx.keys())
            for i in range(T):
                s = int(seg[i])
                val = 0.0
                # Sum over own milestone and all later completed milestones
                for k in completed_segs:
                    if k >= s:
                        e_k = end_idx[k]
                        val += (gamma ** (e_k - i)) * r_max
                step_rewards[ordered_pos[i]] = val

        elif reward_mode == 'reverse_decay_cumulative_split':
            # Milestone-split variant of reverse_decay_cumulative.
            # Each milestone contributes r_max/M instead of r_max.
            r_per_milestone = r_max / M
            completed_segs = sorted(end_idx.keys())
            for i in range(T):
                s = int(seg[i])
                val = 0.0
                if s in end_idx:
                    e_s = end_idx[s]
                    # Own milestone: reverse-decay from its achieving step.
                    val += (gamma ** (e_s - i)) * r_per_milestone
                    # Later completed milestones: discount from e_s to e_j.
                    for j in completed_segs:
                        if j > s:
                            e_j = end_idx[j]
                            val += (gamma ** (e_j - e_s)) * r_per_milestone
                step_rewards[ordered_pos[i]] = val

        elif reward_mode == 'reverse_decay_from_step_split':
            # Milestone-split variant of reverse_decay_from_step.
            # Each milestone contributes r_max/M instead of r_max.
            r_per_milestone = r_max / M
            completed_segs = sorted(end_idx.keys())
            for i in range(T):
                s = int(seg[i])
                val = 0.0
                # Sum over own milestone and all later completed milestones
                for k in completed_segs:
                    if k >= s:
                        e_k = end_idx[k]
                        val += (gamma ** (e_k - i)) * r_per_milestone
                step_rewards[ordered_pos[i]] = val

        else:  # cumulative
            r_sub = r_max / M
            for i in range(T):
                s = int(seg[i])
                val = 0.0
                # Own subtask term (only if this segment was completed).
                if s in end_idx:
                    val += r_sub * (gamma ** (end_idx[s] - i))
                # Later completed subtasks: contiguous s+1, s+2, ... while completed.
                # Each later subtask j contributes beta^(j-s) * r_sub * gamma^(L_j).
                j = s + 1
                while j in end_idx:
                    L_j = seg_len.get(j, 0)
                    val += (beta ** (j - s)) * r_sub * (gamma ** L_j)
                    j += 1
                step_rewards[ordered_pos[i]] = val

    return torch.tensor(step_rewards, dtype=torch.float32,
                        device=batch.batch['input_ids'].device)


# ---------------------------------------------------------- #
# --------------- Subgroup construction -------------------- #
# ---------------------------------------------------------- #

def build_subgroup_uids(step_group_uids: np.ndarray,
                        segment_ids: np.ndarray,
                        summarize: bool = False) -> np.ndarray:
    """
    Refine GiGPO obs-cluster groups by intersecting with segment_ids.

    Two steps share a subgroup iff they share both the GiGPO step-group UID
    (same episode + same anchor observation cluster) AND the same segment_id.

    Args:
        step_group_uids: obs-cluster UIDs from ``core_gigpo.build_step_group``.
        segment_ids: per-step segment ids.
        summarize: if True, print detailed statistics comparing obs-group vs subgroup sizes.

    Returns:
        subgroup_uids: np.ndarray[object] of fresh UUIDs, one per step.
    """
    subgroup_uids = np.empty(len(step_group_uids), dtype=object)
    key_to_uid: Dict[tuple, str] = {}

    # Track how each obs-group is split into subgroups
    obs_to_subgroups: Dict[str, set] = defaultdict(set)

    for i in range(len(step_group_uids)):
        key = (step_group_uids[i], int(segment_ids[i]))
        if key not in key_to_uid:
            key_to_uid[key] = str(uuid.uuid4())
        subgroup_uids[i] = key_to_uid[key]
        obs_to_subgroups[step_group_uids[i]].add(key_to_uid[key])

    if summarize:
        # Compute obs-group sizes
        obs_sizes = defaultdict(int)
        for uid in step_group_uids:
            obs_sizes[uid] += 1
        obs_size_list = list(obs_sizes.values())

        # Compute subgroup sizes
        subgroup_sizes = defaultdict(int)
        for uid in subgroup_uids:
            subgroup_sizes[uid] += 1
        subgroup_size_list = list(subgroup_sizes.values())

        # Overall statistics
        print(f"\n[WiGPO Grouping Statistics]")
        print(f"  GiGPO obs-groups: {len(obs_sizes)} groups")
        print(f"    Avg size: {np.mean(obs_size_list):.2f}, "
              f"Min: {min(obs_size_list)}, Max: {max(obs_size_list)}")

        print(f"  WiGPO subgroups (obs ∩ segment): {len(subgroup_sizes)} groups")
        print(f"    Avg size: {np.mean(subgroup_size_list):.2f}, "
              f"Min: {min(subgroup_size_list)}, Max: {max(subgroup_size_list)}")

        # Sample a few obs-groups to show split details
        print(f"\n  Sample obs-group splits:")
        sample_count = 0
        for obs_uid, sub_uids in obs_to_subgroups.items():
            if sample_count >= 3:
                break
            if len(sub_uids) > 1:  # Only show groups that were actually split
                obs_size = obs_sizes[obs_uid]
                sub_sizes = [subgroup_sizes[sub] for sub in sub_uids]
                print(f"    Obs-group (size {obs_size}) → {len(sub_uids)} subgroups "
                      f"(sizes: {sorted(sub_sizes)})")
                sample_count += 1

        # Summary of splitting degree
        splits = [len(subs) for subs in obs_to_subgroups.values()]
        avg_split = np.mean(splits)
        print(f"\n  Avg subgroups per obs-group: {avg_split:.2f}")
        print(f"  Obs-groups not split: {sum(1 for s in splits if s == 1)} / {len(splits)}")

    return subgroup_uids


# ---------------------------------------------------------- #
# --------------- WiGPO advantage -------------------------- #
# ---------------------------------------------------------- #

def compute_wigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                    step_rewards: torch.Tensor,
                                    response_mask: torch.Tensor,
                                    anchor_obs: np.ndarray,
                                    segment_ids: np.ndarray,
                                    index: np.ndarray,
                                    traj_index: np.ndarray,
                                    epsilon: float = 1e-6,
                                    step_advantage_w: float = 1.0,
                                    subgroup_advantage_w: float = 1.0,
                                    mode: str = "mean_norm",
                                    enable_similarity: bool = False,
                                    similarity_thresh: float = 0.95,
                                    ):
    """
    Compute WiGPO advantages: GiGPO refined by subtask-segment subgroups.

        adv = episode_adv
              + step_advantage_w    * step_adv         (obs-cluster group)
              + subgroup_advantage_w * subgroup_adv      (obs-cluster n segment)

    With ``subgroup_advantage_w == 0`` this is identical to GiGPO.
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Episode-level advantage (GiGPO Eq. 3).
    episode_advantages = core_gigpo.episode_norm_reward(
        token_level_rewards, response_mask, index, traj_index, epsilon, remove_std
    )

    # 2. GiGPO anchor-state (obs) grouping (Eq. 6).
    step_group_uids = core_gigpo.build_step_group(
        anchor_obs, index, enable_similarity, similarity_thresh, summarize=False
    )

    # 3. GiGPO step-level advantage (Eq. 7).
    step_advantages = core_gigpo.step_norm_reward(
        step_rewards, response_mask, step_group_uids, epsilon, remove_std
    )

    # 4. WiGPO subgroup = obs-cluster n segment.
    subgroup_uids = build_subgroup_uids(step_group_uids, segment_ids, summarize=True)

    # 5. Subgroup-level advantage: normalize the SAME step_rewards, finer groups.
    subgroup_advantages = core_gigpo.step_norm_reward(
        step_rewards, response_mask, subgroup_uids, epsilon, remove_std
    )

    # 6. Joint advantage.
    scores = (episode_advantages
              + step_advantage_w * step_advantages
              + subgroup_advantage_w * subgroup_advantages)
    return scores, scores
