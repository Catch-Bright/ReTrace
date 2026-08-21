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
import re
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
# --------------- Action-group aggregation ----------------- #
# ---------------------------------------------------------- #

# Sentinel key for actions the projection function rejected. Invalid actions are
# never merged with valid ones, AND (unlike valid actions) they are never averaged
# with each other: "both invalid" does not mean "the same action".
INVALID_ACTION_KEY = "__INVALID__"

_WS_RE = re.compile(r"\s+")


def canonical_action_key(action: Optional[str], is_valid: bool = True) -> str:
    """
    Map a projected action string to a canonical grouping key.

    The ``<action>...</action>`` body has already been extracted and lowercased by
    the env projection function (see ``alfworld/projection.py``,
    ``webshop/projection.py``), so this only removes residual surface variation:
    case, surrounding/inner whitespace, and trailing punctuation.

    Invalid actions map to ``INVALID_ACTION_KEY`` and are excluded from averaging.
    """
    if not is_valid or action is None:
        return INVALID_ACTION_KEY
    key = _WS_RE.sub(" ", str(action).strip().lower()).strip(" .,;")
    return key if key else INVALID_ACTION_KEY


def aggregate_action_group_rewards(step_rewards: torch.Tensor,
                                   subgroup_uids: np.ndarray,
                                   action_keys: np.ndarray,
                                   summarize: bool = True) -> torch.Tensor:
    """
    Within each subgroup, average ``step_rewards`` over identical canonical actions.

    Steps that share a subgroup (same obs cluster AND same segment) but took the
    same action should receive the same credit: their reward differences come from
    *later* action choices, not from the action being evaluated. Replacing each
    step's reward with its action bucket's mean removes that downstream noise, and
    is the Q-hat(s, a) estimate that the subsequent within-subgroup normalization
    turns into Q-hat(s, a) - V-hat(s).

    Invalid actions (``INVALID_ACTION_KEY``) keep their own original reward; they
    still participate in the subgroup baseline computed downstream.

    Args:
        step_rewards: shape (bsz,)
        subgroup_uids: shape (bsz,), from ``build_subgroup_uids``
        action_keys: shape (bsz,), from ``canonical_action_key``
        summarize: print grouping diagnostics

    Returns:
        A new tensor of shape (bsz,); ``step_rewards`` is left untouched.
    """
    out = step_rewards.clone()

    with torch.no_grad():
        bucket_to_idx: Dict[tuple, List[int]] = defaultdict(list)
        for i in range(len(subgroup_uids)):
            key = action_keys[i]
            if key == INVALID_ACTION_KEY:
                continue  # keep original reward, never averaged
            bucket_to_idx[(subgroup_uids[i], key)].append(i)

        n_aggregated = 0
        for idx in bucket_to_idx.values():
            if len(idx) < 2:
                continue  # singleton bucket: mean == original value
            mean_r = out[idx].mean()
            out[idx] = mean_r
            n_aggregated += len(idx)

    if summarize:
        subgroup_keys: Dict[str, set] = defaultdict(set)
        subgroup_sizes: Dict[str, int] = defaultdict(int)
        for i in range(len(subgroup_uids)):
            subgroup_sizes[subgroup_uids[i]] += 1
            subgroup_keys[subgroup_uids[i]].add(action_keys[i])

        bsz = len(subgroup_uids)
        n_invalid = int(sum(1 for k in action_keys if k == INVALID_ACTION_KEY))
        multi_member = [u for u, s in subgroup_sizes.items() if s > 1]
        distinct_counts = [len(subgroup_keys[u]) for u in multi_member]

        print(f"\n[WiGPO Action-Group Statistics]")
        print(f"  Subgroups: {len(subgroup_sizes)} "
              f"({len(multi_member)} multi-member)")
        if distinct_counts:
            print(f"  Distinct action keys per multi-member subgroup: "
                  f"avg {np.mean(distinct_counts):.2f}, "
                  f"min {min(distinct_counts)}, max {max(distinct_counts)}")
            print(f"  Multi-member subgroups with only 1 action key: "
                  f"{sum(1 for c in distinct_counts if c == 1)} / {len(distinct_counts)}")
        print(f"  Steps whose reward was averaged: {n_aggregated} / {bsz} "
              f"({100.0 * n_aggregated / max(bsz, 1):.1f}%)")
        print(f"  Invalid actions (kept unaveraged): {n_invalid} / {bsz} "
              f"({100.0 * n_invalid / max(bsz, 1):.1f}%)")

    return out


# ---------------------------------------------------------- #
# --------------- think / action token split --------------- #
# ---------------------------------------------------------- #

_ACTION_OPEN_TAG = "<action>"

# The `<action>...</action>` block always sits at the tail of a well-formed
# response, so the span is found by walking backwards a bounded number of tokens
# instead of decoding the whole sequence.
_MAX_ACTION_TAIL_TOKENS = 96

# id -> decoded piece, so the backward scan decodes each vocab entry at most once.
_piece_cache: Dict[int, str] = {}
_piece_cache_owner: Optional[int] = None


def _token_piece(tokenizer, token_id: int) -> str:
    global _piece_cache_owner
    if _piece_cache_owner != id(tokenizer):
        _piece_cache.clear()
        _piece_cache_owner = id(tokenizer)
    piece = _piece_cache.get(token_id)
    if piece is None:
        piece = tokenizer.decode([token_id])
        _piece_cache[token_id] = piece
    return piece


def _locate_action_tag_full(responses_cpu: torch.Tensor,
                            valid_pos: List[int],
                            tokenizer) -> Optional[int]:
    """
    Robust fallback for locating the ``<action>`` opening tag: decode the whole
    valid response and search the full text.

    The fast path (``build_action_token_mask``) only scans the last
    ``max_tail_tokens`` pieces; when the tag lies beyond that window, or the
    tail pieces happen not to reassemble ``<action>`` contiguously, this helper
    tries again on the complete decoded response -- mirroring how the env
    projection extracts the action from the full raw string.

    Returns the token position whose decoded piece contains the first character
    of ``<action>``, or None if the tag is not present.

    Args:
        responses_cpu: token ids for ONE row, shape (1, response_length), on CPU.
        valid_pos: 0/1 validity mask positions for that row, ascending.
        tokenizer: HF tokenizer used to produce ``responses``.
    """
    # Decode only the valid tokens so padding / trailing specials do not shift
    # the character offset of the tag.
    pieces = []
    lengths = []
    text = ""
    for p in valid_pos:
        piece = _token_piece(tokenizer, int(responses_cpu[0, p]))
        pieces.append(piece)
        lengths.append(len(piece))
        text += piece

    char_idx = text.lower().find(_ACTION_OPEN_TAG)
    if char_idx == -1:
        return None

    # Map the character offset back to the token that covers it.
    acc = 0
    for p, ln in zip(valid_pos, lengths):
        if char_idx < acc + ln:
            return p
        acc += ln
    return None


def build_action_token_mask(responses: torch.Tensor,
                            response_mask: torch.Tensor,
                            tokenizer,
                            max_tail_tokens: int = _MAX_ACTION_TAIL_TOKENS,
                            summarize: bool = True):
    """
    Split each response into its *think* part and its *action* part.

    A step's generation is ``<think>...</think><action>...</action>``. The two
    parts answer different questions and therefore deserve different credit (see
    ``compute_prob_weighted_think_action_advantages``), so the tokens have to be separated.

    The action part is defined as every valid response token from the first token
    that contributes to the opening ``<action>`` tag through the end of the
    response. Tokenizers merge the tag with its neighbours in several ways
    (``'><' | 'action' | '>'``, ``'<' | 'action' | '>'``, ``' <' | 'action' | '>'``),
    so the boundary is located on the *decoded* text rather than on token ids, and
    the merged boundary token is assigned to the action part.

    Two search strategies, in order:

    1. Fast path: walk backwards over the last ``max_tail_tokens`` valid tokens,
       accumulating decoded pieces until ``<action>`` appears. Cheap, covers the
       typical case where the tag sits at the tail of the response.
    2. Full fallback: when the fast path fails, decode the WHOLE valid response
       and search the complete text (like the env projection does on the raw
       string). This handles tags beyond the tail window or piece boundaries
       that defeat the backward scan.

    A step is marked ``located`` (and participates in the probability split)
    only if one of the two searches actually found the tag; otherwise its mask
    stays all-zero and it falls back to the raw subgroup advantage downstream.

    Args:
        responses: token ids, shape (bsz, response_length).
        response_mask: 0/1 validity mask, shape (bsz, response_length).
        tokenizer: HF tokenizer used to produce ``responses``.
        max_tail_tokens: how far back from the end of the response to look
            (fast path only).
        summarize: print diagnostics.

    Returns:
        action_token_mask: same shape/dtype as ``response_mask``; 1 on action
            tokens, 0 elsewhere (including on padding). Rows where the tag could
            not be located stay all-zero, which makes them fall back to the
            undecomposed advantage downstream.
        located: np.ndarray[bool] of shape (bsz,), True where the tag was found.
    """
    bsz = response_mask.shape[0]
    action_token_mask = torch.zeros_like(response_mask)
    located = np.zeros(bsz, dtype=bool)

    responses_cpu = responses.detach().cpu()
    mask_cpu = response_mask.detach().cpu()

    n_full_fallback = 0
    for b in range(bsz):
        valid_pos = torch.nonzero(mask_cpu[b], as_tuple=False).flatten().tolist()
        if not valid_pos:
            continue

        # Fast path: bounded backward scan over decoded tail pieces.
        tail = ""
        start_pos = None
        for pos in reversed(valid_pos[-max_tail_tokens:]):
            tail = _token_piece(tokenizer, int(responses_cpu[b, pos])) + tail
            if _ACTION_OPEN_TAG in tail.lower():
                start_pos = pos
                break

        # Full fallback: decode the whole valid response and search the text,
        # mirroring the env projection's find("<action>").
        if start_pos is None:
            n_full_fallback += 1
            start_pos = _locate_action_tag_full(
                responses_cpu[b:b + 1], valid_pos, tokenizer)

        if start_pos is None:
            continue

        located[b] = True
        action_token_mask[b, start_pos:valid_pos[-1] + 1] = 1
        # Only tokens that were valid to begin with may be marked.
        action_token_mask[b] = action_token_mask[b] * mask_cpu[b].to(action_token_mask.dtype)

    action_token_mask = action_token_mask.to(response_mask.device)

    if summarize:
        n_located = int(located.sum())
        n_action = action_token_mask.sum().item()
        n_valid = response_mask.sum().item()
        print(f"\n[WiGPO think/action Split]")
        print(f"  Responses with a located <action> span: {n_located} / {bsz} "
              f"({100.0 * n_located / max(bsz, 1):.1f}%)")
        print(f"  Action tokens: {n_action:.0f} / {n_valid:.0f} valid response tokens "
              f"({100.0 * n_action / max(n_valid, 1):.1f}%)")
        if n_full_fallback:
            print(f"  Full-decoded fallback searches: {n_full_fallback} "
                  f"(fast path failed, full-text search attempted)")
        if n_located < bsz:
            print(f"  Unlocated rows fall back to the undecomposed subgroup advantage.")

    return action_token_mask, located


def compute_action_probs(old_log_probs: torch.Tensor,
                         action_token_mask: torch.Tensor,
                         response_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the average log-probability of action tokens for each step.

    For each step i, compute:
        p_i = exp((1/|a_i|) * sum_{j in action tokens} log π_old(y_{i,j}))

    This represents how much the think part supports the actual action taken.

    Args:
        old_log_probs: per-token log probabilities from π_old, shape (bsz, seq_len)
        action_token_mask: binary mask for action tokens, shape (bsz, seq_len)
        response_mask: binary mask for valid tokens, shape (bsz, seq_len)

    Returns:
        action_probs: shape (bsz,), the probability p_i for each step
    """
    with torch.no_grad():
        # Mask to select only action tokens
        action_mask = action_token_mask * response_mask

        # Sum of log probs over action tokens
        action_log_sum = (old_log_probs * action_mask).sum(dim=-1)

        # Number of action tokens per step
        action_token_count = action_mask.sum(dim=-1).clamp(min=1.0)  # avoid division by zero

        # Average log prob over action tokens
        avg_action_log_prob = action_log_sum / action_token_count

        # Convert to probability
        action_probs = torch.exp(avg_action_log_prob)

    return action_probs


def compute_prob_weighted_think_action_advantages(
        step_rewards: torch.Tensor,
        subgroup_uids: np.ndarray,
        action_keys: np.ndarray,
        action_probs: torch.Tensor,
        located: np.ndarray,
        epsilon: float = 1e-6,
        remove_std: bool = True,
        clip_min: float = 0.5,
        clip_max: float = 1.5,
        summarize: bool = True):
    """
    Compute think and action advantages based on action probability weighting,
    with fallbacks for steps/subgroups where the split is ill-defined.

    Baseline (shared by every step in the same subgroup):

        A_raw_i = normalize(raw_step_reward_i, subgroup 内所有 raw_step_rewards)

        mean_norm:      A_raw_i = r_i - mean(r_subgroup)
        mean_std_norm:  A_raw_i = (r_i - mean(r_subgroup)) / (std(r_subgroup) + eps)

    Only steps satisfying ALL of the following conditions enter the probability-
    weighted think/action split:

      1. ``located[i]`` is True (the <action> boundary was found);
      2. its action category has >= 2 located steps (otherwise p_i / p_bar
         carries no comparison information);
      3. its subgroup has >= 2 distinct valid action categories (otherwise
         action-group averaging collapses every step's advantage to 0).

    For such steps:

        p_bar      = mean of p_i over located steps of the same action category
        rho_i      = clip(p_i / p_bar, clip_min, clip_max)
        A_act_i    = A_public   (shared by all steps taking that action)
        A_think_i  = rho_i * A_public

    where ``A_public`` is the action-bucket mean reward normalized within the
    subgroup, averaged over ALL steps that took the action (unlocated steps
    still provide reward evidence for the action).

    Every other step -- unlocated, or in a singleton action category, or in a
    single-action-category subgroup, or with an invalid action -- falls back to
    ``A_raw_i`` on ALL its valid tokens (think and action alike). In particular:

    * an unlocated step never contributes to ``p_bar`` and never affects other
      steps' ``rho``;
    * a subgroup with a single step gives A_raw = 0 automatically (mean of a
      singleton is the reward itself);
    * a subgroup whose steps all share one valid action category skips
      aggregation entirely and uses A_raw for every step.

    Args:
        step_rewards: raw per-step rewards, shape (bsz,)
        subgroup_uids: subgroup id per step, shape (bsz,)
        action_keys: canonical action key per step, shape (bsz,)
        action_probs: probability p_i per step, shape (bsz,); values for
            unlocated steps are ignored.
        located: bool per step, True where the <action> span was found
            (output of ``build_action_token_mask``).
        epsilon: denominator guard.
        remove_std: if True, only center (no division), matching
            ``core_gigpo.step_norm_reward``'s mean_norm behaviour.
        clip_min: minimum value for rho_i.
        clip_max: maximum value for rho_i.
        summarize: print diagnostics.

    Returns:
        think_adv: shape (bsz,), advantage for think tokens (== A_raw on
            fallback steps, rho_i * A_public on split steps).
        act_adv: shape (bsz,), advantage for action tokens (== A_raw on
            fallback steps, A_public on split steps).
    """
    with torch.no_grad():
        bsz = step_rewards.shape[0]
        device = step_rewards.device
        dtype = step_rewards.dtype

        # ---- per-subgroup raw normalization (A_raw), shared baseline ----
        subgroup_to_indices: Dict[object, List[int]] = defaultdict(list)
        for i in range(bsz):
            subgroup_to_indices[subgroup_uids[i]].append(i)

        subgroup_mean: Dict[object, float] = {}
        subgroup_std: Dict[object, float] = {}
        for sg_id, indices in subgroup_to_indices.items():
            rewards = step_rewards[indices]
            subgroup_mean[sg_id] = rewards.mean().item()
            if remove_std:
                subgroup_std[sg_id] = 1.0
            else:
                # Singleton subgroups get std 1.0 like core_gigpo, so their
                # A_raw = (r - mean([r])) / (1 + eps) = 0.
                subgroup_std[sg_id] = rewards.std().item() if len(indices) > 1 else 1.0

        a_raw = torch.zeros(bsz, dtype=dtype, device=device)
        for i in range(bsz):
            sg = subgroup_uids[i]
            if remove_std:
                # mean_norm: center only, no division (matches core_gigpo and
                # the spec A_raw_i = r_i - mean(r_subgroup)).
                a_raw[i] = step_rewards[i] - subgroup_mean[sg]
            else:
                a_raw[i] = (step_rewards[i] - subgroup_mean[sg]) / (subgroup_std[sg] + epsilon)

        # ---- action categories (group = subgroup x action) ----
        # group_all: every step that took the action (feeds the public reward).
        # group_located: only steps with a detected think/action boundary
        # (feeds p_bar / rho; unlocated steps must not pollute the mean).
        group_all: Dict[tuple, List[int]] = defaultdict(list)
        group_located: Dict[tuple, List[int]] = defaultdict(list)
        subgroup_valid_actions: Dict[object, set] = defaultdict(set)
        for i in range(bsz):
            key = action_keys[i]
            if key == INVALID_ACTION_KEY:
                continue
            gk = (subgroup_uids[i], key)
            group_all[gk].append(i)
            if located[i]:
                group_located[gk].append(i)
                subgroup_valid_actions[subgroup_uids[i]].add(key)

        # An action category is comparable only if its subgroup holds >= 2
        # distinct valid action categories; within it, the split needs >= 2
        # located steps.
        subgroup_eligible = {
            sg: len(actions) >= 2 for sg, actions in subgroup_valid_actions.items()
        }
        group_eligible = {
            gk: len(group_located[gk]) >= 2 for gk in group_all
        }

        # ---- public action advantage A_public for eligible groups ----
        # Action-bucket mean over ALL steps that took the action (unlocated
        # steps still provide reward evidence), normalized within the subgroup.
        public_adv = torch.zeros(bsz, dtype=dtype, device=device)
        for gk, indices in group_all.items():
            if not group_eligible[gk]:
                continue
            sg = gk[0]
            mean_r = step_rewards[indices].mean()
            if remove_std:
                public_adv[indices] = mean_r - subgroup_mean[sg]
            else:
                public_adv[indices] = (mean_r - subgroup_mean[sg]) / (subgroup_std[sg] + epsilon)

        # ---- per-step split factor rho_i (eligible steps only) ----
        rho = torch.ones(bsz, dtype=dtype, device=device)
        for gk, indices in group_located.items():
            if not group_eligible[gk]:
                continue
            p_bar = action_probs[indices].mean().item()
            if p_bar <= 1e-9:
                continue
            for i in indices:
                rho[i] = max(clip_min, min(clip_max, action_probs[i].item() / p_bar))

        # ---- final per-step advantages with fallbacks ----
        think_adv = torch.zeros(bsz, dtype=dtype, device=device)
        act_adv = torch.zeros(bsz, dtype=dtype, device=device)
        n_split = 0
        n_unlocated = 0
        n_singleton_action = 0
        n_single_action_subgroup = 0
        n_invalid = 0
        for i in range(bsz):
            key = action_keys[i]
            if key == INVALID_ACTION_KEY:
                n_invalid += 1
                think_adv[i] = a_raw[i]
                act_adv[i] = a_raw[i]
                continue
            gk = (subgroup_uids[i], key)
            sg = subgroup_uids[i]
            if (located[i] and group_eligible.get(gk, False)
                    and subgroup_eligible.get(sg, False)):
                think_adv[i] = rho[i] * public_adv[i]
                act_adv[i] = public_adv[i]
                n_split += 1
            else:
                if not located[i]:
                    n_unlocated += 1
                elif not group_eligible.get(gk, False):
                    n_singleton_action += 1
                else:
                    n_single_action_subgroup += 1
                think_adv[i] = a_raw[i]
                act_adv[i] = a_raw[i]

        if summarize:
            rho_np = rho.cpu().numpy()
            split_rho = rho_np[np.abs(rho_np - 1.0) > 1e-9]  # rho != 1 means it was computed
            n_clip_low = int((rho_np <= clip_min + 1e-6).sum())
            n_clip_high = int((rho_np >= clip_max - 1e-6).sum())
            print(f"\n[WiGPO Prob-Weighted Think/Action Advantages]")
            print(f"  Action groups: {len(group_all)} "
                  f"({sum(1 for v in group_eligible.values() if v)} eligible for split)")
            print(f"  Split steps: {n_split} / {bsz}")
            print(f"  Fallback to raw subgroup advantage: {bsz - n_split} / {bsz}")
            print(f"    - unlocated think/action boundary: {n_unlocated}")
            print(f"    - singleton action category (<2 located steps): {n_singleton_action}")
            print(f"    - subgroup with a single valid action category: {n_single_action_subgroup}")
            print(f"    - invalid action: {n_invalid}")
            if len(split_rho) > 0:
                print(f"  rho (split steps only) stats: "
                      f"mean={split_rho.mean():.3f}, "
                      f"min={split_rho.min():.3f}, "
                      f"max={split_rho.max():.3f}")
            print(f"  rho clipped at [{clip_min}, {clip_max}]: "
                  f"{n_clip_low + n_clip_high} / {bsz}")

    return think_adv, act_adv


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
                                    action_keys: np.ndarray = None,
                                    action_token_mask: torch.Tensor = None,
                                    located: np.ndarray = None,
                                    old_log_probs: torch.Tensor = None,
                                    prob_reweight_clip_min: float = 0.5,
                                    prob_reweight_clip_max: float = 1.5,
                                    ):
    """
    Compute WiGPO advantages: GiGPO refined by subtask-segment subgroups.

        adv = episode_adv
              + step_advantage_w    * step_adv         (obs-cluster group)
              + subgroup_advantage_w * subgroup_adv      (obs-cluster n segment)

    With ``subgroup_advantage_w == 0`` this is identical to GiGPO.

    The subgroup term has several forms:
    1. ``action_keys is None``: plain within-subgroup normalization of raw step rewards.
    2. ``action_keys`` given, others None: step rewards averaged over identical actions,
       whole response gets the same aggregated advantage.
    3. ``action_keys``, ``action_token_mask``, ``located``, and ``old_log_probs``
       all given: probability-weighted think/action split. Action tokens get the
       public action advantage A_public; think tokens get rho_i * A_public, where
       rho_i reflects how strongly this step's think supports the action compared
       to other steps with the same action in the same subgroup.

       ``located`` gates the split (see
       ``compute_prob_weighted_think_action_advantages`` for the full rules):
       steps without a detected <action> boundary, singleton action categories,
       and subgroups with a single valid action category fall back to the raw
       within-subgroup advantage A_raw on all their tokens. A located step never
       contaminates the p_bar of its action category.

    The obs-cluster ``step_adv`` term always uses the raw, unaggregated step rewards.
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

    # 5. Subgroup-level advantage with probability-weighted think/action split.
    if (action_keys is not None and action_token_mask is not None
        and old_log_probs is not None and located is not None):
        # Probability-weighted split: think and action get different advantages
        # based on how much the think supports the action. ``located`` gates the
        # split: steps without a detected <action> boundary (and their action
        # categories / subgroups, per the fallback rules) use A_raw instead.

        # Compute action probabilities
        action_probs = compute_action_probs(old_log_probs, action_token_mask, response_mask)

        # Compute think and action advantages
        think_adv, act_adv = compute_prob_weighted_think_action_advantages(
            step_rewards=step_rewards,
            subgroup_uids=subgroup_uids,
            action_keys=action_keys,
            action_probs=action_probs,
            located=located,
            epsilon=epsilon,
            remove_std=remove_std,
            clip_min=prob_reweight_clip_min,
            clip_max=prob_reweight_clip_max,
            summarize=True,
        )

        # Broadcast to token level. For fallback steps think_adv == act_adv ==
        # A_raw, so every valid token receives the same raw advantage regardless
        # of the mask; for split steps the masks separate the two roles.
        response_length = response_mask.shape[-1]
        think_adv_tok = think_adv.unsqueeze(-1).tile([1, response_length]) * response_mask
        act_adv_tok = act_adv.unsqueeze(-1).tile([1, response_length]) * response_mask

        # Think mask = valid tokens that are not action tokens
        think_mask = response_mask * (1 - action_token_mask.to(response_mask.dtype))

        # Apply advantages: action tokens get act_adv, think tokens get think_adv
        subgroup_advantages = act_adv_tok * action_token_mask.to(response_mask.dtype) + think_adv_tok * think_mask

    elif action_keys is not None:
        # Action-group aggregation without probability weighting
        subgroup_step_rewards = aggregate_action_group_rewards(
            step_rewards, subgroup_uids, action_keys, summarize=True
        )
        subgroup_advantages = core_gigpo.step_norm_reward(
            subgroup_step_rewards, response_mask, subgroup_uids, epsilon, remove_std,
            std_rewards=step_rewards,
        )
    else:
        # No action grouping: plain subgroup advantage
        subgroup_advantages = core_gigpo.step_norm_reward(
            step_rewards, response_mask, subgroup_uids, epsilon, remove_std
        )

    # 6. Joint advantage.
    scores = (episode_advantages
              + step_advantage_w * step_advantages
              + subgroup_advantage_w * subgroup_advantages)
    return scores, scores
