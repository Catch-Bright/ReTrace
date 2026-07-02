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
WebShop-specific MiGPO segment computation using dynamic milestone detection.

Unlike ALFWorld (which uses static action anchor matching), WebShop milestones
are determined by environment state transitions and goal satisfaction checks:
  - SEARCH: search results contain a goal-consistent product
  - DETAIL: clicked into a goal-consistent product detail page
  - OPTIONS: all required options selected
  - PURCHASE: bought a goal-consistent product

This module processes rollout trajectories post-hoc to assign segment_ids based
on which milestone phase each step belongs to.
"""

import numpy as np
from typing import Dict, List, Optional
from verl import DataProto
from gigpo.webshop_milestone_detector import MilestoneDetector, MilestonePhase


def compute_webshop_segment_ids_from_info(
    actions: List[str],
    states: List[str],
    next_states: List[str],
    infos: List[Dict],
    goal: Dict,
    product_lookup: Dict,
    price_lookup: Dict,
) -> np.ndarray:
    """
    Compute segment_ids for one WebShop trajectory using milestone detection.

    Args:
        actions: list of action strings for this trajectory
        states: list of observation strings (state before each action)
        next_states: list of observation strings (state after each action)
        infos: list of info dicts from environment
        goal: goal dict containing instruction, attributes, options, price_upper
        product_lookup: ASIN -> product metadata
        price_lookup: ASIN -> price

    Returns:
        segment_ids: np.ndarray[int32] of shape (traj_length,)
            0 = SEARCH phase
            1 = DETAIL phase
            2 = OPTIONS phase
            3 = PURCHASE phase
            4 = COMPLETE (post-purchase)
    """
    detector = MilestoneDetector(goal, product_lookup, price_lookup, log_dir=None)

    traj_length = len(actions)
    segment_ids = np.zeros(traj_length, dtype=np.int32)

    phase_to_id = {
        MilestonePhase.SEARCH: 0,
        MilestonePhase.DETAIL: 1,
        MilestonePhase.OPTIONS: 2,
        MilestonePhase.PURCHASE: 3,
        MilestonePhase.COMPLETE: 4,
    }

    for t in range(traj_length):
        state = states[t] if t < len(states) else ""
        next_state = next_states[t] if t < len(next_states) else ""
        info = infos[t] if t < len(infos) else {}

        result = detector.process(
            action=actions[t],
            state=state,
            next_state=next_state,
            info=info
        )

        segment_ids[t] = phase_to_id.get(result.phase, 0)

    return segment_ids


def compute_batch_segment_ids_webshop(
    batch: DataProto,
    product_lookup: Dict,
    price_lookup: Dict,
) -> np.ndarray:
    """
    Compute per-step segment_ids for a batch of WebShop trajectories.

    IMPORTANT: call this while step order within each trajectory is intact
    (i.e. before ``balance_batch`` reorders the batch).

    Requires ``batch.non_tensor_batch`` to contain:
        - traj_uid: unique trajectory identifier
        - actions: action strings
        - states: observation before action
        - next_states: observation after action
        - step_idx: step index within trajectory
        - goals: goal dict for each step (same within trajectory)
        - infos: info dict from environment

    Args:
        batch: DataProto with trajectory data
        product_lookup: ASIN -> product metadata (shared across all trajectories)
        price_lookup: ASIN -> price (shared across all trajectories)

    Returns:
        segment_ids: np.ndarray[int32] of shape (bsz,), aligned to batch order
    """
    ntb = batch.non_tensor_batch
    traj_uids = ntb["traj_uid"]
    actions = ntb.get("actions", None)
    states = ntb.get("states", None)
    next_states = ntb.get("next_states", None)
    step_idx = ntb.get("step_idx", None)
    goals = ntb.get("goals", None)
    infos = ntb.get("infos", None)

    bsz = len(traj_uids)
    segment_ids = np.zeros(bsz, dtype=np.int32)

    if actions is None or states is None or goals is None:
        print("[MiGPO-WebShop Warning] Missing required data (actions/states/goals); all steps -> segment 0.")
        return segment_ids

    # Process each trajectory
    for uid in np.unique(traj_uids):
        traj_pos = np.where(traj_uids == uid)[0]

        # Order steps within the trajectory
        if step_idx is not None:
            order = np.argsort([step_idx[p] for p in traj_pos], kind="stable")
        else:
            order = np.arange(len(traj_pos))
        ordered_pos = traj_pos[order]

        # Extract trajectory data
        traj_actions = [actions[p] for p in ordered_pos]
        traj_states = [states[p] for p in ordered_pos]
        traj_next_states = [next_states[p] for p in ordered_pos] if next_states is not None else [""] * len(traj_actions)
        traj_infos = [infos[p] for p in ordered_pos] if infos is not None else [{}] * len(traj_actions)
        goal = goals[ordered_pos[0]]  # Goal is the same for all steps in a trajectory

        # Compute segment IDs using milestone detection
        try:
            seg = compute_webshop_segment_ids_from_info(
                actions=traj_actions,
                states=traj_states,
                next_states=traj_next_states,
                infos=traj_infos,
                goal=goal,
                product_lookup=product_lookup,
                price_lookup=price_lookup,
            )
            for local_i, p in enumerate(ordered_pos):
                segment_ids[p] = seg[local_i]
        except Exception as e:
            print(f"[MiGPO-WebShop Error] Failed to compute segments for trajectory {uid}: {e}")
            # Fall back to single segment
            for p in ordered_pos:
                segment_ids[p] = 0

    return segment_ids
