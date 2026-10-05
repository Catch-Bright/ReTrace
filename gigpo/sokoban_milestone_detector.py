"""State-based milestones for single-box Sokoban."""

from collections import deque

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class SokobanMilestoneDetector:
    def __init__(self, room_fixed: np.ndarray, room_state: np.ndarray):
        self.walkable = {tuple(pos) for pos in np.argwhere(room_fixed != 0)}
        goals = np.argwhere(room_fixed == 2)
        if len(goals) != 1:
            raise ValueError("Sokoban milestone detection requires exactly one goal")
        self.goal = tuple(goals[0])
        self._distance_cache = {}

        initial_distance = self.distance(room_state)
        if initial_distance is None or initial_distance < 1:
            raise ValueError("Sokoban milestone detection requires a solvable, unfinished room")
        self.initial_distance = initial_distance
        self.num_milestones = min(2, initial_distance)
        self.halfway_remaining = initial_distance // 2
        self.phase = 0

    def distance(self, room_state: np.ndarray):
        players = np.argwhere(room_state == 5)
        boxes = np.argwhere((room_state == 4) | (room_state == 3))
        if len(players) != 1 or len(boxes) != 1:
            raise ValueError("Sokoban milestone detection requires one player and one box")
        start = (tuple(players[0]), tuple(boxes[0]))
        if start in self._distance_cache:
            return self._distance_cache[start]

        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            (player, box), steps = queue.popleft()
            if box == self.goal:
                self._distance_cache[start] = steps
                return steps
            for dr, dc in _DIRECTIONS:
                next_player = (player[0] + dr, player[1] + dc)
                if next_player not in self.walkable:
                    continue
                next_box = box
                if next_player == box:
                    next_box = (box[0] + dr, box[1] + dc)
                    if next_box not in self.walkable:
                        continue
                next_state = (next_player, next_box)
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, steps + 1))

        self._distance_cache[start] = None
        return None

    def process(self, room_state: np.ndarray, won: bool) -> bool:
        if self.phase == self.num_milestones:
            return False
        if self.num_milestones == 1 or self.phase == 1:
            achieved = won
        else:
            remaining = self.distance(room_state)
            achieved = remaining is not None and remaining <= self.halfway_remaining
        if achieved:
            self.phase += 1
        return achieved
