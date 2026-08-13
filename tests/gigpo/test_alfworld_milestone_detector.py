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
Unit tests for the ALFWorld facts-based milestone detector.

The tests drive ``AlfworldMilestoneDetector`` with *synthetic TextWorld facts*
(reproducing the PDDL predicate semantics) derived from the expert action
trajectories shipped in ``gigpo/alfworld_milestones.json``. This exercises the
full state transition logic without requiring a TextWorld runtime.

The synthetic fact generator mirrors the ALFWorld state model:
  * ``take <obj> from <recep>``        -> holds(agent, obj)
  * ``move <obj> to <recep>``          -> inreceptacle(obj, recep)
  * ``heat/cool/clean <obj> with <r>`` -> ishot/iscool/isclean(obj)
  * ``use <toggle>``                   -> istoggled(toggle)
  * every object/receptacle mentioned  -> objecttype(<inst>, <class>)
"""

import re
from collections import defaultdict

import pytest

import numpy as np

from gigpo.alfworld_milestone_detector import (
    AlfworldMilestoneDetector,
    MilestonePhase,
    compute_num_milestones,
)

# ---------------------------------------------------------------------------
# Synthetic TextWorld-fact simulator driven by expert action strings
# ---------------------------------------------------------------------------

_OBJ_RE = re.compile(r"^(?P<cls>[a-z]+)(?:(?=\b)|(?=\|))")
_ACTION_VERB = re.compile(
    r"^(go to|take|move|put|heat|cool|clean|use|open|close|slice) "
)


def _mangle(cls: str, idx: int) -> str:
    """Mangle a class name + index into a TextWorld-style instance id."""
    return f"{cls}|0|{idx}"


class SyntheticFactsGenerator:
    """Build a per-step fact list from an expert action trajectory.

    Mangles every mentioned object/receptacle into a *persistent* instance id
    (keyed by class + occurrence count) and emits the PDDL predicates the
    detector consumes: holds / inreceptacle / ishot / iscool / isclean /
    istoggled / objecttype.
    """

    def __init__(self):
        self.cls_counter = defaultdict(int)      # cls -> next instance index
        self._id_of = {}                          # demangled token -> instance id
        self._seen_cls = set()                    # class names seen so far
        self.held: set = set()                    # inst ids held by agent
        self.in_receptacle: set = set()           # (inst, recep_inst)
        self.states: dict = defaultdict(set)      # pred -> set(...)
        self.toggled: set = set()

    def _inst(self, token: str) -> str:
        """Return the persistent instance id for a demangled token like 'potato 3'
        (or 'potato3'/'potato|3')."""
        token = token.strip()
        if token in self._id_of:
            return self._id_of[token]

        m = re.match(r"^([a-z]+)(?:\s+|\|)(\d+)$", token)
        if m:
            cls, num = m.group(1), int(m.group(2))
        elif token in self._seen_cls:
            cls, num = token, None
        else:
            cls, num = token, None

        cls_lower = cls.lower()
        idx = num if num is not None else self.cls_counter[cls_lower] + 1
        if num is None:
            self.cls_counter[cls_lower] += 1
        inst = _mangle(cls_lower, idx)
        self._id_of[token] = inst
        self.states["objecttype"].add((inst, cls_lower))
        self._seen_cls.add(cls_lower)
        return inst

    def get_facts(self, trajectory) -> list:
        """Return a list of fact-string lists, one per step, plus a final won flag."""
        facts_per_step = []
        last_won = False
        for action in trajectory:
            a = action.strip()
            toks = a.split()
            verb = toks[0]
            facts = []
            # basic: keep objecttype for already-seen instances
            if verb == "go":
                pass
            elif verb == "take":
                # take <obj> from <recep>
                obj = self._inst(self._obj_tok(toks[1:]))
                recep = self._inst(self._obj_tok(self._kw_arg(toks, "from").split()))
                self.held.add(obj)
                self.in_receptacle.discard((obj, recep))
            elif verb in ("move", "put"):
                # move <obj> to <recep>
                obj = self._inst(self._obj_tok(toks[1:]))
                recep = self._inst(self._obj_tok(self._kw_arg(toks, "to").split()))
                if obj in self.held:
                    self.held.discard(obj)
                self.in_receptacle.add((obj, recep))
            elif verb in ("heat", "cool", "clean"):
                # heat <obj> with <recep>
                obj = self._inst(self._obj_tok(toks[1:]))
                self.states[self._state_pred(verb)].add(obj)
            elif verb == "use":
                # use <toggle> -> istoggled
                t = self._inst(self._obj_tok(toks[1:]))
                self.toggled.add(t)
            elif verb in ("open", "close", "slice"):
                pass
            else:
                # unrecognized future action; skip (no-op)
                pass

            # emit facts snapshot for this step
            for inst, cls in self.states.get("objecttype", set()):
                facts.append(f"objecttype {inst} {cls}".split())
            for obj in self.held:
                facts.append(f"holds agent1 {obj}".split())
            for obj, recep in self.in_receptacle:
                facts.append(f"inreceptacle {obj} {recep}".split())
            for pred, insts in self.states.items():
                if pred == "objecttype":
                    continue
                for inst in insts:
                    facts.append(f"{pred} {inst}".split())
            for t in self.toggled:
                facts.append(f"istoggled {t}".split())

            facts_per_step.append(facts)
            # won iff final placement goal satisfied (simplified: once any
            # object is in the target receptacle, we consider the episode done).
            if self._goal_satisfied(trajectory):
                last_won = True
        return facts_per_step, last_won

    @staticmethod
    def _state_pred(verb: str) -> str:
        return {"heat": "ishot", "cool": "iscool", "clean": "isclean"}[verb]

    @staticmethod
    def _obj_tok(toks) -> str:
        """Combine a leading class token with a following integer index into
        'cls num', e.g. ``['watch','2']`` -> ``'watch 2'``. Tokens after the
        index (e.g. ``from``) are ignored."""
        if len(toks) >= 2 and toks[1].isdigit():
            return f"{toks[0]} {toks[1]}"
        return toks[0]

    @staticmethod
    def _kw_arg(toks, keyword: str) -> str:
        """Value tokens following a positional keyword in an action, e.g. the
        token after ``from``/``to``/``with`` in ``take X from R``."""
        try:
            i = toks.index(keyword)
            return " ".join(toks[i + 1: i + 3]).strip()
        except ValueError:
            return toks[-1]

    def _goal_satisfied(self, trajectory) -> bool:
        return bool(self.in_receptacle)


_ID_RE = re.compile(r".*?(?:_trial_|/)([\w-]+-[A-Za-z]+-[\w-]+-\d+)$")


def _parse_tid(tid: str):
    """Split a milestone-JSON id into (task_type, obj, mrecep, parent, scene)."""
    folder = tid.split("_trial_")[0]
    return folder, folder.split("-")


def _load_trajectories():
    import json
    from os import path
    here = path.dirname(path.abspath(__file__))
    # repo root = up 2 (tests/gigpo/..)
    root = path.dirname(path.dirname(here))
    with open(path.join(root, "gigpo", "alfworld_milestones.json")) as f:
        data = json.load(f)
    return data["trajectories"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _goals(parts):
    """Convert id folder parts (as in the file: task_type-Obj-Mrecep-Parent-scene)."""
    tt, obj, mrecep, parent, scene = parts
    goal = {"object_target": obj.lower(), "parent_target": parent.lower() if parent != "None" else ""}
    return tt, goal


@pytest.mark.parametrize("task_type,expected", [
    ("pick_and_place_simple", 2),
    ("look_at_obj_in_light", 2),
    ("pick_clean_then_place_in_recep", 3),
    ("pick_heat_then_place_in_recep", 3),
    ("pick_cool_then_place_in_recep", 3),
    ("pick_two_obj_and_place", 4),
])
def test_num_milestones(task_type, expected):
    assert compute_num_milestones(task_type) == expected


def test_expert_trajectories_reach_all_milestones():
    """Every expert trajectory that *completes* the task must let the detector
    reach all milestones; incomplete trajectories must never report the final
    milestone (state machine must not fabricate a completed anchor)."""
    trajectories = _load_trajectories()
    assert len(trajectories) > 3000
    failures = []

    def is_complete(tid, actions, tt, parent_lower):
        """A trajectory is considered complete iff its final action performs
        the task's terminal step (placing into the parent / toggling / ...)."""
        v = actions[-1].split()[0]
        if tt in ("pick_and_place_simple", "pick_two_obj_and_place"):
            return v == "move" and parent_lower in actions[-1]
        if tt in ("pick_clean_then_place_in_recep",
                  "pick_heat_then_place_in_recep",
                  "pick_cool_then_place_in_recep"):
            return v == "move" and parent_lower in actions[-1]
        if tt == "look_at_obj_in_light":
            return v in ("use", "take")
        return True

    for traj in trajectories:
        tid = traj["id"]
        actions = traj["actions"]
        parts = tid.split("_trial_")[0].split("-")
        if len(parts) != 5:
            failures.append((tid, "bad id format"))
            continue
        tt, goal = _goals(parts)
        if tt not in ("pick_and_place_simple", "pick_clean_then_place_in_recep",
                      "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
                      "look_at_obj_in_light", "pick_two_obj_and_place"):
            continue

        det = AlfworldMilestoneDetector(tt, goal, log_dir=None)
        if tt == "look_at_obj_in_light":
            det.toggle_cls = parts[3].lower()  # parent slot holds the lamp
        facts_per_step, won = SyntheticFactsGenerator().get_facts(actions)
        for facts in facts_per_step:
            det.process(facts, won=False)

        if is_complete(tid, actions, tt, parts[3].lower()):
            if det.completed_count < det.num_milestones:
                failures.append((tid, det.completed_count, det.num_milestones))
        else:
            # Incomplete trajectory: must never hit the final milestone.
            if det.completed_count >= det.num_milestones:
                failures.append((tid, det.completed_count, det.num_milestones, "over-fire"))

    assert failures == [], f"Wrong milestone counts for {len(failures)} trajectories: {failures[:10]}"


# ---------------------------------------------------------------------------
# Negative / spec tests
# ---------------------------------------------------------------------------

def test_clean_then_place_different_instance_does_not_complete():
    """clean one instance, then place a *different* instance: final anchor must
    not fire (the placed instance lacks isclean)."""
    det = AlfworldMilestoneDetector(
        "pick_clean_then_place_in_recep",
        {"object_target": "cup", "parent_target": "shelf"},
        log_dir=None,
    )
    # take cup 1 -> isclean cup 1 -> then place a DIFFERENT (uncleaned) cup 2.
    det.process(["objecttype cup|0|1 cup", "holds agent1 cup|0|1"], won=False)   # milestone TAKE
    det.process(["objecttype cup|0|1 cup", "isclean cup|0|1"], won=False)        # milestone TRANSFORM
    r = det.process(["objecttype cup|0|2 cup", "inreceptacle cup|0|2 shelf|0|1"], won=False)
    assert det.completed_count == 2  # only TAKE + TRANSFORM
    assert not r.achieved


def test_heat_then_place_same_instance_completes():
    det = AlfworldMilestoneDetector(
        "pick_heat_then_place_in_recep",
        {"object_target": "potato", "parent_target": "diningtable"},
        log_dir=None,
    )
    det.process(["objecttype potato|0|1 potato", "holds agent1 potato|0|1"], won=False)
    det.process(["objecttype potato|0|1 potato", "ishot potato|0|1"], won=False)
    r = det.process(["objecttype potato|0|1 potato",
                     "ishot potato|0|1",
                     "inreceptacle potato|0|1 diningtable|0|1"], won=False)
    assert det.completed_count == 3
    assert r.achieved
    assert r.phase == MilestonePhase.PLACE


def test_heat_then_place_different_instance_does_not_complete():
    det = AlfworldMilestoneDetector(
        "pick_heat_then_place_in_recep",
        {"object_target": "potato", "parent_target": "diningtable"},
        log_dir=None,
    )
    det.process(["objecttype potato|0|1 potato", "holds agent1 potato|0|1"], won=False)
    det.process(["objecttype potato|0|1 potato", "ishot potato|0|1"], won=False)
    r = det.process(["objecttype potato|0|2 potato",
                     "inreceptacle potato|0|2 diningtable|0|1"], won=False)
    assert det.completed_count == 2  # TAKE + TRANSFORM, NOT PLACE (placed inst not hot)
    assert not r.achieved


def test_pick_two_requires_distinct_instances():
    det = AlfworldMilestoneDetector(
        "pick_two_obj_and_place",
        {"object_target": "remotecontrol", "parent_target": "armchair"},
        log_dir=None,
    )
    # take rc1 -> place rc1 -> take rc1 AGAIN (same instance) -> place rc1
    det.process(["objecttype remotecontrol|0|1 remotecontrol", "holds agent1 remotecontrol|0|1"], False)
    placed = ["objecttype remotecontrol|0|1 remotecontrol",
              "inreceptacle remotecontrol|0|1 armchair|0|1"]
    det.process(placed, False)
    # take attempt on the SAME instance, then place it again
    det.process(["objecttype remotecontrol|0|1 remotecontrol", "holds agent1 remotecontrol|0|1"], False)
    r = det.process(placed, False)
    assert det.completed_count < 4
    assert not r.achieved


def test_pick_two_distinct_instances_completes():
    det = AlfworldMilestoneDetector(
        "pick_two_obj_and_place",
        {"object_target": "watch", "parent_target": "shelf"},
        log_dir=None,
    )
    det.process(["objecttype watch|0|1 watch", "holds agent1 watch|0|1"], False)
    placed1 = ["objecttype watch|0|1 watch", "inreceptacle watch|0|1 shelf|0|1"]
    det.process(placed1, False)   # place object 1
    det.process(["objecttype watch|0|1 watch", "objecttype watch|0|2 watch",
                 "holds agent1 watch|0|2", "inreceptacle watch|0|1 shelf|0|1"], False)  # hold object 2
    r = det.process(["objecttype watch|0|1 watch", "objecttype watch|0|2 watch",
                     "inreceptacle watch|0|1 shelf|0|1",
                     "inreceptacle watch|0|2 shelf|0|1"], False)  # place object 2
    assert det.completed_count == 4
    assert r.achieved


def test_look_at_both_orders():
    # order 1: pick object first, then toggle (holds persists)
    det = AlfworldMilestoneDetector(
        "look_at_obj_in_light",
        {"object_target": "alarmclock", "parent_target": "", "toggle_target": "desklamp"},
        log_dir=None,
    )
    det.process(["objecttype alarmclock|0|1 alarmclock", "holds agent1 alarmclock|0|1"], False)
    r = det.process(["objecttype alarmclock|0|1 alarmclock", "holds agent1 alarmclock|0|1",
                     "objecttype desklamp|0|1 desklamp", "istoggled desklamp|0|1"], False)
    assert det.completed_count == 2 and r.achieved

    # order 2: toggle first, then pick (istoggled persists)
    det2 = AlfworldMilestoneDetector(
        "look_at_obj_in_light",
        {"object_target": "book", "parent_target": "", "toggle_target": "desklamp"},
        log_dir=None,
    )
    det2.process(["objecttype desklamp|0|1 desklamp", "istoggled desklamp|0|1"], False)
    r2 = det2.process(["objecttype desklamp|0|1 desklamp", "istoggled desklamp|0|1",
                       "objecttype book|0|1 book", "holds agent1 book|0|1"], False)
    assert det2.completed_count == 2 and r2.achieved


def test_repeated_facts_do_not_reemit():
    det = AlfworldMilestoneDetector(
        "pick_and_place_simple",
        {"object_target": "apple", "parent_target": "fridge"},
        log_dir=None,
    )
    det.process(["objecttype apple|0|1 apple", "holds agent1 apple|0|1"], False)
    det.process(["objecttype apple|0|1 apple", "holds agent1 apple|0|1"], False)  # repeat
    assert det.completed_count == 1
    det.process(["objecttype apple|0|1 apple", "inreceptacle apple|0|1 fridge|0|1"], False)
    det.process(["objecttype apple|0|1 apple", "inreceptacle apple|0|1 fridge|0|1"], False)  # repeat
    assert det.completed_count == 2


def test_irrelevant_actions_do_not_advance():
    det = AlfworldMilestoneDetector(
        "pick_and_place_simple",
        {"object_target": "apple", "parent_target": "fridge"},
        log_dir=None,
    )
    # opening / going somewhere must not advance milestones
    r = det.process(["objecttype curtain|0|1 curtain"], False)
    assert not r.achieved and det.completed_count == 0
    # holding a NON-goal object also does not advance
    r2 = det.process(["objecttype knife|0|1 knife", "holds agent1 knife|0|1"], False)
    assert not r2.achieved and det.completed_count == 0


def test_won_is_authoritative():
    det = AlfworldMilestoneDetector(
        "pick_and_place_simple",
        {"object_target": "apple", "parent_target": "fridge"},
        log_dir=None,
    )
    det.process(["objecttype apple|0|1 apple"], won=False)  # nothing yet
    r = det.process(["objecttype apple|0|1 apple"], won=True)
    assert r.phase == MilestonePhase.COMPLETE
    # and the detector is now closed.
    det2 = AlfworldMilestoneDetector(
        "pick_and_place_simple",
        {"object_target": "apple", "parent_target": "fridge"},
        log_dir=None,
    )
    det2.process(["objecttype apple|0|1 apple"], won=True)
    det2.process(["objecttype apple|0|1 apple"], won=False)  # after won, no re-fire
    assert det2.completed_count == 0


# ---------------------------------------------------------------------------
# Generic dynamic segment-generation tests (per-trajectory M)
# ---------------------------------------------------------------------------

class _FakeBatch:
    """Minimal DataProto-like container holding a non_tensor_batch dict."""

    def __init__(self, ntb):
        self.non_tensor_batch = ntb


def _seg(batch):
    from gigpo.core_wigpo_webshop import compute_batch_segment_meta_dynamic
    segment_ids, flags, num = compute_batch_segment_meta_dynamic(batch)
    return segment_ids, flags, num


def test_dynamic_segments_different_M_across_trajectories():
    """Two trajectories with different M must each get their own segment ids
    and per-step num_milestones, including when the batch is shuffled."""
    import numpy as np

    # Trajectory A (M=3): milestones on steps 1 and 3.
    # Trajectory B (M=4): milestones on steps 0 and 2.
    ntb = {
        "traj_uid": np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object),
        "step_idx": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        "milestone_achieved": np.array(
            [False, True, False, True, True, False, True, False], dtype=bool),
        # per-trajectory M: A=3, B=4
        "num_milestones": np.array([3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int32),
    }
    # Shuffle the batch order while keeping step_idx to prove ordering robustness.
    order = [6, 1, 4, 0, 7, 2, 5, 3]
    ntb = {k: v[order] for k, v in ntb.items()}

    seg, flags, num = _seg(_FakeBatch(ntb))
    # Reconstruct per-trajectory, step-ordered segment ids and M (the segment
    # helper reorders internally via step_idx; here we sort by step to check).
    rec = {}
    for uid, s, n, si in zip(ntb["traj_uid"], seg, num, ntb["step_idx"]):
        rec.setdefault(uid, []).append((int(si), int(s), int(n)))
    ordered = {uid: [v for _, v, _ in sorted(pair)] for uid, pair in rec.items()}
    ordered_M = {uid: [m for _, _, m in sorted(pair)] for uid, pair in rec.items()}

    # A: steps 0..1 seg0 (milestone@1 -> last of seg0), steps2..3 seg1 (milestone@3).
    assert ordered["A"] == [0, 0, 1, 1]
    assert ordered_M["A"] == [3, 3, 3, 3]
    # B: step0 seg0 (milestone), steps1..2 seg1 (milestone@2), step3 seg2.
    assert ordered["B"] == [0, 1, 1, 2]
    assert ordered_M["B"] == [4, 4, 4, 4]


def test_dynamic_segments_default_M_when_absent():
    """WebShop-compatible: batches without num_milestones default to M=4."""
    import numpy as np

    ntb = {
        "traj_uid": np.array(["W", "W", "W"], dtype=object),
        "step_idx": np.array([0, 1, 2], dtype=np.int64),
        "milestone_achieved": np.array([True, False, True], dtype=bool),
    }
    seg, flags, num = _seg(_FakeBatch(ntb))
    assert seg.tolist() == [0, 1, 1]
    assert num.tolist() == [4, 4, 4]
    assert flags.tolist() == [True, False, True]
