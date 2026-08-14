#!/usr/bin/env python3
"""
Standalone milestone detector for ALFWorld (TextWorld) trajectories.

ALFWorld milestones are derived from the *true world state* using the TextWorld
facts that are requested per step (``game_state["facts"]``), rather than from
action text or rule-based feedback parsing. A fact has the form

    ``<predicate> <instance> [<instance> ...]``

e.g. ``holds agent1 potato|0|1``, ``ishot potato|0|1``,
``inreceptacle potato|0|1 diningtable|0|1``. Each instance is a mangled
entity id; the predicate names come from ``data/alfred.pddl``.

The detector is task-type specific, and its milestones follow the six ALFWorld
task types:

    pick_and_place_simple            holds(obj) -> inreceptacle(obj, parent)          M=2
    look_at_obj_in_light             holds(obj) and istoggled(obj_toggle)             M=2
    pick_clean_then_place_in_recep   holds(obj) -> isclean(obj) -> isclean+inreceptacle M=3
    pick_heat_then_place_in_recep    holds(obj) -> ishot(obj)  -> ishot+inreceptacle   M=3
    pick_cool_then_place_in_recep    holds(obj) -> iscool(obj) -> iscool+inreceptacle  M=3
    pick_two_obj_and_place           holds(a) -> inreceptacle(a,parent)
                                     -> holds(b, b!=a) -> inreceptacle(b,parent)      M=4

Design notes:
  * State only moves forward: repeating a fact / repeating an action never
    re-emits a milestone.
  * Goal-object constraints bind to a concrete instance id `x`; the final
    milestone requires *the same `x`* satisfying all conjuncts, so e.g. heat
    one instance and place another never completes the final anchor.
  * `look_at_obj_in_light` uses two independent conditions, allowing the agent
    to turn the lamp on before picking the object.
  * `pick_two_obj_and_place` advances only when two *different* target
    instances have been placed.
  * `won` never advances the state machine; only the current expected fact
    condition can emit the next milestone.

The detector needs only:
  * the task_type, and goal pddl_params (object_target / parent_target /
    toggle_target) -- read once from the episode's own ``traj_data.json``;
  * the per-step ``facts`` list.

It deliberately does NOT read ``alfworld_milestones.json``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Milestone config per task type
# ---------------------------------------------------------------------------

_NUM_MILESTONES = {
    "pick_and_place_simple": 2,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 3,
    "pick_cool_then_place_in_recep": 3,
    "pick_two_obj_and_place": 4,
}

_MILESTONE_NAMES = {
    "pick_and_place_simple": ["获得目标物", "目标物放入目标容器"],
    "pick_clean_then_place_in_recep": ["获得目标物", "目标物已清洗", "已清洗目标物放入目标容器"],
    "pick_heat_then_place_in_recep": ["获得目标物", "目标物已加热", "已加热目标物放入目标容器"],
    "pick_cool_then_place_in_recep": ["获得目标物", "目标物已冷却", "已冷却目标物放入目标容器"],
    "look_at_obj_in_light": ["获得目标物", "目标灯已点亮"],
    "pick_two_obj_and_place": ["获得物体1", "放置物体1", "获得物体2", "放置物体2"],
}

# Predicate names (lowercased) from data/alfred.pddl.
_PRED = {
    "holds": "holds",
    "in_receptacle": "inreceptacle",
    "is_clean": "isclean",
    "is_hot": "ishot",
    "is_cool": "iscool",
    "is_toggled": "istoggled",
    "object_type": "objecttype",
    "receptacle_type": "receptacletype",
}


def compute_num_milestones(task_type: str) -> int:
    """Number of milestones M for a given ALFWorld task type."""
    return _NUM_MILESTONES.get(task_type, 1)


def load_goal_params(gamefile) -> Dict[str, str]:
    """
    Read a task's goal pddl_params from the episode's own ``traj_data.json``.

    ``gamefile`` is the path to the episode's ``game.tw-pddl`` (which the env
    surfaces via ``extra.gamefile``); ``traj_data.json`` sits in the same trial
    directory. Returns a dict usable as ``goal_params`` for the detector
    (``object_target`` / ``parent_target`` / ``toggle_target``). Returns an
    empty dict if the file cannot be located/read; the caller can then fall
    back to no milestone detection for that episode.
    """
    task_type, goal = load_task_params(gamefile)
    return goal


def load_task_params(gamefile) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Read ``(task_type, goal_params)`` from the episode's own ``traj_data.json``.

    ``gamefile`` is the path to the episode's ``game.tw-pddl``; ``traj_data.json``
    sits in the same trial directory. ``goal_params`` is a dict with
    ``object_target`` / ``parent_target`` / ``toggle_target``. If the file is
    missing/unreadable, returns ``(None, {})`` so the caller can disable
    milestone detection for that episode.
    """
    if gamefile is None:
        return None, {}
    path = str(gamefile)
    if os.path.basename(path).startswith("game") or path.endswith((".tw-pddl", ".pddl")):
        path = os.path.dirname(path)
    traj_json = os.path.join(path, "traj_data.json")
    if not os.path.exists(traj_json):
        return None, {}
    try:
        import json

        with open(traj_json) as f:
            traj = json.load(f)
        pp = traj.get("pddl_params", {})
        goal = {
            "object_target": str(pp.get("object_target", "")),
            "parent_target": str(pp.get("parent_target", "")),
            "toggle_target": str(pp.get("toggle_target", "")),
        }
        return str(traj.get("task_type", "")), goal
    except Exception as e:  # noqa: BLE001 - best-effort read
        print(f"[Alfworld] failed to load task params from {traj_json}: {e}")
        return None, {}


class MilestonePhase(Enum):
    TAKE = auto()
    TRANSFORM = auto()    # clean / heat / cool
    PLACE = auto()
    TOGGLE = auto()       # look_at_obj_in_light: turn the lamp on
    COMPLETE = auto()


@dataclass
class MilestoneResult:
    """Return value for each processed step."""

    phase: MilestonePhase
    achieved: bool
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fact parsing helpers
# ---------------------------------------------------------------------------

def _parse_facts(facts: Iterable[Any]) -> List[List[str]]:
    """
    Tokenize a facts collection into ``[predicate, arg1, arg2, ...]`` lists.

    Each argument is kept as ONE token even when it contains a space: ALFWorld
    demangles entity names, so real TextWorld facts look like
    ``Proposition('inreceptacle', (Variable('bread 3'), Variable('diningtable 1')))``
    and must NOT be split on whitespace.

    ``facts`` elements may be:
      * TextWorld ``Proposition`` objects (``.name`` + ``.names``/``.arguments``),
      * already-split token lists/tuples (arguments pre-separated),
      * ``"pred arg1 arg2"`` strings (whitespace-split; only safe for
        single-word args, e.g. mangled ids used in tests).
    """
    out: List[List[str]] = []
    for f in facts:
        # TextWorld Proposition: has .name and .names (or .arguments).
        name = getattr(f, "name", None)
        if name is not None:
            tokens = [str(name).strip().lower()]
            names = getattr(f, "names", None)
            if names:
                tokens += [str(n).strip().lower() for n in names]
            else:
                args = getattr(f, "arguments", None)
                if args:
                    tokens += [str(getattr(a, "name", a)).strip().lower() for a in args]
            out.append(tokens)
            continue
        # Already-split token list (arguments already separated).
        if isinstance(f, (list, tuple)):
            tokens = [str(tok).strip().lower() for tok in f if str(tok).strip()]
            if tokens:
                out.append(tokens)
            continue
        if isinstance(f, str):
            tokens = [tok.strip().lower() for tok in f.split() if tok.strip()]
            if tokens:
                out.append(tokens)
    return out


def _entity_class(token: str) -> str:
    """
    Class name of an entity token.

    Handles the three shapes seen in practice:
      * demangled instance ``"bread 3"``      -> ``"bread"``
      * objectType/receptacleType value       -> ``"breadtype "`` -> ``"bread"``
      * mangled id ``"bread|0|3"``            -> ``"bread"``
    """
    tok = token.strip().lower()
    if not tok:
        return ""
    tok = tok.split("|", 1)[0].strip()      # mangled id -> class
    tok = tok.split(" ", 1)[0].strip()      # "bread 3"  -> "bread"
    if tok.endswith("type"):                # "breadtype" -> "bread"
        tok = tok[: -len("type")]
    return tok


def _facts_with_pred(facts: List[List[str]], pred: str) -> List[List[str]]:
    """All fact token-lists whose first token equals ``pred``."""
    return [toks for toks in facts if toks and toks[0] == pred]


class AlfworldMilestoneDetector:
    """
    Stateful, one-task detector that walks ALFWorld milestone phases.

    Usage:
        detector = AlfworldMilestoneDetector(task_type, goal_params)
        result = detector.process(facts, won)
        if result.achieved:
            print("Milestone reached:", result.phase)
    """

    def __init__(
        self,
        task_type: str,
        goal_params: Dict[str, str],
        log_dir: Optional[str] = None,
    ) -> None:
        self.task_type = task_type
        self.obj_cls = str(goal_params.get("object_target", "")).lower()
        self.parent_cls = str(goal_params.get("parent_target", "")).lower()
        self.toggle_cls = str(goal_params.get("toggle_target", "")).lower()

        self.num_milestones = compute_num_milestones(task_type)
        self.milestone_names = _MILESTONE_NAMES.get(task_type, [])

        self.log_dir = Path(log_dir) if log_dir else Path("logs/alfworld_milestone_debug")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"milestone_debug_{self.session_id}.log"
        self._log_init()

        self.reset()

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------
    def reset(self) -> None:
        self.phase = MilestonePhase.TAKE
        self.completed_count = 0
        self._completed = False
        self.step_count = 0
        self.history: List[MilestoneResult] = []
        self.action_history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def process(self, facts: Iterable[Any], won: bool = False) -> MilestoneResult:
        """Consume one environment transition and check the next milestone.

        ``won`` is retained only for call-site compatibility and is deliberately
        ignored. Progress is derived exclusively from facts, and a call can
        advance at most one milestone.
        """
        self.step_count += 1
        if self._completed:
            result = MilestoneResult(
                phase=MilestonePhase.COMPLETE, achieved=False,
                message="All milestones completed.",
            )
            self._record(result)
            return result

        result = self._advance(_parse_facts(facts))
        if result.achieved:
            self.completed_count += 1
            self._mark_milestone(result)
        self._record(result)
        return result

    # -----------------------------------------------------------------------
    # State machine dispatch
    # -----------------------------------------------------------------------
    def _advance(self, facts: List[List[str]]) -> MilestoneResult:
        if self.task_type == "pick_and_place_simple":
            return self._advance_simple(facts)
        if self.task_type in ("pick_clean_then_place_in_recep",
                              "pick_heat_then_place_in_recep",
                              "pick_cool_then_place_in_recep"):
            return self._advance_clean_heat_cool(facts)
        if self.task_type == "look_at_obj_in_light":
            return self._advance_look_at(facts)
        if self.task_type == "pick_two_obj_and_place":
            return self._advance_pick_two(facts)
        return MilestoneResult(phase=self.phase, achieved=False,
                               message=f"Unsupported task_type: {self.task_type}")

    # -----------------------------------------------------------------------
    # pick_and_place_simple: M=2
    # -----------------------------------------------------------------------
    def _advance_simple(self, facts: List[List[str]]) -> MilestoneResult:
        # milestone 0: a goal-class instance is held by the agent.
        if self.completed_count == 0 and self._holds_goal_instance(facts):
            return MilestoneResult(phase=MilestonePhase.TAKE, achieved=True,
                                   message="拿取目标物")
        # milestone 1: a goal-class instance is inside the goal parent.
        if self.completed_count == 1 and self._goal_in_parent_ids(facts):
            return MilestoneResult(phase=MilestonePhase.PLACE, achieved=True,
                                   message="目标物放入目标容器")
        return MilestoneResult(phase=self.phase, achieved=False, message="进行中")

    # -----------------------------------------------------------------------
    # pick_clean / heat / cool: M=3
    # -----------------------------------------------------------------------
    def _advance_clean_heat_cool(self, facts: List[List[str]]) -> MilestoneResult:
        state_pred = {
            "pick_clean_then_place_in_recep": _PRED["is_clean"],
            "pick_heat_then_place_in_recep": _PRED["is_hot"],
            "pick_cool_then_place_in_recep": _PRED["is_cool"],
        }[self.task_type]

        held = self._holds_goal_instance(facts)
        placed_ids = self._goal_in_parent_ids(facts)
        placed_with_state = [
            i for i in placed_ids if self._instance_has_state(facts, i, state_pred)
        ]

        if self.completed_count == 0 and held:
            return MilestoneResult(phase=MilestonePhase.TAKE, achieved=True,
                                   message="拿取目标物")
        if self.completed_count == 1 and self._some_goal_instance_has_state(facts, state_pred):
            return MilestoneResult(phase=MilestonePhase.TRANSFORM, achieved=True,
                                   message=f"目标物已{state_pred}")
        if self.completed_count == 2 and placed_with_state:
            return MilestoneResult(phase=MilestonePhase.PLACE, achieved=True,
                                   message="已{状态}目标物放入目标容器")
        return MilestoneResult(phase=self.phase, achieved=False, message="进行中")

    # -----------------------------------------------------------------------
    # look_at_obj_in_light: M=2 (independent conditions, order free)
    # -----------------------------------------------------------------------
    def _advance_look_at(self, facts: List[List[str]]) -> MilestoneResult:
        have_obj = self._holds_goal_instance(facts)
        have_toggle = self._toggle_is_on(facts)
        if self.completed_count == 0 and (have_obj or have_toggle):
            return MilestoneResult(phase=MilestonePhase.TAKE, achieved=True,
                                   message="look_at: 第一个条件成立(取物/开灯)")
        if self.completed_count == 1 and have_obj and have_toggle:
            return MilestoneResult(phase=MilestonePhase.TOGGLE, achieved=True,
                                   message="look_at: 已取目标物且已开灯")
        return MilestoneResult(phase=self.phase, achieved=False, message="进行中")

    # -----------------------------------------------------------------------
    # pick_two_obj_and_place: M=4
    # -----------------------------------------------------------------------
    def _advance_pick_two(self, facts: List[List[str]]) -> MilestoneResult:
        held = self._held_goal_instances(facts)
        placed_ids = self._goal_in_parent_ids(facts)
        unique_placed = set(placed_ids)
        # goal instances held that are NOT among those already placed
        other_held = held - unique_placed

        if self.completed_count == 0 and other_held:
            return MilestoneResult(phase=MilestonePhase.TAKE, achieved=True,
                                   message="pick_two: 获得物体1")
        if self.completed_count == 1 and len(unique_placed) >= 1:
            return MilestoneResult(phase=MilestonePhase.PLACE, achieved=True,
                                   message="pick_two: 放置物体1")
        if self.completed_count == 2 and other_held:
            return MilestoneResult(phase=MilestonePhase.TAKE, achieved=True,
                                   message="pick_two: 获得物体2(不同实例)")
        if self.completed_count == 3 and len(unique_placed) >= 2:
            return MilestoneResult(phase=MilestonePhase.PLACE, achieved=True,
                                   message="pick_two: 放置物体2")
        return MilestoneResult(phase=self.phase, achieved=False, message="进行中")

    # -----------------------------------------------------------------------
    # Fact helpers
    # -----------------------------------------------------------------------
    def _instance_of_goal_cls(self, inst: str, facts: List[List[str]]) -> bool:
        return inst in self._instances_of_cls(facts, self.obj_cls)

    def _instances_of_cls(self, facts: List[List[str]], cls: str) -> Set[str]:
        """All entity instances whose class is ``cls``.

        Primary source is the explicit ``objecttype``/``receptacletype`` facts
        (``('mug 2', 'mugtype ')``). We additionally accept any argument token
        whose own class matches, which covers entities that have no type fact in
        the current snapshot (and the mangled ids used by the unit tests).
        """
        if not cls:
            return set()
        cls = _entity_class(cls)
        insts: Set[str] = set()
        for pred in (_PRED["object_type"], _PRED["receptacle_type"]):
            for toks in _facts_with_pred(facts, pred):
                if len(toks) >= 3 and _entity_class(toks[2]) == cls:
                    insts.add(toks[1])
        for toks in facts:
            # Skip the *type value* argument of type facts (e.g. "mugtype "),
            # which names a class rather than an instance.
            skip_last = toks and toks[0] in (_PRED["object_type"], _PRED["receptacle_type"])
            args = toks[1:-1] if (skip_last and len(toks) >= 3) else toks[1:]
            for tok in args:
                if _entity_class(tok) == cls:
                    insts.add(tok)
        return insts

    def _held_goal_instances(self, facts: List[List[str]]) -> Set[str]:
        """Goal-class instances currently held by the agent."""
        held: Set[str] = set()
        for toks in _facts_with_pred(facts, _PRED["holds"]):
            # holds <agent> <object>
            if len(toks) >= 3 and self._instance_of_goal_cls(toks[2], facts):
                held.add(toks[2])
        return held

    def _holds_goal_instance(self, facts: List[List[str]]) -> bool:
        return bool(self._held_goal_instances(facts))

    def _goal_in_parent_ids(self, facts: List[List[str]]) -> List[str]:
        """Goal-class instances inside the goal parent class receptacle."""
        parents = self._instances_of_cls(facts, self.parent_cls)
        out: List[str] = []
        for toks in _facts_with_pred(facts, _PRED["in_receptacle"]):
            # inreceptacle <object> <receptacle>
            if len(toks) >= 3 and toks[2] in parents and self._instance_of_goal_cls(toks[1], facts):
                out.append(toks[1])
        return out

    def _instance_has_state(self, facts: List[List[str]], inst: str, state_pred: str) -> bool:
        for toks in _facts_with_pred(facts, state_pred):
            if len(toks) >= 2 and toks[1] == inst:
                return True
        return False

    def _some_goal_instance_has_state(self, facts: List[List[str]], state_pred: str) -> bool:
        for toks in _facts_with_pred(facts, state_pred):
            # state_pred <object>
            if len(toks) >= 2 and self._instance_of_goal_cls(toks[1], facts):
                return True
        return False

    def _toggle_is_on(self, facts: List[List[str]]) -> bool:
        togglers = self._instances_of_cls(facts, self.toggle_cls)
        for toks in _facts_with_pred(facts, _PRED["is_toggled"]):
            if len(toks) >= 2 and toks[1] in togglers:
                return True
        return False

    # -----------------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------------
    def _mark_milestone(self, result: MilestoneResult) -> None:
        if self.completed_count >= self.num_milestones:
            self._completed = True
            self.phase = MilestonePhase.COMPLETE
            return
        if result.phase == MilestonePhase.TAKE:
            # cosmetic phase label for the next in-progress milestone
            if self.task_type in ("pick_clean_then_place_in_recep",
                                    "pick_heat_then_place_in_recep",
                                    "pick_cool_then_place_in_recep"):
                self.phase = MilestonePhase.TRANSFORM
            else:
                self.phase = MilestonePhase.PLACE
        elif result.phase == MilestonePhase.TRANSFORM:
            self.phase = MilestonePhase.PLACE
        elif result.phase in (MilestonePhase.PLACE, MilestonePhase.TOGGLE):
            if self.task_type == "pick_two_obj_and_place":
                self.phase = MilestonePhase.TAKE
            # else stays PLACE / COMPLETE

    def _record(self, result: MilestoneResult) -> None:
        self.history.append(result)
        self.action_history.append({
            "step": self.step_count,
            "achieved": result.achieved,
            "phase": result.phase.name,
            "message": result.message,
            "metadata": result.metadata,
        })
        self._log_step(result)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def _log_init(self) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "w") as f:
            f.write(f"{'='*80}\nSESSION INITIALIZED: {self.session_id}\n{'='*80}\n")
            f.write(f"task_type: {self.task_type}\n"
                    f"object_target: {self.obj_cls}\n"
                    f"parent_target: {self.parent_cls}\n"
                    f"toggle_target: {self.toggle_cls}\n"
                    f"num_milestones: {self.num_milestones}\n"
                    f"{'='*80}\n\n")

    def _log_step(self, result: MilestoneResult) -> None:
        if not self.log_file:
            return
        with open(self.log_file, "a") as f:
            f.write(f"[Step {self.step_count}] {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"  phase: {result.phase.name} | achieved: {result.achieved}\n")
            f.write(f"  message: {result.message}\n")
            if result.metadata:
                f.write(f"  metadata: {result.metadata}\n")
            f.write("\n")
