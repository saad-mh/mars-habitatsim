"""Free-text nav-command decomposition into an ordered sequence of sub-goals,
consumed by RoverController's mission stepper (see rover_controller.py's
`_start_mission_subgoal`) to drive nav/gui.py's Command panel end to end.

Splitting the raw instruction is delegated to the Qwen VLM
(`sam_vla.vlm.qwen_client.parse_nav_command`, see
`qwen_prompts.build_parse_nav_command_prompt`) rather than done here --
that call needs a camera frame and is slow, so RoverController runs it on a
background thread (`_dispatch_nav_command`/`_nav_command_worker`) and hands
this module only the resulting `(directions, goals)` two-list split. This
module's job is turning that split into an ordered `SubGoal` sequence:
`parse_parts` below, wrapped by `Mission`.

`reached_current_goal`-style green-pixel/VLM heuristics from the original
standalone prototype (`check.py`, kept for reference at the repo root) were
dropped rather than ported -- RoverController already has a strictly better
reached-goal signal (ground-truth distance for MODE_POINT,
BeliefGoalTracker.distance() for MODE_RESOLVE, see its `display.goal_reached`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class GoalKind(Enum):
    GO_TO = auto()  # navigate to a landmark (open-vocabulary Qwen grounding)
    RETURN = auto()  # return to the spawn/home point
    TURN = auto()  # in-place rotation
    FIND = auto()  # same handling as GO_TO -- search/scan until a target appears
    DONE = auto()


@dataclass
class SubGoal:
    kind: GoalKind
    target: str  # e.g. "flag", "home station", "right"
    raw: str  # the original phrase, for status display


# Direction words (qwen_prompts.build_parse_nav_command_prompt's closed
# vocabulary) that need an in-place turn -- "front" is deliberately excluded,
# the rover is already facing that way. The actual turn-heading-degrees
# mapping lives once, in rover_controller.py's MODE_TURN handling.
_TURN_DIRECTIONS = {"left", "right", "back"}

# Goal phrases naming the spawn point rather than a new landmark -- e.g.
# "home base", "the spawn point", "go back" -- become RETURN, not GO_TO.
_RETURN_WORDS = re.compile(r"\b(home|base|spawn)\b", re.I)


def _direction_subgoal(direction: str) -> Optional[SubGoal]:
    """One VLM-emitted direction word -> an in-place TURN sub-goal, or None
    for "front" (the rover is already facing that way -- nothing to turn)."""
    if direction not in _TURN_DIRECTIONS:
        return None
    return SubGoal(GoalKind.TURN, direction, direction)


def _goal_subgoal(goal: str) -> SubGoal:
    kind = GoalKind.RETURN if _RETURN_WORDS.search(goal) else GoalKind.GO_TO
    return SubGoal(kind, goal, goal)


def parse_parts(directions: List[str], goals: List[str]) -> List[SubGoal]:
    """Turn the VLM's (directions, goals) split into an ordered sub-goal
    sequence, in two parts: every direction first (each an in-place TURN),
    then every goal (GO_TO, or RETURN for phrases naming home/base/spawn).
    Each part preserves the VLM's own within-list ordering (see
    qwen_prompts.build_parse_nav_command_prompt), but the VLM never orders
    directions relative to goals across the two lists -- "part 1 then part
    2" is the closest ordering available, not a reconstruction of the
    original interleaving."""
    steps: List[SubGoal] = []
    for direction in directions:
        sub = _direction_subgoal(direction)
        if sub is not None:
            steps.append(sub)
    for goal in goals:
        steps.append(_goal_subgoal(goal))
    steps.append(SubGoal(GoalKind.DONE, "", "mission complete"))
    return steps


@dataclass
class Mission:
    instruction: str
    directions: List[str] = field(default_factory=list)
    goal_texts: List[str] = field(default_factory=list)
    goals: List[SubGoal] = field(default_factory=list)
    idx: int = 0

    def __post_init__(self):
        if not self.goals:
            self.goals = parse_parts(self.directions, self.goal_texts)

    @property
    def current(self) -> SubGoal:
        return self.goals[self.idx]

    @property
    def finished(self) -> bool:
        return self.current.kind == GoalKind.DONE

    def status(self) -> str:
        cur = self.current
        return (
            f"[step {self.idx + 1}/{len(self.goals)}] "
            f"{cur.kind.name} -> '{cur.target}'  ({cur.raw!r})"
        )

    def advance(self):
        if not self.finished:
            self.idx += 1
