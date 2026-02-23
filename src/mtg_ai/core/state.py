from __future__ import annotations

from dataclasses import dataclass, field

from mtg_ai.core.actions import TargetRef
from mtg_ai.core.enums import Step


@dataclass
class PlayerState:
    player_id: int
    life: int
    library: list[str]
    hand: list[str] = field(default_factory=list)
    graveyard: list[str] = field(default_factory=list)
    exile: list[str] = field(default_factory=list)
    battlefield: list[str] = field(default_factory=list)
    mana_pool_red: int = 0
    lands_played_this_turn: int = 0
    mulligans_taken: int = 0


@dataclass
class PermanentState:
    permanent_id: str
    card_id: str
    controller_id: int
    tapped: bool = False
    damage_marked: int = 0
    entered_turn_number: int = 0


@dataclass
class StackObject:
    stack_id: str
    card_id: str
    controller_id: int
    targets: tuple[TargetRef, ...]


@dataclass
class CombatState:
    attackers: list[str] = field(default_factory=list)
    blocks: dict[str, list[str]] = field(default_factory=dict)
    defending_player_id: int | None = None


@dataclass
class TurnState:
    turn_number: int = 1
    active_player_id: int = 0
    priority_player_id: int = 0
    step: Step = Step.UNTAP
    starting_player_id: int = 0
    passed_priority_in_row: int = 0


@dataclass
class GameState:
    players: dict[int, PlayerState]
    card_instances: dict[str, "CardInstance"]
    permanents: dict[str, PermanentState]
    stack: list[StackObject]
    turn: TurnState
    combat: CombatState = field(default_factory=CombatState)
    game_over: bool = False
    winner: int | None = None
    result: str | None = None
    priority_waiting_for: int | None = None
    pending_draw_loss: set[int] = field(default_factory=set)
    event_index: int = 0

