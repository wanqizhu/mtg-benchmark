from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from mtg_ai.bots.base import BotPolicy
from mtg_ai.cards.registry import CardRegistry
from mtg_ai.core.actions import GameAction
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from mtg_ai.core.state import GameState
from mtg_ai.replay.logging import ReplayLogger


DecisionFn = Callable[[GameState, list[GameAction]], GameAction]


@dataclass(frozen=True)
class SessionResult:
    winner: int | None
    result: str
    turns: int
    actions: int


class GameSession:
    def __init__(
        self,
        decks: dict[int, list[str]],
        controllers: dict[int, BotPolicy | DecisionFn],
        config: GameConfig | None = None,
        seed: int = 0,
        preserve_deck_order: bool = False,
    ) -> None:
        self.registry = CardRegistry()
        self.engine = RulesEngine(registry=self.registry, config=config or GameConfig(random_seed=seed))
        self.state, setup_events = self.engine.create_game(decks=decks, preserve_deck_order=preserve_deck_order)
        self.controllers = controllers
        self.rng = random.Random(seed)
        self.action_count = 0
        self.replay = ReplayLogger()
        self.replay.record_events(setup_events)
        self.replay.record_state(self.state, note="initial")

    def step(self) -> bool:
        if self.state.game_over:
            return False
        actor = self.state.turn.priority_player_id
        legal_actions = self.engine.legal_actions(self.state, actor)
        if not legal_actions:
            raise ValueError(f"No legal actions available for player {actor}.")
        controller = self.controllers[actor]
        if isinstance(controller, BotPolicy):  # type: ignore[arg-type]
            action = controller.choose_action(self.state, legal_actions, self.rng)
        else:
            action = controller(self.state, legal_actions)
        if action not in legal_actions:
            raise ValueError(f"Controller for player {actor} returned illegal action: {action}")
        self.action_count += 1
        self.replay.record_action_obj(action)
        events = self.engine.apply_action(self.state, action)
        self.replay.record_events(events)
        self.replay.record_state(self.state, note=f"after_{self.action_count}")
        return not self.state.game_over

    def run(self, max_actions: int = 300) -> SessionResult:
        while self.action_count < max_actions and not self.state.game_over:
            self.step()
        if not self.state.game_over:
            self.state.game_over = True
            self.state.winner = None
            self.state.result = "draw_max_actions"
        return SessionResult(
            winner=self.state.winner,
            result=self.state.result or "draw",
            turns=self.state.turn.turn_number,
            actions=self.action_count,
        )

