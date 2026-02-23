from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass

from mtg_ai.bots.fixed_bots import BoltFaceBot, GoblinAggroBot, MixedBoltFaceBot
from mtg_ai.bots.random_bot import RandomBot
from mtg_ai.core.actions import GameAction
from mtg_ai.core.rules_engine import GameConfig, RulesEngine
from mtg_ai.core.serialization import action_to_dict, serialize_state
from mtg_ai.game.session import GameSession
from mtg_ai.training.obs import encode_action
from mtg_ai.training.rl_policy import WeightedHeuristicPolicy


def build_deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return ["Mountain"] * mountains + ["Lightning Bolt"] * bolts + ["Raging Goblin"] * goblins


class DeterministicMinimaxSolver:
    def __init__(self, engine: RulesEngine, max_depth: int = 8) -> None:
        self.engine = engine
        self.max_depth = max_depth
        self._memo: dict[tuple[str, int, int], float] = {}

    def best_action(self, state, player_id: int):
        values = self.action_values(state, player_id)
        if not values:
            return None, 0.0
        best_action, best_value = max(values, key=lambda item: item[1])
        return best_action, best_value

    def action_values(self, state, player_id: int) -> list[tuple[GameAction, float]]:
        legal_actions = self.engine.legal_actions(state, player_id)
        if not legal_actions:
            return []
        values: list[tuple[GameAction, float]] = []
        for action in legal_actions:
            engine_copy = copy.deepcopy(self.engine)
            state_copy = copy.deepcopy(state)
            engine_copy.apply_action(state_copy, action)
            value = self._value(engine_copy, state_copy, maximizing_player=player_id, depth=self.max_depth - 1)
            values.append((action, value))
        return values

    def _value(self, engine: RulesEngine, state, maximizing_player: int, depth: int) -> float:
        if state.game_over:
            if state.winner is None:
                return 0.0
            return 1.0 if state.winner == maximizing_player else -1.0
        if depth <= 0:
            opponent = 1 - maximizing_player
            return (state.players[maximizing_player].life - state.players[opponent].life) / 20.0

        state_key = repr(serialize_state(state))
        cache_key = (state_key, maximizing_player, depth)
        if cache_key in self._memo:
            return self._memo[cache_key]

        actor = state.turn.priority_player_id
        legal_actions = engine.legal_actions(state, actor)
        if not legal_actions:
            return 0.0
        values: list[float] = []
        for action in legal_actions:
            engine_copy = copy.deepcopy(engine)
            state_copy = copy.deepcopy(state)
            engine_copy.apply_action(state_copy, action)
            values.append(self._value(engine_copy, state_copy, maximizing_player, depth - 1))
        if actor == maximizing_player:
            value = max(values)
        else:
            value = min(values)
        self._memo[cache_key] = value
        return value


@dataclass
class SolverAlignmentResult:
    compared_states: int
    trained_agreement: float
    mixed_agreement: float
    bolt_agreement: float
    goblin_agreement: float


def _collect_states_for_solver(seed: int, sample_count: int, life: int, opening_hand: int) -> list[tuple]:
    deck = build_deck(4, 4, 4)
    sampled_states: list[tuple] = []
    session_seed = seed
    while len(sampled_states) < sample_count:
        session = GameSession(
            decks={0: list(deck), 1: list(deck)},
            controllers={0: RandomBot(), 1: RandomBot()},
            config=GameConfig(starting_life=life, opening_hand_size=opening_hand, random_seed=session_seed),
            seed=session_seed,
        )
        for _ in range(220):
            if session.state.game_over or len(sampled_states) >= sample_count:
                break
            actor = session.state.turn.priority_player_id
            legal = session.engine.legal_actions(session.state, actor)
            if actor == 0 and len(legal) > 1:
                sampled_states.append((copy.deepcopy(session.state), [copy.deepcopy(action) for action in legal]))
            session.step()
        session_seed += 101
        if session_seed > seed + 2000:
            break
    return sampled_states


def evaluate_solver_alignment(
    trained_policy: WeightedHeuristicPolicy,
    seed: int = 3,
    life: int = 5,
    opening_hand: int = 3,
    depth: int = 6,
    sample_count: int = 8,
) -> SolverAlignmentResult:
    sampled_states = _collect_states_for_solver(seed=seed, sample_count=sample_count, life=life, opening_hand=opening_hand)
    trained_matches = 0
    mixed_matches = 0
    bolt_matches = 0
    goblin_matches = 0

    mixed = MixedBoltFaceBot()
    bolt = BoltFaceBot()
    goblin = GoblinAggroBot()
    rng = random.Random(seed)

    for state, _ in sampled_states:
        engine = RulesEngine(config=GameConfig(starting_life=life, opening_hand_size=opening_hand, random_seed=seed))
        solver = DeterministicMinimaxSolver(engine, max_depth=depth)
        actor = state.turn.priority_player_id
        values = solver.action_values(state, actor)
        if not values:
            continue
        best_value = max(value for _, value in values)
        optimal_actions = {
            encode_action(action, actor)
            for action, value in values
            if abs(value - best_value) < 1e-6
        }
        legal = [action for action, _ in values]
        trained_choice = encode_action(trained_policy.choose_action(state, legal, rng), actor)
        mixed_choice = encode_action(mixed.choose_action(state, legal, rng), actor)
        bolt_choice = encode_action(bolt.choose_action(state, legal, rng), actor)
        goblin_choice = encode_action(goblin.choose_action(state, legal, rng), actor)

        if trained_choice in optimal_actions:
            trained_matches += 1
        if mixed_choice in optimal_actions:
            mixed_matches += 1
        if bolt_choice in optimal_actions:
            bolt_matches += 1
        if goblin_choice in optimal_actions:
            goblin_matches += 1

    total = max(len(sampled_states), 1)
    return SolverAlignmentResult(
        compared_states=len(sampled_states),
        trained_agreement=trained_matches / total,
        mixed_agreement=mixed_matches / total,
        bolt_agreement=bolt_matches / total,
        goblin_agreement=goblin_matches / total,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic minimax on low-life settings for validation.")
    parser.add_argument("--life", type=int, default=5)
    parser.add_argument("--opening-hand", type=int, default=3)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--policy-path", default="artifacts/training/self_play/policy_final.json")
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    engine = RulesEngine(config=GameConfig(starting_life=args.life, opening_hand_size=args.opening_hand, random_seed=args.seed))
    deck_a = build_deck(3, 3, 3)
    deck_b = build_deck(3, 3, 3)
    state, _ = engine.create_game(decks={0: deck_a, 1: deck_b}, preserve_deck_order=True, starting_player_id=0)
    solver = DeterministicMinimaxSolver(engine, max_depth=args.depth)
    action, value = solver.best_action(state, player_id=state.turn.priority_player_id)
    print(f"Best opening action value={value:.3f}: {action_to_dict(action) if action else None}")

    bolt_bot = BoltFaceBot()
    mixed_bot = MixedBoltFaceBot()
    legal = engine.legal_actions(state, state.turn.priority_player_id)
    bolt_choice = bolt_bot.choose_action(state, legal, random.Random(args.seed))
    mixed_choice = mixed_bot.choose_action(state, legal, random.Random(args.seed + 1))
    print(f"BoltFace opening action: {encode_action(bolt_choice, state.turn.priority_player_id)}")
    print(f"Mixed opening action: {encode_action(mixed_choice, state.turn.priority_player_id)}")

    trained_policy = WeightedHeuristicPolicy.load(args.policy_path)
    alignment = evaluate_solver_alignment(
        trained_policy=trained_policy,
        seed=args.seed,
        life=args.life,
        opening_hand=args.opening_hand,
        depth=args.depth,
        sample_count=args.sample_count,
    )
    print(
        "Solver alignment "
        f"(states={alignment.compared_states}): "
        f"trained={alignment.trained_agreement:.3f} "
        f"mixed={alignment.mixed_agreement:.3f} "
        f"bolt={alignment.bolt_agreement:.3f} "
        f"goblin={alignment.goblin_agreement:.3f}"
    )


if __name__ == "__main__":
    main()

