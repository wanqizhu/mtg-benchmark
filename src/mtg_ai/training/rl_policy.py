from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from mtg_ai.bots.base import BotPolicy
from mtg_ai.core.actions import CastSpellAction, DeclareAttackersAction, DeclareBlockersAction, GameAction, PassPriorityAction, PlayLandAction
from mtg_ai.core.state import GameState
from mtg_ai.training.obs import encode_action, encode_state


@dataclass
class TabularQPolicy(BotPolicy):
    name: str = "tabular_q"
    q_values: dict[str, dict[str, float]] = field(default_factory=dict)
    epsilon: float = 0.1

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        actor = legal_actions[0].actor_id
        state_key = repr(encode_state(state, actor))
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        if rng.random() < self.epsilon:
            return rng.choice(legal_actions)
        return self.greedy_action(state, legal_actions)

    def greedy_action(self, state: GameState, legal_actions: list[GameAction]) -> GameAction:
        actor = legal_actions[0].actor_id
        state_key = repr(encode_state(state, actor))
        q_for_state = self.q_values.setdefault(state_key, {})

        best_action = legal_actions[0]
        best_value = float("-inf")
        for action in legal_actions:
            action_key = encode_action(action, actor)
            value = q_for_state.get(action_key, 0.0)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def update(self, state_key: str, action_key: str, target: float, alpha: float) -> None:
        q_for_state = self.q_values.setdefault(state_key, {})
        previous = q_for_state.get(action_key, 0.0)
        q_for_state[action_key] = previous + alpha * (target - previous)

    def save(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "epsilon": self.epsilon,
            "q_values": self.q_values,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "TabularQPolicy":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(name=payload["name"], epsilon=payload.get("epsilon", 0.0), q_values=payload["q_values"])


class GreedyTabularBot(BotPolicy):
    def __init__(self, policy: TabularQPolicy, name: str = "trained_greedy") -> None:
        self.policy = policy
        self.name = name

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        return self.policy.greedy_action(state, legal_actions)


@dataclass
class WeightedHeuristicPolicy(BotPolicy):
    name: str = "weighted_heuristic"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "play_land": 4.0,
            "cast_goblin": 3.0,
            "cast_bolt_face": 2.0,
            "cast_bolt_face_lethal": 30.0,
            "cast_bolt_creature": 4.0,
            "cast_bolt_creature_threat": 2.0,
            "attack_per_creature": 4.0,
            "attack_lethal_bonus": 25.0,
            "block_per_assignment": 1.5,
            "block_save_life": 3.0,
            "pass_penalty": -2.5,
        }
    )

    def choose_action(self, state: GameState, legal_actions: list[GameAction], rng: random.Random) -> GameAction:
        actor = legal_actions[0].actor_id
        opponent = 1 - actor
        best_action = legal_actions[0]
        best_score = float("-inf")
        for action in legal_actions:
            score = self.score_action(state, action, actor, opponent)
            if score > best_score:
                best_action = action
                best_score = score
        return best_action

    def score_action(self, state: GameState, action: GameAction, actor: int, opponent: int) -> float:
        score = 0.0
        if isinstance(action, PlayLandAction):
            score += self.weights["play_land"]
        elif isinstance(action, CastSpellAction):
            if len(action.targets) == 0:
                score += self.weights["cast_goblin"]
            else:
                target = action.targets[0]
                if target.kind == "player":
                    score += self.weights["cast_bolt_face"]
                    if state.players[opponent].life <= 3:
                        score += self.weights["cast_bolt_face_lethal"]
                else:
                    score += self.weights["cast_bolt_creature"]
                    permanent = state.permanents.get(str(target.id))
                    if permanent is not None:
                        definition = state.card_instances[permanent.card_id].definition_name
                        if definition == "Raging Goblin":
                            score += self.weights["cast_bolt_creature_threat"] * 1.0
        elif isinstance(action, DeclareAttackersAction):
            attack_count = len(action.attacker_ids)
            score += self.weights["attack_per_creature"] * attack_count
            if attack_count >= state.players[opponent].life:
                score += self.weights["attack_lethal_bonus"]
        elif isinstance(action, DeclareBlockersAction):
            block_count = len(action.blocks)
            score += self.weights["block_per_assignment"] * block_count
            if state.players[actor].life <= block_count:
                score += self.weights["block_save_life"] * block_count
        elif isinstance(action, PassPriorityAction):
            score += self.weights["pass_penalty"]
        return score

    def mutate(self, rng: random.Random, sigma: float) -> "WeightedHeuristicPolicy":
        new_weights = {}
        for key, value in self.weights.items():
            new_weights[key] = value + rng.gauss(0.0, sigma)
        return WeightedHeuristicPolicy(name=self.name, weights=new_weights)

    def save(self, path: str | Path) -> None:
        payload = {"name": self.name, "policy_type": "weighted_heuristic", "weights": self.weights}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "WeightedHeuristicPolicy":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(name=payload.get("name", "weighted_heuristic"), weights=payload["weights"])

    @classmethod
    def blend(
        cls,
        policy_a: "WeightedHeuristicPolicy",
        policy_b: "WeightedHeuristicPolicy",
        alpha: float,
        name: str = "blended_weighted_heuristic",
    ) -> "WeightedHeuristicPolicy":
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be in [0, 1].")
        keys = set(policy_a.weights) | set(policy_b.weights)
        blended: dict[str, float] = {}
        for key in keys:
            value_a = policy_a.weights.get(key, 0.0)
            value_b = policy_b.weights.get(key, 0.0)
            blended[key] = alpha * value_a + (1.0 - alpha) * value_b
        return cls(name=name, weights=blended)

