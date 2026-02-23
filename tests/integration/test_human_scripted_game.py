from __future__ import annotations

from mtg_ai.core.actions import PassPriorityAction
from mtg_ai.core.rules_engine import GameConfig
from mtg_ai.game.session import GameSession


def _first_non_pass(state, legal_actions):
    for action in legal_actions:
        if not isinstance(action, PassPriorityAction):
            return action
    return legal_actions[0]


def test_scripted_human_like_game_completes_and_logs_replay(tmp_path) -> None:
    deck = ["Mountain"] * 10 + ["Lightning Bolt"] * 10 + ["Raging Goblin"] * 10
    session = GameSession(
        decks={0: list(deck), 1: list(deck)},
        controllers={0: _first_non_pass, 1: _first_non_pass},
        config=GameConfig(starting_life=10, opening_hand_size=7, random_seed=9),
        seed=9,
    )
    result = session.run(max_actions=120)
    assert result.result is not None
    replay_path = tmp_path / "scripted_replay.jsonl"
    session.replay.save_jsonl(replay_path)
    assert replay_path.exists()

