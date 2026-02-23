from __future__ import annotations
from .game import Game, make_deck
from .agents import Agent
from .enums import GameResult


def evaluate_deck(
    agent_cls,
    agent_kwargs: dict,
    opponent_cls,
    opponent_kwargs: dict,
    deck: list[str],
    opp_deck: list[str],
    starting_life: int = 20,
    num_games: int = 200,
) -> float:
    wins = 0
    for _ in range(num_games):
        a = agent_cls(**agent_kwargs) if agent_kwargs else agent_cls()
        b = opponent_cls(**opponent_kwargs) if opponent_kwargs else opponent_cls()
        game = Game(
            decks=[list(deck), list(opp_deck)],
            agents=[a, b],
            starting_life=starting_life,
            verbose=False,
        )
        result = game.run_game()
        if result == GameResult.WIN_0:
            wins += 1
    return wins / num_games


def optimize_deck(
    agent_cls,
    agent_kwargs: dict,
    opponent_cls,
    opponent_kwargs: dict,
    starting_life: int = 20,
    min_deck_size: int = 15,
    max_deck_size: int = 60,
    step: int = 5,
    games_per_deck: int = 200,
    top_k: int = 10,
) -> list[tuple[dict, float]]:
    results = []
    candidates = []

    for total in range(min_deck_size, max_deck_size + 1, step):
        for n_mountains in range(max(1, total // 5), min(total, total * 4 // 5) + 1):
            remaining = total - n_mountains
            for n_bolts in range(0, remaining + 1):
                n_goblins = remaining - n_bolts
                candidates.append({
                    "mountains": n_mountains,
                    "bolts": n_bolts,
                    "goblins": n_goblins,
                    "total": total,
                })

    # Phase 1: quick screen with fewer games
    screen_results = []
    for comp in candidates:
        deck = make_deck(comp["mountains"], comp["bolts"], comp["goblins"])
        wr = evaluate_deck(
            agent_cls, agent_kwargs,
            opponent_cls, opponent_kwargs,
            deck, deck, starting_life,
            num_games=max(20, games_per_deck // 10),
        )
        screen_results.append((comp, wr))

    screen_results.sort(key=lambda x: -x[1])
    top_candidates = screen_results[:top_k * 3]

    # Phase 2: full evaluation of top candidates
    for comp, _ in top_candidates:
        deck = make_deck(comp["mountains"], comp["bolts"], comp["goblins"])
        wr = evaluate_deck(
            agent_cls, agent_kwargs,
            opponent_cls, opponent_kwargs,
            deck, deck, starting_life,
            num_games=games_per_deck,
        )
        results.append((comp, wr))

    results.sort(key=lambda x: -x[1])
    return results[:top_k]
