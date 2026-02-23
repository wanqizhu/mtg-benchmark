#!/usr/bin/env python3
"""Run ELO tournament between various agents."""
import argparse
from pathlib import Path
from mtg.game import make_deck
from mtg.agents import RandomAgent, BoltFaceAgent, AggroGoblinAgent, MixedBoltFaceAgent
from mtg.training import make_nn_agent
from mtg.elo import Tournament
from mtg.analyze import plot_elo_history, plot_win_matrix
from play import parse_deck


def main():
    parser = argparse.ArgumentParser(description="Run MTG tournament")
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--deck", default="20M,10B,10G")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    deck = parse_deck(args.deck)
    model_dir = Path(args.model_dir)

    tournament = Tournament()
    tournament.register("random", RandomAgent(), deck)
    tournament.register("bolt_face", BoltFaceAgent(), deck)
    tournament.register("aggro_goblin", AggroGoblinAgent(), deck)
    tournament.register("mixed", MixedBoltFaceAgent(), deck)

    # Load trained checkpoints
    checkpoints = sorted(model_dir.glob("checkpoint_*.pt"))
    for ckpt_path in checkpoints:
        name = ckpt_path.stem
        agent = make_nn_agent(str(ckpt_path))
        tournament.register(name, agent, deck)

    final_path = model_dir / "final.pt"
    if final_path.exists():
        agent = make_nn_agent(str(final_path))
        tournament.register("trained_final", agent, deck)

    print(f"Tournament: {len(tournament.agents)} agents, {args.games} games per pair, life={args.life}")
    tournament.round_robin(games_per_pair=args.games, starting_life=args.life)

    report = tournament.report()
    print(report)

    names = sorted(tournament.ratings.keys(), key=lambda n: -tournament.ratings[n])
    plot_win_matrix(names, tournament.win_matrix, tournament.game_matrix, "tournament_win_matrix.png")

    # Build elo snapshot from history
    elo_snapshots: dict[str, list[tuple[int, float]]] = {}
    game_counts: dict[str, int] = {n: 0 for n in tournament.agents}
    for mr in tournament.history:
        game_counts[mr.agent_a] = game_counts.get(mr.agent_a, 0) + 1
        game_counts[mr.agent_b] = game_counts.get(mr.agent_b, 0) + 1
        elo_snapshots.setdefault(mr.agent_a, []).append((game_counts[mr.agent_a], mr.elo_a_after))
        elo_snapshots.setdefault(mr.agent_b, []).append((game_counts[mr.agent_b], mr.elo_b_after))

    plot_elo_history(elo_snapshots, "tournament_elo.png", "ELO During Tournament")

    with open("results/tournament_report.txt", "w") as f:
        f.write(report)
    print("\nReport saved to results/tournament_report.txt")


if __name__ == "__main__":
    main()
