#!/usr/bin/env python3
"""Train MTG bots via DQN self-play with curriculum learning."""
import argparse
import time
from pathlib import Path

from mtg.game import make_deck
from mtg.agents import RandomAgent, BoltFaceAgent, AggroGoblinAgent, MixedBoltFaceAgent
from mtg.training import DQNTrainer, NNAgent, make_nn_agent
from mtg.elo import Tournament
from mtg.enums import GameResult
from mtg.analyze import plot_training_loss, plot_elo_history


def evaluate_vs(trainer, opponent_cls, deck, n_games=200, starting_life=20):
    wins = 0
    for _ in range(n_games):
        agent0 = NNAgent(trainer.q_net, epsilon=0.0)
        agent1 = opponent_cls()
        from mtg.game import Game
        game = Game(
            decks=[list(deck), list(deck)],
            agents=[agent0, agent1],
            starting_life=starting_life,
            verbose=False,
        )
        result = game.run_game()
        if result == GameResult.WIN_0:
            wins += 1
    return wins / n_games


def train(
    starting_life: int = 20,
    deck_str: str = "20M,10B,10G",
    total_episodes: int = 50_000,
    save_dir: str = "models",
    eval_interval: int = 2000,
    checkpoint_interval: int = 5000,
):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    from play import parse_deck
    deck = parse_deck(deck_str)

    trainer = DQNTrainer(
        lr=5e-4,
        gamma=0.99,
        batch_size=256,
        target_update_freq=500,
        buffer_capacity=100_000,
        min_buffer=2000,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=total_episodes * 2 // 3,
    )

    elo_history: dict[str, list[tuple[int, float]]] = {
        "trained": [],
        "random": [(0, 1000.0)],
        "bolt_face": [(0, 1000.0)],
        "aggro_goblin": [(0, 1000.0)],
        "mixed": [(0, 1000.0)],
    }

    opponents = [
        ("random", RandomAgent),
        ("bolt_face", BoltFaceAgent),
        ("aggro_goblin", AggroGoblinAgent),
        ("mixed", MixedBoltFaceAgent),
    ]

    checkpoint_agents = []
    phase1_cutoff = total_episodes // 5
    phase2_cutoff = total_episodes * 3 // 5

    start_time = time.time()
    print(f"Training with life={starting_life}, deck={deck_str}, episodes={total_episodes}")

    for ep in range(1, total_episodes + 1):
        # Phase 1: vs random
        if ep <= phase1_cutoff:
            opp = RandomAgent()
        # Phase 2: self-play with checkpoint pool
        elif ep <= phase2_cutoff:
            if checkpoint_agents and ep % 3 != 0:
                import random as rng
                ckpt_net = rng.choice(checkpoint_agents)
                opp = NNAgent(ckpt_net, epsilon=0.05)
            else:
                opp = None  # self-play
        # Phase 3: pure self-play
        else:
            opp = None

        result = trainer.run_episode(deck, deck, opponent=opp, starting_life=starting_life)

        if ep % eval_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed

            wr_strs = []
            for name, cls in opponents:
                wr = evaluate_vs(trainer, cls, deck, n_games=100, starting_life=starting_life)
                wr_strs.append(f"{name}:{wr:.0%}")

            elo_tournament = Tournament()
            from mtg.training import NNAgent as NNAgentCls
            trained_agent = NNAgentCls(trainer.q_net, epsilon=0.0)
            elo_tournament.register("trained", trained_agent, deck)
            for name, cls in opponents:
                elo_tournament.register(name, cls(), deck)
            elo_tournament.round_robin(games_per_pair=50, starting_life=starting_life)

            for name in elo_tournament.ratings:
                elo_history.setdefault(name, []).append((ep, elo_tournament.ratings[name]))

            eps_str = f"{trainer.get_epsilon():.3f}"
            loss_str = f"{sum(trainer.losses[-100:]) / max(1, len(trainer.losses[-100:])):.4f}" if trainer.losses else "N/A"
            print(f"  Ep {ep}/{total_episodes} ({eps_per_sec:.0f}/s) | ε={eps_str} | loss={loss_str} | {' '.join(wr_strs)}")

        if ep % checkpoint_interval == 0:
            ckpt_path = save_path / f"checkpoint_{ep}.pt"
            trainer.save(str(ckpt_path))
            import copy
            ckpt_net_copy = type(trainer.q_net)()
            ckpt_net_copy.load_state_dict(copy.deepcopy(trainer.q_net.state_dict()))
            checkpoint_agents.append(ckpt_net_copy)
            if len(checkpoint_agents) > 5:
                checkpoint_agents.pop(0)

    trainer.save(str(save_path / "final.pt"))
    print(f"\nTraining complete. Saved to {save_path / 'final.pt'}")

    plot_training_loss(trainer.losses, "training_loss.png")
    plot_elo_history(elo_history, "elo_history.png")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train MTG DQN agent")
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--deck", default="20M,10B,10G")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    args = parser.parse_args()

    train(
        starting_life=args.life,
        deck_str=args.deck,
        total_episodes=args.episodes,
        save_dir=args.save_dir,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
