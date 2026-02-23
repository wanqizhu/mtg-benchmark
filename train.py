#!/usr/bin/env python3
"""Train MTG bots via DQN self-play with curriculum learning."""
import argparse
import copy
import random as rng
import time
from pathlib import Path

from mtg.game import Game, make_deck
from mtg.agents import RandomAgent, BoltFaceAgent, AggroGoblinAgent, MixedBoltFaceAgent
from mtg.training import DQNTrainer, NNAgent
from mtg.elo import Tournament
from mtg.enums import GameResult
from mtg.analyze import plot_training_loss, plot_elo_history


FIXED_OPPONENTS = [
    ("random", RandomAgent),
    ("bolt_face", BoltFaceAgent),
    ("aggro_goblin", AggroGoblinAgent),
    ("mixed", MixedBoltFaceAgent),
]


def evaluate_vs(trainer, opponent_cls, deck, n_games=200, starting_life=20):
    wins = 0
    for _ in range(n_games):
        agent0 = NNAgent(trainer.q_net, epsilon=0.0)
        agent1 = opponent_cls()
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
    pretrained: str | None = None,
):
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    from play import parse_deck
    deck = parse_deck(deck_str)

    trainer = DQNTrainer(
        lr=3e-4,
        gamma=0.99,
        batch_size=256,
        target_update_freq=1000,
        buffer_capacity=200_000,
        min_buffer=3000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=total_episodes * 3 // 4,
    )

    if pretrained:
        trainer.load(pretrained)
        trainer.episodes = 0
        trainer.train_steps = 0
        trainer.losses = []
        print(f"Loaded pretrained model from {pretrained}")

    elo_history: dict[str, list[tuple[int, float]]] = {}
    checkpoint_nets = []

    start_time = time.time()
    print(f"Training with life={starting_life}, deck={deck_str}, episodes={total_episodes}")

    for ep in range(1, total_episodes + 1):
        # Training strategy: heavy focus on beating fixed bots + self-play for generalization
        r = rng.random()
        if r < 0.5:
            opp_name, opp_cls = rng.choice(FIXED_OPPONENTS)
            opp = opp_cls()
        elif r < 0.7 and checkpoint_nets:
            ckpt_net = rng.choice(checkpoint_nets)
            opp = NNAgent(ckpt_net, epsilon=0.05)
        else:
            opp = None  # self-play

        result = trainer.run_episode(deck, deck, opponent=opp, starting_life=starting_life)

        if ep % eval_interval == 0:
            elapsed = time.time() - start_time
            eps_per_sec = ep / elapsed

            wr_strs = []
            for name, cls in FIXED_OPPONENTS:
                wr = evaluate_vs(trainer, cls, deck, n_games=100, starting_life=starting_life)
                wr_strs.append(f"{name}:{wr:.0%}")

            # Quick ELO measurement
            tournament = Tournament()
            trained_agent = NNAgent(trainer.q_net, epsilon=0.0)
            tournament.register("trained", trained_agent, deck)
            for name, cls in FIXED_OPPONENTS:
                tournament.register(name, cls(), deck)
            tournament.round_robin(games_per_pair=50, starting_life=starting_life)

            for name in tournament.ratings:
                elo_history.setdefault(name, []).append((ep, tournament.ratings[name]))

            eps_str = f"{trainer.get_epsilon():.3f}"
            loss_str = f"{sum(trainer.losses[-100:]) / max(1, min(100, len(trainer.losses))):.4f}" if trainer.losses else "N/A"
            trained_elo = tournament.ratings.get("trained", 1000)
            print(f"  Ep {ep:>6}/{total_episodes} ({eps_per_sec:.0f}/s) ε={eps_str} loss={loss_str} ELO={trained_elo:.0f} | {' '.join(wr_strs)}")

        if ep % checkpoint_interval == 0:
            ckpt_path = save_path / f"checkpoint_{ep}.pt"
            trainer.save(str(ckpt_path))
            ckpt_copy = type(trainer.q_net)()
            ckpt_copy.load_state_dict(copy.deepcopy(trainer.q_net.state_dict()))
            checkpoint_nets.append(ckpt_copy)
            if len(checkpoint_nets) > 8:
                checkpoint_nets.pop(0)

    trainer.save(str(save_path / "final.pt"))
    print(f"\nTraining complete. Saved to {save_path / 'final.pt'}")

    plot_training_loss(trainer.losses, f"training_loss_life{starting_life}.png")
    plot_elo_history(elo_history, f"elo_history_life{starting_life}.png",
                     f"ELO History (Life={starting_life})")

    return trainer


def curriculum_train():
    """Curriculum: low life → high life, transferring weights."""
    curriculum = [
        (3, "12M,12B,6G", 15_000),
        (5, "15M,10B,5G", 20_000),
        (7, "15M,10B,5G", 20_000),
        (10, "18M,10B,7G", 25_000),
        (20, "20M,10B,10G", 40_000),
    ]

    pretrained = None
    for life, deck, episodes in curriculum:
        save_dir = f"models/life{life}"
        print(f"\n{'='*60}")
        print(f"  CURRICULUM: Life={life}, Deck={deck}, Episodes={episodes}")
        print(f"{'='*60}")
        trainer = train(
            starting_life=life,
            deck_str=deck,
            total_episodes=episodes,
            save_dir=save_dir,
            eval_interval=max(2000, episodes // 10),
            checkpoint_interval=max(5000, episodes // 4),
            pretrained=pretrained,
        )
        pretrained = f"{save_dir}/final.pt"


def main():
    parser = argparse.ArgumentParser(description="Train MTG DQN agent")
    parser.add_argument("--life", type=int, default=20)
    parser.add_argument("--deck", default="20M,10B,10G")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--curriculum", action="store_true",
                        help="Run curriculum training (low→high life)")
    args = parser.parse_args()

    if args.curriculum:
        curriculum_train()
    else:
        train(
            starting_life=args.life,
            deck_str=args.deck,
            total_episodes=args.episodes,
            save_dir=args.save_dir,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            pretrained=args.pretrained,
        )


if __name__ == "__main__":
    main()
