from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")


def ensure_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


def plot_elo_history(
    elo_snapshots: dict[str, list[tuple[int, float]]],
    filename: str = "elo_history.png",
    title: str = "ELO Rating Over Training",
):
    ensure_dir()
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, points in elo_snapshots.items():
        episodes = [p[0] for p in points]
        elos = [p[1] for p in points]
        ax.plot(episodes, elos, label=name, marker="o", markersize=3)
    ax.set_xlabel("Games Played")
    ax.set_ylabel("ELO Rating")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / filename}")


def plot_win_matrix(
    names: list[str],
    win_matrix: dict[str, dict[str, int]],
    game_matrix: dict[str, dict[str, int]],
    filename: str = "win_matrix.png",
    title: str = "Win Rate Matrix (%)",
):
    ensure_dir()
    n = len(names)
    data = np.zeros((n, n))
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                data[i][j] = 50.0
            else:
                wins = win_matrix.get(na, {}).get(nb, 0)
                games = game_matrix.get(na, {}).get(nb, 0)
                data[i][j] = 100.0 * wins / games if games > 0 else 50.0

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(6, n * 1.2)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [name[:15] for name in names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{data[i][j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent (win %)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / filename}")


def plot_training_loss(
    losses: list[float],
    filename: str = "training_loss.png",
    window: int = 100,
):
    ensure_dir()
    if len(losses) < window:
        return
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(smoothed)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("DQN Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / filename}")


def plot_deck_optimization(
    results_by_life: dict[int, list[tuple[dict, float]]],
    filename: str = "optimal_decks.png",
):
    ensure_dir()
    life_totals = sorted(results_by_life.keys())
    fig, axes = plt.subplots(1, len(life_totals), figsize=(5 * len(life_totals), 5))
    if len(life_totals) == 1:
        axes = [axes]

    for ax, life in zip(axes, life_totals):
        if not results_by_life[life]:
            continue
        top = results_by_life[life][0]
        comp = top[0]
        total = comp["total"]
        fracs = [comp["mountains"] / total, comp["bolts"] / total, comp["goblins"] / total]
        labels = ["Mountain", "Bolt", "Goblin"]
        colors = ["#8B4513", "#FF4444", "#44AA44"]
        bars = ax.bar(labels, fracs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_title(f"Life={life}\n{comp['mountains']}M/{comp['bolts']}B/{comp['goblins']}G\nWR={top[1]*100:.1f}%")
        ax.set_ylabel("Fraction of Deck")
        for bar, frac in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{frac:.0%}", ha="center", fontsize=9)

    fig.suptitle("Optimal Deck Composition by Starting Life", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / filename}")


def plot_game_length_distribution(
    lengths_by_matchup: dict[str, list[int]],
    filename: str = "game_lengths.png",
):
    ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, lengths in lengths_by_matchup.items():
        ax.hist(lengths, bins=30, alpha=0.5, label=name)
    ax.set_xlabel("Game Length (turns)")
    ax.set_ylabel("Count")
    ax.set_title("Game Length Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {RESULTS_DIR / filename}")
