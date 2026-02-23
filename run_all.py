#!/usr/bin/env python3
"""Run comprehensive analysis: tournament, deck optimization, and charts."""
import sys
from pathlib import Path

from mtg.game import Game, make_deck
from mtg.agents import RandomAgent, BoltFaceAgent, AggroGoblinAgent, MixedBoltFaceAgent
from mtg.training import NNAgent, make_nn_agent
from mtg.elo import Tournament
from mtg.enums import GameResult
from mtg.deck import optimize_deck
from mtg.analyze import (
    plot_elo_history, plot_win_matrix, plot_training_loss,
    plot_deck_optimization, plot_game_length_distribution,
)
from play import parse_deck


def run_tournament(model_path: str, deck_str: str = "20M,10B,10G", life: int = 20, games: int = 200):
    deck = parse_deck(deck_str)
    model_dir = Path(model_path).parent

    tournament = Tournament()
    tournament.register("random", RandomAgent(), deck)
    tournament.register("bolt_face", BoltFaceAgent(), deck)
    tournament.register("aggro_goblin", AggroGoblinAgent(), deck)
    tournament.register("mixed", MixedBoltFaceAgent(), deck)

    if Path(model_path).exists():
        agent = make_nn_agent(model_path)
        tournament.register("trained", agent, deck)

    checkpoints = sorted(model_dir.glob("checkpoint_*.pt"))
    for ckpt in checkpoints[-3:]:
        name = ckpt.stem
        agent = make_nn_agent(str(ckpt))
        tournament.register(name, agent, deck)

    print(f"\nTournament: {len(tournament.agents)} agents, {games} games/pair, life={life}")
    tournament.round_robin(games_per_pair=games, starting_life=life)
    print(tournament.report())

    names = sorted(tournament.ratings.keys(), key=lambda n: -tournament.ratings[n])
    plot_win_matrix(names, tournament.win_matrix, tournament.game_matrix,
                    f"tournament_matrix_life{life}.png",
                    f"Win Rate Matrix (Life={life})")

    return tournament


def run_deck_optimization(model_path: str, life_totals: list[int] = None):
    if life_totals is None:
        life_totals = [3, 5, 7, 10, 20]

    results_by_life = {}

    for life in life_totals:
        print(f"\n  Optimizing deck for life={life}...")
        if Path(model_path).exists():
            from mtg.network import QNetwork
            import torch
            agent_cls = NNAgent
            q_net = QNetwork()
            ckpt = torch.load(model_path, weights_only=True)
            q_net.load_state_dict(ckpt["q_net"])
            agent_kwargs = {"q_net": q_net, "epsilon": 0.0}
        else:
            agent_cls = MixedBoltFaceAgent
            agent_kwargs = {}

        results = optimize_deck(
            agent_cls=agent_cls,
            agent_kwargs=agent_kwargs,
            opponent_cls=MixedBoltFaceAgent,
            opponent_kwargs={},
            starting_life=life,
            min_deck_size=15,
            max_deck_size=50,
            step=5,
            games_per_deck=200,
            top_k=5,
        )

        results_by_life[life] = results
        print(f"  Life={life} best deck: {results[0][0]} (WR: {results[0][1]:.1%})")
        for comp, wr in results[:5]:
            print(f"    {comp['mountains']}M/{comp['bolts']}B/{comp['goblins']}G (total {comp['total']}) → {wr:.1%}")

    plot_deck_optimization(results_by_life, "optimal_decks.png")
    return results_by_life


def run_game_length_analysis(model_path: str, life: int = 20):
    deck = parse_deck("20M,10B,10G")
    matchups = {
        "Random vs Random": (RandomAgent, RandomAgent),
        "BoltFace vs BoltFace": (BoltFaceAgent, BoltFaceAgent),
        "Mixed vs Mixed": (MixedBoltFaceAgent, MixedBoltFaceAgent),
    }

    if Path(model_path).exists():
        matchups["Trained vs Mixed"] = ("trained", MixedBoltFaceAgent)

    lengths_by_matchup = {}
    for name, (cls1, cls2) in matchups.items():
        lengths = []
        for _ in range(200):
            if cls1 == "trained":
                a1 = make_nn_agent(model_path)
            else:
                a1 = cls1()
            a2 = cls2()
            game = Game(
                decks=[list(deck), list(deck)],
                agents=[a1, a2],
                starting_life=life,
                verbose=False,
            )
            game.run_game()
            lengths.append(game.turn_number)
        lengths_by_matchup[name] = lengths
        print(f"  {name}: avg={sum(lengths)/len(lengths):.1f} turns")

    plot_game_length_distribution(lengths_by_matchup, f"game_lengths_life{life}.png")


def decision_analysis(model_path: str):
    """Show what decisions the trained agent makes in key situations."""
    if not Path(model_path).exists():
        print("  No trained model found, skipping decision analysis")
        return

    from mtg.features import state_to_features, get_legal_action_mask, index_to_action, ACTION_DIM
    from mtg.game import GameView, Action, Target
    from mtg.enums import Step, TargetType, ActionType
    from mtg.cards import CardInstance, CARD_REGISTRY
    import torch
    import numpy as np

    q_net = __import__("mtg.network", fromlist=["QNetwork"]).QNetwork()
    ckpt = torch.load(model_path, weights_only=True)
    q_net.load_state_dict(ckpt["q_net"])

    print("\n=== Decision Analysis ===\n")
    print("What does the trained agent do in key situations?\n")

    scenarios = [
        {
            "name": "Turn 1, Mountain+Bolt in hand, opp at 3 life",
            "life": 20, "opp_life": 3,
            "hand_cards": ["Mountain", "Lightning Bolt", "Raging Goblin", "Mountain"],
            "my_lands": 0, "opp_lands": 0,
            "my_creatures": 0, "opp_creatures": 0,
            "step": Step.PRECOMBAT_MAIN,
        },
        {
            "name": "Turn 3, 2 Mountains, Bolt+Goblin in hand, opp at 15",
            "life": 20, "opp_life": 15,
            "hand_cards": ["Lightning Bolt", "Raging Goblin", "Mountain"],
            "my_lands": 2, "opp_lands": 2,
            "my_creatures": 1, "opp_creatures": 0,
            "step": Step.PRECOMBAT_MAIN,
        },
        {
            "name": "Opponent at 3 life, Bolt in hand, Mountain untapped",
            "life": 5, "opp_life": 3,
            "hand_cards": ["Lightning Bolt"],
            "my_lands": 1, "opp_lands": 3,
            "my_creatures": 0, "opp_creatures": 2,
            "step": Step.PRECOMBAT_MAIN,
        },
    ]

    for scenario in scenarios:
        iid = 100
        hand = []
        for cn in scenario["hand_cards"]:
            hand.append(CardInstance(CARD_REGISTRY[cn], instance_id=iid, owner=0))
            iid += 1

        my_bf = []
        for _ in range(scenario["my_lands"]):
            my_bf.append(CardInstance(CARD_REGISTRY["Mountain"], instance_id=iid, owner=0))
            iid += 1
        for _ in range(scenario["my_creatures"]):
            c = CardInstance(CARD_REGISTRY["Raging Goblin"], instance_id=iid, owner=0)
            iid += 1
            my_bf.append(c)

        opp_bf = []
        for _ in range(scenario["opp_lands"]):
            opp_bf.append(CardInstance(CARD_REGISTRY["Mountain"], instance_id=iid, owner=1))
            iid += 1
        for _ in range(scenario["opp_creatures"]):
            c = CardInstance(CARD_REGISTRY["Raging Goblin"], instance_id=iid, owner=1)
            iid += 1
            opp_bf.append(c)

        view = GameView(
            my_player=0, my_life=scenario["life"], opp_life=scenario["opp_life"],
            my_hand=hand, my_battlefield=my_bf, opp_battlefield=opp_bf,
            my_graveyard=[], opp_graveyard=[],
            opp_hand_size=4, my_library_size=20, opp_library_size=20,
            stack=[], step=scenario["step"], active_player=0,
            turn_number=3, land_played_this_turn=False, starting_life=20,
        )

        state = state_to_features(view)
        with torch.no_grad():
            q_values = q_net(torch.from_numpy(state).float().unsqueeze(0)).squeeze(0).numpy()

        # Show top 5 Q-values
        top_indices = np.argsort(q_values)[::-1][:10]
        print(f"  Scenario: {scenario['name']}")
        for rank, idx in enumerate(top_indices[:5]):
            print(f"    {rank+1}. Action #{idx}: Q={q_values[idx]:.3f}")
        print()


def main():
    model_path = "models/life20/final.pt"

    # Check for v2
    v2_path = "models/life20_v2/final.pt"
    if Path(v2_path).exists():
        model_path = v2_path
        print(f"Using v2 model: {model_path}")

    print("=" * 60)
    print("  MTG AI ANALYSIS")
    print("=" * 60)

    print("\n--- Tournament ---")
    run_tournament(model_path, life=20)

    print("\n--- Game Length Analysis ---")
    run_game_length_analysis(model_path, life=20)

    print("\n--- Deck Optimization ---")
    run_deck_optimization(model_path)

    print("\n--- Decision Analysis ---")
    decision_analysis(model_path)

    print("\n\nAll charts saved to results/")


if __name__ == "__main__":
    main()
