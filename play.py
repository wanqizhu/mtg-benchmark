#!/usr/bin/env python3
"""Play MTG games: human vs human, human vs bot, or bot vs bot."""
import argparse
import sys
from mtg.game import Game, make_deck
from mtg.agents import (
    HumanAgent, RandomAgent, BoltFaceAgent, AggroGoblinAgent, MixedBoltFaceAgent,
)
from mtg.enums import GameResult


AGENT_TYPES = {
    "human": HumanAgent,
    "random": RandomAgent,
    "bolt_face": BoltFaceAgent,
    "aggro_goblin": AggroGoblinAgent,
    "mixed": MixedBoltFaceAgent,
    "trained": None,  # handled specially
}


def parse_deck(deck_str: str) -> list[str]:
    """Parse deck string like '20M,10B,10G' into card list."""
    cards = []
    for part in deck_str.split(","):
        part = part.strip()
        if not part:
            continue
        count = ""
        label = ""
        for ch in part:
            if ch.isdigit():
                count += ch
            else:
                label += ch
        n = int(count) if count else 1
        name_map = {"M": "Mountain", "B": "Lightning Bolt", "G": "Raging Goblin"}
        card_name = name_map.get(label.upper(), label)
        cards.extend([card_name] * n)
    return cards


def main():
    parser = argparse.ArgumentParser(description="Play MTG games")
    parser.add_argument("--p1", default="human", choices=list(AGENT_TYPES.keys()),
                        help="Player 1 type")
    parser.add_argument("--p2", default="random", choices=list(AGENT_TYPES.keys()),
                        help="Player 2 type")
    parser.add_argument("--deck1", default="20M,10B,10G",
                        help="Player 1 deck (e.g. '20M,10B,10G')")
    parser.add_argument("--deck2", default=None,
                        help="Player 2 deck (defaults to same as deck1)")
    parser.add_argument("--life", type=int, default=20, help="Starting life total")
    parser.add_argument("--hand", type=int, default=7, help="Starting hand size")
    parser.add_argument("--games", type=int, default=1, help="Number of games (for bot vs bot)")
    parser.add_argument("--verbose", action="store_true", help="Show game log")
    parser.add_argument("--first", type=int, default=None, choices=[0, 1],
                        help="Who goes first (random if not set)")
    parser.add_argument("--model", default="models/life20_v3/finetuned.pt",
                        help="Model path for trained agents (used by both unless overridden)")
    parser.add_argument("--model1", default=None,
                        help="Model path for trained player 1 (overrides --model)")
    parser.add_argument("--model2", default=None,
                        help="Model path for trained player 2 (overrides --model)")
    args = parser.parse_args()

    deck1 = parse_deck(args.deck1)
    deck2 = parse_deck(args.deck2) if args.deck2 else list(deck1)

    model1 = args.model1 or args.model
    model2 = args.model2 or args.model

    def make_agent(ptype, model_path):
        if ptype == "trained":
            from mtg.training import make_nn_agent
            return make_nn_agent(model_path)
        return AGENT_TYPES[ptype]()

    is_human = args.p1 == "human" or args.p2 == "human"
    verbose = args.verbose or is_human

    if args.games == 1 or is_human:
        agent1 = make_agent(args.p1, model1)
        agent2 = make_agent(args.p2, model2)
        game = Game(
            decks=[deck1, deck2],
            agents=[agent1, agent2],
            starting_life=args.life,
            starting_hand_size=args.hand,
            first_player=args.first,
            verbose=verbose,
        )
        result = game.run_game()
        print(f"\nResult: {result.name} (Turn {game.turn_number})")
    else:
        wins = [0, 0, 0]
        for i in range(args.games):
            agent1 = make_agent(args.p1, model1)
            agent2 = make_agent(args.p2, model2)
            game = Game(
                decks=[deck1, deck2],
                agents=[agent1, agent2],
                starting_life=args.life,
                starting_hand_size=args.hand,
                verbose=False,
            )
            result = game.run_game()
            wins[result.value] += 1

        total = args.games
        print(f"\nResults over {total} games:")
        print(f"  Player 0 ({args.p1}): {wins[0]} wins ({100*wins[0]/total:.1f}%)")
        print(f"  Player 1 ({args.p2}): {wins[1]} wins ({100*wins[1]/total:.1f}%)")
        if wins[2]:
            print(f"  Draws: {wins[2]} ({100*wins[2]/total:.1f}%)")


if __name__ == "__main__":
    main()
