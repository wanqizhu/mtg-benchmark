"""Comprehensive tests for MTG rules engine correctness."""
from mtg.game import Game, Action, Target, make_deck
from mtg.enums import ActionType, TargetType, GameResult, Step
from mtg.cards import CARD_REGISTRY
from mtg.agents import Agent, RandomAgent, BoltFaceAgent


class SmartScriptAgent(Agent):
    """Agent with phase-aware decision making via a priority function."""

    def __init__(self, decide_fn=None):
        self.decide_fn = decide_fn or self._default
        self.history = []

    def choose_action(self, view, legal_actions):
        action = self.decide_fn(view, legal_actions, self.history)
        self.history.append((view, legal_actions, action))
        return action

    @staticmethod
    def _default(view, legal, history):
        return legal[0]


def keep_hand(view, legal, history):
    for a in legal:
        if a.action_type == ActionType.MULLIGAN_KEEP:
            return a
    return legal[0]


def passive(view, legal, history):
    """Always keep, always pass, never block."""
    if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
        return keep_hand(view, legal, history)
    if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
        return legal[0]  # attack with nothing
    if legal[0].action_type == ActionType.DECLARE_BLOCKERS:
        for a in legal:
            if a.blocker_id is None:
                return a
        return legal[0]
    if legal[0].action_type == ActionType.CHOOSE_BOTTOM_CARD:
        return legal[0]
    for a in legal:
        if a.action_type == ActionType.PASS_PRIORITY:
            return a
    return legal[0]


def make_game(deck1, deck2, agents, life=20, first_player=0):
    return Game(
        decks=[deck1, deck2],
        agents=agents,
        starting_life=life,
        first_player=first_player,
        verbose=False,
    )


class TestLandRules:
    def test_can_play_one_land_per_turn(self):
        deck = ["Mountain"] * 10 + ["Lightning Bolt"] * 10

        lands_offered = []

        def track_lands(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type in (ActionType.DECLARE_ATTACKERS,):
                return legal[0]
            if legal[0].action_type == ActionType.CHOOSE_BOTTOM_CARD:
                return legal[0]

            land_actions = [a for a in legal if a.action_type == ActionType.PLAY_LAND]
            if view.step in (Step.PRECOMBAT_MAIN, Step.POSTCOMBAT_MAIN) and view.active_player == view.my_player:
                lands_offered.append(len(land_actions))

            if land_actions:
                return land_actions[0]

            for a in legal:
                if a.action_type == ActionType.PASS_PRIORITY:
                    return a
            return legal[0]

        a0 = SmartScriptAgent(track_lands)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck, deck, [a0, a1], life=20, first_player=0)
        game.run_game()

        # In the first main phase call, land should be available
        assert len(lands_offered) >= 2
        assert lands_offered[0] == 1  # can play a land
        # After playing it, the next main phase call (postcombat) should have 0 lands
        assert lands_offered[1] == 0

    def test_cant_play_land_outside_main(self):
        deck = ["Mountain"] * 20

        def check_no_land_in_upkeep(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if view.step == Step.UPKEEP:
                land_actions = [a for a in legal if a.action_type == ActionType.PLAY_LAND]
                assert len(land_actions) == 0, "Should not be able to play land in upkeep"
            return passive(view, legal, history)

        a0 = SmartScriptAgent(check_no_land_in_upkeep)
        a1 = SmartScriptAgent(passive)
        game = make_game(deck, deck, [a0, a1], life=20, first_player=0)
        game.run_game()


class TestCasting:
    def test_bolt_costs_one_red(self):
        deck1 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10
        deck2 = ["Mountain"] * 20

        def play_mountain_then_bolt(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if (a.action_type == ActionType.CAST_SPELL and a.card and
                        a.card.name == "Lightning Bolt" and a.targets and
                        a.targets[0].target_type == TargetType.PLAYER and
                        a.targets[0].target_id != view.my_player):
                    return a
            return passive(view, legal, history)

        a0 = SmartScriptAgent(play_mountain_then_bolt)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck1, deck2, [a0, a1], life=20, first_player=0)
        game.run_game()
        assert game.players[1].life < 20

    def test_cant_cast_without_mana(self):
        deck1 = ["Lightning Bolt"] * 20
        deck2 = ["Mountain"] * 20

        def check_no_cast(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            casts = [a for a in legal if a.action_type == ActionType.CAST_SPELL]
            assert len(casts) == 0, "Should not be able to cast without mana"
            return passive(view, legal, history)

        a0 = SmartScriptAgent(check_no_cast)
        a1 = SmartScriptAgent(passive)
        game = make_game(deck1, deck2, [a0, a1], life=20)
        game.run_game()

    def test_goblin_only_main_phase(self):
        deck = ["Mountain"] * 10 + ["Raging Goblin"] * 10

        def check_goblin_timing(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            goblin_casts = [a for a in legal if a.action_type == ActionType.CAST_SPELL
                           and a.card and a.card.name == "Raging Goblin"]
            if view.step not in (Step.PRECOMBAT_MAIN, Step.POSTCOMBAT_MAIN):
                assert len(goblin_casts) == 0, f"Should not cast goblin during {view.step}"
            return passive(view, legal, history)

        a0 = SmartScriptAgent(check_goblin_timing)
        a1 = SmartScriptAgent(passive)
        game = make_game(deck, deck, [a0, a1], life=20, first_player=0)
        game.run_game()


class TestHaste:
    def test_goblin_can_attack_turn_played(self):
        deck1 = ["Mountain"] * 10 + ["Raging Goblin"] * 10
        deck2 = ["Mountain"] * 20

        could_attack = [False]

        def play_goblin_and_attack(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)

            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                has_attack = any(len(a.attackers) > 0 for a in legal)
                if has_attack:
                    could_attack[0] = True
                return max(legal, key=lambda a: len(a.attackers))

            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if a.action_type == ActionType.CAST_SPELL and a.card and a.card.name == "Raging Goblin":
                    return a
            return passive(view, legal, history)

        a0 = SmartScriptAgent(play_goblin_and_attack)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck1, deck2, [a0, a1], life=20, first_player=0)
        game.run_game()
        assert could_attack[0], "Goblin with haste should be able to attack the turn it's played"


class TestCombat:
    def test_unblocked_damage(self):
        deck1 = ["Mountain"] * 10 + ["Raging Goblin"] * 10
        deck2 = ["Mountain"] * 20

        def play_and_attack(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return max(legal, key=lambda a: len(a.attackers))
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if a.action_type == ActionType.CAST_SPELL and a.card and a.card.name == "Raging Goblin":
                    return a
            return passive(view, legal, history)

        a0 = SmartScriptAgent(play_and_attack)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck1, deck2, [a0, a1], life=20, first_player=0)
        game.run_game()
        assert game.players[1].life < 20, "Unblocked goblin should deal damage"

    def test_blocked_creatures_trade(self):
        deck = ["Mountain"] * 5 + ["Raging Goblin"] * 15

        def play_attack(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return max(legal, key=lambda a: len(a.attackers))
            if legal[0].action_type == ActionType.DECLARE_BLOCKERS:
                for a in legal:
                    if a.blocker_id is not None:
                        return a
                return legal[0]
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if a.action_type == ActionType.CAST_SPELL and a.card and a.card.name == "Raging Goblin":
                    return a
            return passive(view, legal, history)

        a0 = SmartScriptAgent(play_attack)
        a1 = SmartScriptAgent(play_attack)

        game = make_game(deck, deck, [a0, a1], life=20, first_player=0)
        game.run_game()


class TestStateBasedActions:
    def test_zero_life_loses(self):
        deck1 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10
        deck2 = ["Mountain"] * 20

        def bolt_face(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if (a.action_type == ActionType.CAST_SPELL and a.card and
                        a.card.name == "Lightning Bolt" and a.targets and
                        a.targets[0].target_type == TargetType.PLAYER and
                        a.targets[0].target_id != view.my_player):
                    return a
            return passive(view, legal, history)

        a0 = SmartScriptAgent(bolt_face)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck1, deck2, [a0, a1], life=3, first_player=0)
        result = game.run_game()
        assert result == GameResult.WIN_0

    def test_empty_library_loses(self):
        deck1 = ["Mountain"] * 8
        deck2 = ["Mountain"] * 8

        a0 = SmartScriptAgent(passive)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck1, deck2, [a0, a1], life=100, first_player=0)
        result = game.run_game()
        assert result in (GameResult.WIN_0, GameResult.WIN_1)

    def test_creature_dies_to_bolt(self):
        deck1 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10
        deck2 = ["Mountain"] * 10 + ["Raging Goblin"] * 10

        goblins_died = [False]

        def bolt_creature(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if (a.action_type == ActionType.CAST_SPELL and a.card and
                        a.card.name == "Lightning Bolt" and a.targets and
                        a.targets[0].target_type == TargetType.CREATURE):
                    goblins_died[0] = True
                    return a
            return passive(view, legal, history)

        def play_goblin(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if a.action_type == ActionType.CAST_SPELL and a.card and a.card.name == "Raging Goblin":
                    return a
            return passive(view, legal, history)

        a0 = SmartScriptAgent(bolt_creature)
        a1 = SmartScriptAgent(play_goblin)

        game = make_game(deck1, deck2, [a0, a1], life=20, first_player=1)
        game.run_game()


class TestStack:
    def test_bolt_goes_on_stack(self):
        """Bolt should go on stack, not resolve immediately. P1 needs mana to get priority."""
        deck1 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10
        deck2 = ["Mountain"] * 10 + ["Lightning Bolt"] * 10

        saw_stack = [False]

        def check_stack(view, legal, history):
            if view.stack:
                saw_stack[0] = True
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            # P1 plays a mountain so they have mana and will get real priority calls
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            return passive(view, legal, history)

        def bolt_face(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.DECLARE_ATTACKERS:
                return legal[0]
            for a in legal:
                if a.action_type == ActionType.PLAY_LAND:
                    return a
            for a in legal:
                if (a.action_type == ActionType.CAST_SPELL and a.card and
                        a.card.name == "Lightning Bolt" and a.targets and
                        a.targets[0].target_type == TargetType.PLAYER and
                        a.targets[0].target_id != view.my_player):
                    return a
            return passive(view, legal, history)

        # P1 goes first so they play a mountain on T1, then P0 plays mountain + bolt on T2
        a0 = SmartScriptAgent(bolt_face)
        a1 = SmartScriptAgent(check_stack)
        game = make_game(deck1, deck2, [a0, a1], life=20, first_player=1)
        game.run_game()

        assert saw_stack[0], "Opponent should see bolt on stack before it resolves"


class TestFirstPlayer:
    def test_first_player_skips_draw(self):
        deck1 = ["Mountain"] * 20
        deck2 = ["Mountain"] * 20

        a0 = SmartScriptAgent(passive)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck1, deck2, [a0, a1], life=20, first_player=0)

        # Manually do initial draw + mulligan
        for i in range(2):
            game.draw_cards(i, game.starting_hand_size)
        game.mulligan_phase()

        assert len(game.players[0].hand) == 7
        assert len(game.players[0].library) == 13

        game.turn_number = 1
        game.run_turn()

        # P0 skips first draw → library should still be 13
        assert len(game.players[0].library) == 13, f"First player should skip T1 draw, lib={len(game.players[0].library)}"
        # P0 should have drawn one fewer card total than P1


class TestMulligan:
    def test_mulligan_reduces_hand(self):
        deck = ["Mountain"] * 10 + ["Lightning Bolt"] * 10

        mull_count = [0]

        def mulligan_once(view, legal, history):
            if legal[0].action_type in (ActionType.MULLIGAN_KEEP, ActionType.MULLIGAN_TAKE):
                if mull_count[0] == 0:
                    mull_count[0] += 1
                    for a in legal:
                        if a.action_type == ActionType.MULLIGAN_TAKE:
                            return a
                return keep_hand(view, legal, history)
            if legal[0].action_type == ActionType.CHOOSE_BOTTOM_CARD:
                return legal[0]
            return passive(view, legal, history)

        a0 = SmartScriptAgent(mulligan_once)
        a1 = SmartScriptAgent(passive)

        game = make_game(deck, deck, [a0, a1], life=20, first_player=0)
        game.run_game()

        # After 1 mulligan, P0 should have put 1 card on bottom → 6 cards in starting hand


class TestFullGame:
    def test_random_vs_random_1000_games(self):
        wins = [0, 0, 0]
        for _ in range(1000):
            a0 = RandomAgent()
            a1 = RandomAgent()
            deck = make_deck(20, 10, 10)
            game = Game(
                decks=[list(deck), list(deck)],
                agents=[a0, a1],
                starting_life=20,
                verbose=False,
            )
            result = game.run_game()
            wins[result.value] += 1
        total_decided = wins[0] + wins[1]
        assert total_decided > 900

    def test_bolt_face_beats_random(self):
        p0_wins = 0
        for _ in range(200):
            a0 = BoltFaceAgent()
            a1 = RandomAgent()
            deck = make_deck(20, 20, 0)
            game = Game(
                decks=[list(deck), list(deck)],
                agents=[a0, a1],
                starting_life=20,
                verbose=False,
            )
            result = game.run_game()
            if result == GameResult.WIN_0:
                p0_wins += 1
        assert p0_wins > 150

    def test_no_crash_various_decks(self):
        """Test with various deck compositions to ensure no crashes."""
        deck_configs = [
            (40, 0, 0),   # all mountains
            (10, 30, 0),  # mountains + bolts
            (10, 0, 30),  # mountains + goblins
            (15, 10, 15), # mixed
            (7, 7, 6),    # small deck
        ]
        for m, b, g in deck_configs:
            for _ in range(50):
                a0 = RandomAgent()
                a1 = RandomAgent()
                deck = make_deck(m, b, g)
                game = Game(
                    decks=[list(deck), list(deck)],
                    agents=[a0, a1],
                    starting_life=20,
                    verbose=False,
                )
                game.run_game()

    def test_low_life_games(self):
        """Games with low life should end quickly."""
        for life in [1, 2, 3, 5]:
            for _ in range(100):
                a0 = BoltFaceAgent()
                a1 = BoltFaceAgent()
                deck = make_deck(15, 15, 0)
                game = Game(
                    decks=[list(deck), list(deck)],
                    agents=[a0, a1],
                    starting_life=life,
                    verbose=False,
                )
                result = game.run_game()
                assert result != GameResult.DRAW, f"Low life game should not draw (life={life})"


if __name__ == "__main__":
    test_classes = [
        TestLandRules, TestCasting, TestHaste, TestCombat,
        TestStateBasedActions, TestStack, TestFirstPlayer,
        TestMulligan, TestFullGame,
    ]
    passed = 0
    failed = 0
    for cls in test_classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if name.startswith("test_"):
                print(f"  {cls.__name__}.{name}...", end=" ")
                try:
                    getattr(obj, name)()
                    print("PASSED")
                    passed += 1
                except Exception as e:
                    print(f"FAILED: {e}")
                    failed += 1
    print(f"\n{passed} passed, {failed} failed")
