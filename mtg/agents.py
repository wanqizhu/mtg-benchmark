from __future__ import annotations
import random as rng
from abc import ABC, abstractmethod

from .enums import ActionType, GameResult, TargetType
from .game import Action, GameView


class Agent(ABC):
    @abstractmethod
    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        ...

    def game_over_callback(self, result: GameResult, player: int) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomAgent(Agent):
    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        return rng.choice(legal_actions)


class HumanAgent(Agent):
    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        self._display(view)
        print()
        for i, a in enumerate(legal_actions):
            print(f"  {i + 1}. {a}")
        print()

        while True:
            try:
                choice = int(input("Choose action: ")) - 1
                if 0 <= choice < len(legal_actions):
                    return legal_actions[choice]
            except (ValueError, EOFError):
                pass
            print(f"Enter a number 1-{len(legal_actions)}")

    def _display(self, view: GameView):
        print()
        print("=" * 50)
        opp = 1 - view.my_player
        print(f"  OPPONENT (Player {opp}) | Life: {view.opp_life} | Hand: {view.opp_hand_size} | Library: {view.opp_library_size}")
        print(f"  Battlefield: {self._fmt_battlefield(view.opp_battlefield)}")
        print("-" * 50)
        if view.stack:
            stack_str = ", ".join(self._fmt_stack_item(s) for s in reversed(view.stack))
            print(f"  Stack: [{stack_str}]")
            print("-" * 50)
        print(f"  YOU (Player {view.my_player}) | Life: {view.my_life} | Library: {view.my_library_size}")
        print(f"  Hand: {', '.join(c.name for c in view.my_hand)}")
        print(f"  Battlefield: {self._fmt_battlefield(view.my_battlefield)}")
        print("-" * 50)
        active_str = "Your turn" if view.active_player == view.my_player else "Opponent's turn"
        land_str = " (land played)" if view.land_played_this_turn else ""
        print(f"  Turn {view.turn_number} | {view.step.name} | {active_str}{land_str}")
        print("=" * 50)

    def _fmt_battlefield(self, cards: list) -> str:
        if not cards:
            return "(empty)"
        parts = []
        lands = [c for c in cards if c.is_land]
        creatures = [c for c in cards if c.is_creature]
        if lands:
            land_strs = []
            for c in lands:
                s = c.name
                if c.tapped:
                    s += "(T)"
                land_strs.append(s)
            parts.append("Lands: " + " ".join(land_strs))
        if creatures:
            creature_strs = []
            for c in creatures:
                s = f"{c.name} {c.power}/{c.toughness}"
                if c.tapped:
                    s += "(T)"
                if c.damage_marked > 0:
                    s += f"[{c.damage_marked}dmg]"
                creature_strs.append(s)
            parts.append("Creatures: " + " ".join(creature_strs))
        return " | ".join(parts)

    def _fmt_stack_item(self, item) -> str:
        target_str = ""
        if item.targets:
            t = item.targets[0]
            if t.target_type == TargetType.PLAYER:
                target_str = f" → Player {t.target_id}"
            else:
                target_str = f" → Creature#{t.target_id}"
        return f"{item.card.name}{target_str}"


class BoltFaceAgent(Agent):
    """Play mountains, bolt opponent's face, attack with everything, never block."""

    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        for a in legal_actions:
            if a.action_type == ActionType.PLAY_LAND:
                return a

        for a in legal_actions:
            if (a.action_type == ActionType.CAST_SPELL and
                    a.card and a.card.name == "Lightning Bolt" and
                    a.targets and a.targets[0].target_type == TargetType.PLAYER and
                    a.targets[0].target_id != view.my_player):
                return a

        if legal_actions[0].action_type == ActionType.DECLARE_ATTACKERS:
            return max(legal_actions, key=lambda a: len(a.attackers))

        if legal_actions[0].action_type == ActionType.DECLARE_BLOCKERS:
            for a in legal_actions:
                if a.blocker_id is None:
                    return a

        if legal_actions[0].action_type == ActionType.MULLIGAN_KEEP:
            return self._mulligan_decision(view, legal_actions, min_lands=2, min_bolts=1, min_goblins=0)

        if legal_actions[0].action_type == ActionType.CHOOSE_BOTTOM_CARD:
            return self._bottom_card_decision(view, legal_actions, prefer_keep_bolts=True)

        return self._pass(legal_actions)

    def _mulligan_decision(self, view, legal_actions, min_lands, min_bolts, min_goblins):
        lands = sum(1 for c in view.my_hand if c.is_land)
        bolts = sum(1 for c in view.my_hand if c.name == "Lightning Bolt")
        goblins = sum(1 for c in view.my_hand if c.name == "Raging Goblin")
        if lands >= min_lands and bolts >= min_bolts and goblins >= min_goblins:
            return legal_actions[0]  # KEEP
        if len(view.my_hand) <= 5:
            return legal_actions[0]  # keep if already mulled twice
        return legal_actions[1]  # MULLIGAN

    def _bottom_card_decision(self, view, legal_actions, prefer_keep_bolts=True):
        lands = sum(1 for c in view.my_hand if c.is_land)
        if lands > 3:
            for a in legal_actions:
                if a.card and a.card.is_land:
                    return a
        for a in legal_actions:
            if a.card and a.card.name == "Raging Goblin":
                return a
        return legal_actions[-1]

    def _pass(self, legal_actions):
        for a in legal_actions:
            if a.action_type == ActionType.PASS_PRIORITY:
                return a
        return legal_actions[0]


class AggroGoblinAgent(Agent):
    """Play mountains, cast goblins, attack with everything, never block."""

    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        for a in legal_actions:
            if a.action_type == ActionType.PLAY_LAND:
                return a

        for a in legal_actions:
            if (a.action_type == ActionType.CAST_SPELL and
                    a.card and a.card.name == "Raging Goblin"):
                return a

        if legal_actions[0].action_type == ActionType.DECLARE_ATTACKERS:
            return max(legal_actions, key=lambda a: len(a.attackers))

        if legal_actions[0].action_type == ActionType.DECLARE_BLOCKERS:
            for a in legal_actions:
                if a.blocker_id is None:
                    return a

        if legal_actions[0].action_type == ActionType.MULLIGAN_KEEP:
            lands = sum(1 for c in view.my_hand if c.is_land)
            goblins = sum(1 for c in view.my_hand if c.name == "Raging Goblin")
            if lands >= 2 and goblins >= 1:
                return legal_actions[0]
            if len(view.my_hand) <= 5:
                return legal_actions[0]
            return legal_actions[1]

        if legal_actions[0].action_type == ActionType.CHOOSE_BOTTOM_CARD:
            for a in legal_actions:
                if a.card and a.card.name == "Lightning Bolt":
                    return a
            lands = sum(1 for c in view.my_hand if c.is_land)
            if lands > 3:
                for a in legal_actions:
                    if a.card and a.card.is_land:
                        return a
            return legal_actions[-1]

        return self._pass(legal_actions)

    def _pass(self, legal_actions):
        for a in legal_actions:
            if a.action_type == ActionType.PASS_PRIORITY:
                return a
        return legal_actions[0]


class MixedBoltFaceAgent(Agent):
    """Play mountains, cast goblins, bolt face, attack with everything."""

    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        for a in legal_actions:
            if a.action_type == ActionType.PLAY_LAND:
                return a

        for a in legal_actions:
            if (a.action_type == ActionType.CAST_SPELL and
                    a.card and a.card.name == "Raging Goblin"):
                return a

        for a in legal_actions:
            if (a.action_type == ActionType.CAST_SPELL and
                    a.card and a.card.name == "Lightning Bolt" and
                    a.targets and a.targets[0].target_type == TargetType.PLAYER and
                    a.targets[0].target_id != view.my_player):
                return a

        if legal_actions[0].action_type == ActionType.DECLARE_ATTACKERS:
            return max(legal_actions, key=lambda a: len(a.attackers))

        if legal_actions[0].action_type == ActionType.DECLARE_BLOCKERS:
            for a in legal_actions:
                if a.blocker_id is None:
                    return a

        if legal_actions[0].action_type == ActionType.MULLIGAN_KEEP:
            lands = sum(1 for c in view.my_hand if c.is_land)
            spells = sum(1 for c in view.my_hand if not c.is_land)
            if lands >= 2 and spells >= 1:
                return legal_actions[0]
            if len(view.my_hand) <= 5:
                return legal_actions[0]
            return legal_actions[1]

        if legal_actions[0].action_type == ActionType.CHOOSE_BOTTOM_CARD:
            lands = sum(1 for c in view.my_hand if c.is_land)
            if lands > 3:
                for a in legal_actions:
                    if a.card and a.card.is_land:
                        return a
            return legal_actions[-1]

        return self._pass(legal_actions)

    def _pass(self, legal_actions):
        for a in legal_actions:
            if a.action_type == ActionType.PASS_PRIORITY:
                return a
        return legal_actions[0]
