from __future__ import annotations
import random as rng
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .enums import (
    ActionType, CardType, GameResult, Phase, Step,
    STEP_ORDER, MAIN_PHASE_STEPS, STEP_TO_PHASE,
    TargetType, Zone,
)
from .cards import CardDef, CardInstance, CARD_REGISTRY

if TYPE_CHECKING:
    from .agents import Agent

MAX_CREATURES_PER_SIDE = 8
MAX_TURNS = 200


@dataclass(frozen=True)
class Target:
    target_type: TargetType
    target_id: int  # player index for PLAYER, instance_id for CREATURE


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    card: CardInstance | None = None
    targets: tuple[Target, ...] = ()
    attackers: tuple[int, ...] = ()  # creature instance_ids
    blocker_id: int | None = None  # creature instance_id used as blocker
    attacker_to_block_id: int | None = None
    hand_index: int | None = None

    def __repr__(self) -> str:
        if self.action_type == ActionType.PASS_PRIORITY:
            return "Pass priority"
        if self.action_type == ActionType.PLAY_LAND:
            return f"Play {self.card.name}"
        if self.action_type == ActionType.CAST_SPELL:
            target_strs = []
            for t in self.targets:
                if t.target_type == TargetType.PLAYER:
                    target_strs.append(f"Player {t.target_id}")
                else:
                    target_strs.append(f"Creature#{t.target_id}")
            tgt = ", ".join(target_strs) if target_strs else "no target"
            return f"Cast {self.card.name} → {tgt}"
        if self.action_type == ActionType.DECLARE_ATTACKERS:
            if not self.attackers:
                return "Attack with no creatures"
            return f"Attack with creatures {self.attackers}"
        if self.action_type == ActionType.DECLARE_BLOCKERS:
            if self.blocker_id is None:
                return f"Don't block attacker #{self.attacker_to_block_id}"
            return f"Block attacker #{self.attacker_to_block_id} with creature #{self.blocker_id}"
        if self.action_type == ActionType.MULLIGAN_KEEP:
            return "Keep hand"
        if self.action_type == ActionType.MULLIGAN_TAKE:
            return "Mulligan"
        if self.action_type == ActionType.CHOOSE_BOTTOM_CARD:
            return f"Put {self.card.name} on bottom (hand index {self.hand_index})"
        if self.action_type == ActionType.CONCEDE:
            return "Concede"
        return f"Action({self.action_type})"


@dataclass
class StackItem:
    card: CardInstance
    controller: int
    targets: list[Target]


@dataclass
class PlayerState:
    life: int
    library: list[CardInstance]
    hand: list[CardInstance]
    battlefield: list[CardInstance]
    graveyard: list[CardInstance]
    mana_pool: int = 0  # red mana (only type in current card pool)
    land_played_this_turn: bool = False
    has_drawn_from_empty: bool = False


@dataclass
class GameView:
    """Observable game state from one player's perspective."""
    my_player: int
    my_life: int
    opp_life: int
    my_hand: list[CardInstance]
    my_battlefield: list[CardInstance]
    opp_battlefield: list[CardInstance]
    my_graveyard: list[CardInstance]
    opp_graveyard: list[CardInstance]
    opp_hand_size: int
    my_library_size: int
    opp_library_size: int
    stack: list[StackItem]
    step: Step
    active_player: int
    turn_number: int
    land_played_this_turn: bool
    starting_life: int


class Game:
    def __init__(
        self,
        decks: list[list[str]],
        agents: list[Agent],
        starting_life: int = 20,
        starting_hand_size: int = 7,
        first_player: int | None = None,
        verbose: bool = False,
    ):
        self.agents = agents
        self.starting_life = starting_life
        self.starting_hand_size = starting_hand_size
        self.verbose = verbose
        self.next_instance_id = 0
        self.game_over = False
        self.result: GameResult | None = None
        self.loser: int | None = None
        self.turn_number = 0
        self.step = Step.UNTAP
        self.stack: list[StackItem] = []
        self.active_player = first_player if first_player is not None else rng.randint(0, 1)
        self.first_player_first_turn = True
        self.attacking_creatures: list[CardInstance] = []
        self.block_assignments: dict[int, int] = {}  # attacker_id -> blocker_id
        self.pending_attacker_blocks: list[CardInstance] = []
        self.available_blockers: list[CardInstance] = []
        self.decision_callbacks: list = []  # (player, view, legal_actions, chosen_action) log

        self.players = []
        for i, deck_list in enumerate(decks):
            library = []
            for card_name in deck_list:
                card_def = CARD_REGISTRY[card_name]
                inst = CardInstance(card_def=card_def, instance_id=self._next_id(), owner=i)
                library.append(inst)
            rng.shuffle(library)
            self.players.append(PlayerState(
                life=starting_life,
                library=library,
                hand=[],
                battlefield=[],
                graveyard=[],
            ))

    def _next_id(self) -> int:
        id_ = self.next_instance_id
        self.next_instance_id += 1
        return id_

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def get_view(self, player: int) -> GameView:
        me = self.players[player]
        opp = self.players[1 - player]
        return GameView(
            my_player=player,
            my_life=me.life,
            opp_life=opp.life,
            my_hand=list(me.hand),
            my_battlefield=list(me.battlefield),
            opp_battlefield=list(opp.battlefield),
            my_graveyard=list(me.graveyard),
            opp_graveyard=list(opp.graveyard),
            opp_hand_size=len(opp.hand),
            my_library_size=len(me.library),
            opp_library_size=len(opp.library),
            stack=list(self.stack),
            step=self.step,
            active_player=self.active_player,
            turn_number=self.turn_number,
            land_played_this_turn=me.land_played_this_turn,
            starting_life=self.starting_life,
        )

    # --- Drawing ---

    def draw_card(self, player: int) -> CardInstance | None:
        ps = self.players[player]
        if not ps.library:
            ps.has_drawn_from_empty = True
            self._log(f"  Player {player} tries to draw from empty library!")
            return None
        card = ps.library.pop()
        ps.hand.append(card)
        self._log(f"  Player {player} draws {card.name}")
        return card

    def draw_cards(self, player: int, count: int):
        for _ in range(count):
            if self.game_over:
                return
            self.draw_card(player)
            self.check_state_based_actions()

    # --- State-Based Actions ---

    def check_state_based_actions(self) -> bool:
        changed = False
        for i, ps in enumerate(self.players):
            if ps.life <= 0 and not self.game_over:
                self._end_game(1 - i)
                changed = True
            if ps.has_drawn_from_empty and not self.game_over:
                self._end_game(1 - i)
                changed = True

        for i, ps in enumerate(self.players):
            dead = [c for c in ps.battlefield if c.is_creature and c.damage_marked >= c.toughness]
            for c in dead:
                ps.battlefield.remove(c)
                ps.graveyard.append(c)
                c.damage_marked = 0
                c.tapped = False
                self._log(f"  {c.name}#{c.instance_id} dies (SBA)")
                changed = True

        return changed

    def _end_game(self, winner: int):
        if self.game_over:
            return
        self.game_over = True
        self.result = GameResult(winner)
        self.loser = 1 - winner
        self._log(f"  >>> Player {winner} wins! <<<")

    # --- Mana ---

    def _count_untapped_mountains(self, player: int) -> int:
        return sum(1 for c in self.players[player].battlefield
                   if c.name == "Mountain" and not c.tapped)

    def _tap_mountain(self, player: int):
        for c in self.players[player].battlefield:
            if c.name == "Mountain" and not c.tapped:
                c.tapped = True
                self.players[player].mana_pool += 1
                return
        raise RuntimeError("No untapped mountain to tap")

    def _pay_mana(self, player: int, cost_red: int):
        ps = self.players[player]
        while ps.mana_pool < cost_red:
            self._tap_mountain(player)
        ps.mana_pool -= cost_red

    def _empty_mana_pools(self):
        for ps in self.players:
            ps.mana_pool = 0

    # --- Legal Actions ---

    def get_legal_actions(self, player: int) -> list[Action]:
        actions: list[Action] = [Action(ActionType.PASS_PRIORITY)]
        ps = self.players[player]
        is_active = player == self.active_player
        in_main = self.step in MAIN_PHASE_STEPS
        stack_empty = len(self.stack) == 0

        if in_main and stack_empty and is_active:
            if not ps.land_played_this_turn:
                for card in ps.hand:
                    if card.is_land:
                        actions.append(Action(ActionType.PLAY_LAND, card=card))
                        break  # all mountains are identical, only offer once

        can_pay_red = self._count_untapped_mountains(player) > 0 or ps.mana_pool > 0

        if can_pay_red:
            for card in ps.hand:
                if card.name == "Lightning Bolt":
                    targets = self._get_bolt_targets(player)
                    for t in targets:
                        actions.append(Action(ActionType.CAST_SPELL, card=card, targets=(t,)))
                    break  # all bolts identical

            if in_main and stack_empty and is_active:
                for card in ps.hand:
                    if card.name == "Raging Goblin":
                        actions.append(Action(ActionType.CAST_SPELL, card=card))
                        break

        return actions

    def _get_bolt_targets(self, caster: int) -> list[Target]:
        targets = []
        targets.append(Target(TargetType.PLAYER, 1 - caster))
        targets.append(Target(TargetType.PLAYER, caster))
        for ps in self.players:
            for c in ps.battlefield:
                if c.is_creature:
                    targets.append(Target(TargetType.CREATURE, c.instance_id))
        return targets

    def get_attacker_options(self, player: int) -> list[Action]:
        ps = self.players[player]
        attackable = [c for c in ps.battlefield if c.can_attack()]
        if not attackable:
            return [Action(ActionType.DECLARE_ATTACKERS, attackers=())]

        subsets = []
        n = len(attackable)
        for mask in range(1 << n):
            ids = tuple(attackable[i].instance_id for i in range(n) if mask & (1 << i))
            subsets.append(Action(ActionType.DECLARE_ATTACKERS, attackers=ids))
        return subsets

    def get_blocker_options(self, defender: int, attacker: CardInstance) -> list[Action]:
        actions = [Action(
            ActionType.DECLARE_BLOCKERS,
            blocker_id=None,
            attacker_to_block_id=attacker.instance_id,
        )]
        for c in self.available_blockers:
            actions.append(Action(
                ActionType.DECLARE_BLOCKERS,
                blocker_id=c.instance_id,
                attacker_to_block_id=attacker.instance_id,
            ))
        return actions

    # --- Action Execution ---

    def execute_action(self, player: int, action: Action):
        self._log(f"  P{player}: {action}")

        if action.action_type == ActionType.PASS_PRIORITY:
            return

        if action.action_type == ActionType.PLAY_LAND:
            ps = self.players[player]
            card = action.card
            ps.hand.remove(card)
            ps.battlefield.append(card)
            ps.land_played_this_turn = True
            return

        if action.action_type == ActionType.CAST_SPELL:
            card = action.card
            ps = self.players[player]
            ps.hand.remove(card)
            cost_red = card.card_def.mana_cost.red
            if cost_red > 0:
                self._pay_mana(player, cost_red)

            if card.is_creature:
                # creatures go on stack then resolve to battlefield
                self.stack.append(StackItem(card=card, controller=player, targets=list(action.targets)))
            elif card.is_instant or card.card_def.is_sorcery:
                self.stack.append(StackItem(card=card, controller=player, targets=list(action.targets)))
            return

        if action.action_type == ActionType.CONCEDE:
            self._end_game(1 - player)
            return

    # --- Stack Resolution ---

    def resolve_top_of_stack(self):
        if not self.stack:
            return
        item = self.stack.pop()
        card = item.card
        controller = item.controller
        self._log(f"  Resolving: {card.name}")

        if card.name == "Lightning Bolt":
            self._resolve_bolt(item)
            self.players[controller].graveyard.append(card)
        elif card.name == "Raging Goblin":
            self._resolve_creature(item)
        else:
            self.players[controller].graveyard.append(card)

    def _resolve_bolt(self, item: StackItem):
        if not item.targets:
            return
        target = item.targets[0]
        if target.target_type == TargetType.PLAYER:
            pid = target.target_id
            self.players[pid].life -= 3
            self._log(f"  Lightning Bolt deals 3 damage to Player {pid} (now {self.players[pid].life})")
        elif target.target_type == TargetType.CREATURE:
            creature = self._find_creature_on_battlefield(target.target_id)
            if creature:
                creature.damage_marked += 3
                self._log(f"  Lightning Bolt deals 3 damage to {creature.name}#{creature.instance_id}")
            else:
                self._log(f"  Lightning Bolt fizzles (target gone)")

    def _resolve_creature(self, item: StackItem):
        card = item.card
        controller = item.controller
        card.summoning_sick = True
        card.tapped = False
        card.damage_marked = 0
        self.players[controller].battlefield.append(card)
        self._log(f"  {card.name}#{card.instance_id} enters the battlefield for Player {controller}")

    def _find_creature_on_battlefield(self, instance_id: int) -> CardInstance | None:
        for ps in self.players:
            for c in ps.battlefield:
                if c.instance_id == instance_id and c.is_creature:
                    return c
        return None

    # --- Priority System ---

    def priority_loop(self):
        current = self.active_player
        consecutive_passes = 0

        while not self.game_over:
            while self.check_state_based_actions():
                if self.game_over:
                    return

            legal = self.get_legal_actions(current)

            if len(legal) == 1 and legal[0].action_type == ActionType.PASS_PRIORITY:
                consecutive_passes += 1
            else:
                view = self.get_view(current)
                view_with_actions = view
                action = self.agents[current].choose_action(view_with_actions, legal)

                if action.action_type == ActionType.PASS_PRIORITY:
                    consecutive_passes += 1
                else:
                    self.execute_action(current, action)
                    consecutive_passes = 0
                    continue  # same player keeps priority

            if consecutive_passes >= 2:
                if self.stack:
                    self.resolve_top_of_stack()
                    consecutive_passes = 0
                    current = self.active_player
                else:
                    return  # phase/step ends
            else:
                current = 1 - current

    # --- Turn Structure ---

    def run_game(self) -> GameResult:
        for i in range(2):
            self.draw_cards(i, self.starting_hand_size)

        self.mulligan_phase()

        if self.game_over:
            return self.result

        while not self.game_over:
            self.turn_number += 1
            if self.turn_number > MAX_TURNS:
                self.game_over = True
                self.result = GameResult.DRAW
                return self.result
            self._log(f"\n=== Turn {self.turn_number} (Player {self.active_player}) ===")
            self.run_turn()
            if not self.game_over:
                self.active_player = 1 - self.active_player

        for i, agent in enumerate(self.agents):
            agent.game_over_callback(self.result, i)

        return self.result

    def run_turn(self):
        self.untap_step()
        if self.game_over:
            return
        self.upkeep_step()
        if self.game_over:
            return
        self.draw_step()
        if self.game_over:
            return
        self.main_phase(Step.PRECOMBAT_MAIN)
        if self.game_over:
            return
        self.combat_phase()
        if self.game_over:
            return
        self.main_phase(Step.POSTCOMBAT_MAIN)
        if self.game_over:
            return
        self.end_step()
        if self.game_over:
            return
        self.cleanup_step()

    def untap_step(self):
        self.step = Step.UNTAP
        ap = self.active_player
        ps = self.players[ap]
        for c in ps.battlefield:
            c.tapped = False
            if c.is_creature:
                c.summoning_sick = False
        ps.land_played_this_turn = False
        self._empty_mana_pools()

    def upkeep_step(self):
        self.step = Step.UPKEEP
        self._empty_mana_pools()
        self.priority_loop()
        self._empty_mana_pools()

    def draw_step(self):
        self.step = Step.DRAW
        self._empty_mana_pools()
        if self.first_player_first_turn and self.turn_number == 1:
            self.first_player_first_turn = False
            self._log(f"  Player {self.active_player} skips first draw")
        else:
            self.draw_card(self.active_player)
            self.check_state_based_actions()
            if self.game_over:
                return
        self.priority_loop()
        self._empty_mana_pools()

    def main_phase(self, step: Step):
        self.step = step
        self._empty_mana_pools()
        self.priority_loop()
        self._empty_mana_pools()

    def combat_phase(self):
        self.begin_combat_step()
        if self.game_over:
            return
        self.declare_attackers_step()
        if self.game_over:
            return
        if self.attacking_creatures:
            self.declare_blockers_step()
            if self.game_over:
                return
            self.combat_damage_step()
            if self.game_over:
                return
        self.end_combat_step()

    def begin_combat_step(self):
        self.step = Step.BEGIN_COMBAT
        self._empty_mana_pools()
        self.attacking_creatures = []
        self.block_assignments = {}
        self.priority_loop()
        self._empty_mana_pools()

    def declare_attackers_step(self):
        self.step = Step.DECLARE_ATTACKERS
        self._empty_mana_pools()

        ap = self.active_player
        options = self.get_attacker_options(ap)

        if len(options) == 1 and not options[0].attackers:
            self.attacking_creatures = []
            self._log(f"  Player {ap} has no creatures to attack with")
        else:
            view = self.get_view(ap)
            action = self.agents[ap].choose_action(view, options)
            self._log(f"  P{ap}: {action}")

            self.attacking_creatures = []
            ps = self.players[ap]
            for cid in action.attackers:
                for c in ps.battlefield:
                    if c.instance_id == cid:
                        c.tapped = True
                        self.attacking_creatures.append(c)
                        break

        if self.attacking_creatures:
            self.priority_loop()
        self._empty_mana_pools()

    def declare_blockers_step(self):
        self.step = Step.DECLARE_BLOCKERS
        self._empty_mana_pools()

        defender = 1 - self.active_player
        self.block_assignments = {}
        self.available_blockers = [c for c in self.players[defender].battlefield if c.can_block()]

        for attacker in self.attacking_creatures:
            if self.game_over:
                return
            if not self.available_blockers:
                break

            options = self.get_blocker_options(defender, attacker)
            if len(options) == 1:
                continue  # only "don't block"

            view = self.get_view(defender)
            action = self.agents[defender].choose_action(view, options)
            self._log(f"  P{defender}: {action}")

            if action.blocker_id is not None:
                self.block_assignments[attacker.instance_id] = action.blocker_id
                self.available_blockers = [b for b in self.available_blockers
                                           if b.instance_id != action.blocker_id]

        self.priority_loop()
        self._empty_mana_pools()

    def combat_damage_step(self):
        self.step = Step.COMBAT_DAMAGE
        self._empty_mana_pools()

        defender_player = 1 - self.active_player

        for attacker in self.attacking_creatures:
            if attacker not in self.players[self.active_player].battlefield:
                continue  # died to bolt during combat

            blocker_id = self.block_assignments.get(attacker.instance_id)
            if blocker_id is not None:
                blocker = self._find_creature_on_battlefield(blocker_id)
                if blocker:
                    attacker.damage_marked += blocker.power
                    blocker.damage_marked += attacker.power
                    self._log(f"  {attacker.name}#{attacker.instance_id} and {blocker.name}#{blocker.instance_id} exchange damage")
                else:
                    self.players[defender_player].life -= attacker.power
                    self._log(f"  {attacker.name}#{attacker.instance_id} deals {attacker.power} to Player {defender_player} (blocker gone)")
            else:
                self.players[defender_player].life -= attacker.power
                self._log(f"  {attacker.name}#{attacker.instance_id} deals {attacker.power} to Player {defender_player} (now {self.players[defender_player].life})")

        self.check_state_based_actions()
        if self.game_over:
            return
        self.priority_loop()
        self._empty_mana_pools()

    def end_combat_step(self):
        self.step = Step.END_COMBAT
        self._empty_mana_pools()
        self.attacking_creatures = []
        self.block_assignments = {}
        self.priority_loop()
        self._empty_mana_pools()

    def end_step(self):
        self.step = Step.END_STEP
        self._empty_mana_pools()
        self.priority_loop()
        self._empty_mana_pools()

    def cleanup_step(self):
        self.step = Step.CLEANUP
        self._empty_mana_pools()

        ap = self.active_player
        ps = self.players[ap]
        while len(ps.hand) > 7:
            view = self.get_view(ap)
            discard_actions = []
            for i, card in enumerate(ps.hand):
                discard_actions.append(Action(
                    ActionType.CHOOSE_BOTTOM_CARD,
                    card=card,
                    hand_index=i,
                ))
            action = self.agents[ap].choose_action(view, discard_actions)
            card_to_discard = ps.hand[action.hand_index]
            ps.hand.remove(card_to_discard)
            ps.graveyard.append(card_to_discard)
            self._log(f"  Player {ap} discards {card_to_discard.name}")

        for ps in self.players:
            for c in ps.battlefield:
                if c.is_creature:
                    c.damage_marked = 0

    # --- Mulligan ---

    def mulligan_phase(self):
        for player_offset in range(2):
            player = (self.active_player + player_offset) % 2
            mulligan_count = 0

            while not self.game_over:
                if mulligan_count >= self.starting_hand_size:
                    break

                view = self.get_view(player)
                keep_action = Action(ActionType.MULLIGAN_KEEP)
                mull_action = Action(ActionType.MULLIGAN_TAKE)
                legal = [keep_action, mull_action]

                action = self.agents[player].choose_action(view, legal)

                if action.action_type == ActionType.MULLIGAN_KEEP:
                    self._log(f"  Player {player} keeps (mulliganed {mulligan_count} times)")
                    break
                else:
                    mulligan_count += 1
                    self._log(f"  Player {player} mulligans (count: {mulligan_count})")
                    ps = self.players[player]
                    ps.library.extend(ps.hand)
                    ps.hand.clear()
                    rng.shuffle(ps.library)
                    self.draw_cards(player, self.starting_hand_size)

            ps = self.players[player]
            for _ in range(mulligan_count):
                if not ps.hand:
                    break
                view = self.get_view(player)
                bottom_actions = []
                for i, card in enumerate(ps.hand):
                    bottom_actions.append(Action(
                        ActionType.CHOOSE_BOTTOM_CARD,
                        card=card,
                        hand_index=i,
                    ))
                action = self.agents[player].choose_action(view, bottom_actions)
                card_to_bottom = ps.hand[action.hand_index]
                ps.hand.remove(card_to_bottom)
                ps.library.insert(0, card_to_bottom)
                self._log(f"  Player {player} puts {card_to_bottom.name} on bottom")


def make_deck(mountains: int, bolts: int, goblins: int) -> list[str]:
    return (["Mountain"] * mountains +
            ["Lightning Bolt"] * bolts +
            ["Raging Goblin"] * goblins)
