from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from mtg_ai.cards.registry import CardRegistry
from mtg_ai.core.actions import (
    ActivateManaAbilityAction,
    CastSpellAction,
    ConcedeAction,
    DeclareAttackersAction,
    DeclareBlockersAction,
    GameAction,
    PassPriorityAction,
    PlayLandAction,
    TargetRef,
)
from mtg_ai.core.cards import CardInstance
from mtg_ai.core.enums import MAIN_PHASE_STEPS, Step, TURN_STEPS
from mtg_ai.core.events import GameEvent
from mtg_ai.core.sba import apply_state_based_actions
from mtg_ai.core.state import CombatState, GameState, PermanentState, PlayerState, StackObject, TurnState


@dataclass(frozen=True)
class GameConfig:
    starting_life: int = 20
    opening_hand_size: int = 7
    max_mulligans: int = 7
    random_seed: int = 0
    expose_mana_actions: bool = False


MulliganDecider = Callable[[int, Sequence[str], int], bool]
BottomChooser = Callable[[int, Sequence[str], int], Sequence[str]]


class RulesEngine:
    def __init__(self, registry: CardRegistry | None = None, config: GameConfig | None = None) -> None:
        self.registry = registry or CardRegistry()
        self.config = config or GameConfig()
        self._random = random.Random(self.config.random_seed)
        self._next_stack_id = 1
        self._next_permanent_id = 1

    def create_game(
        self,
        decks: dict[int, list[str]],
        starting_player_id: int | None = None,
        mulligan_decider: MulliganDecider | None = None,
        bottom_chooser: BottomChooser | None = None,
        preserve_deck_order: bool = False,
    ) -> tuple[GameState, list[GameEvent]]:
        if sorted(decks) != [0, 1]:
            raise ValueError("Engine currently supports exactly two players with ids 0 and 1.")
        card_instances: dict[str, CardInstance] = {}
        players: dict[int, PlayerState] = {}
        for player_id in (0, 1):
            deck = decks[player_id]
            for card_name in deck:
                if not self.registry.has(card_name):
                    raise ValueError(f"Illegal card in deck: {card_name}")
            library: list[str] = []
            for idx, card_name in enumerate(deck):
                card_id = f"p{player_id}c{idx}"
                card_instances[card_id] = CardInstance(card_id=card_id, definition_name=card_name, owner_id=player_id)
                library.append(card_id)
            if not preserve_deck_order:
                self._random.shuffle(library)
            players[player_id] = PlayerState(player_id=player_id, life=self.config.starting_life, library=library)

        if starting_player_id is None:
            starting_player_id = self._random.choice([0, 1])
        other_player = 1 - starting_player_id

        state = GameState(
            players=players,
            card_instances=card_instances,
            permanents={},
            stack=[],
            turn=TurnState(
                turn_number=1,
                active_player_id=starting_player_id,
                priority_player_id=starting_player_id,
                step=Step.UNTAP,
                starting_player_id=starting_player_id,
                passed_priority_in_row=0,
            ),
        )
        events: list[GameEvent] = []

        for player_id in (0, 1):
            for _ in range(self.config.opening_hand_size):
                self._draw_card(state, player_id)

        self._handle_mulligans(state, mulligan_decider, bottom_chooser)
        self._check_sba(state, events)
        self._begin_step(state, events)

        if state.turn.active_player_id != starting_player_id:
            state.turn.active_player_id = starting_player_id
            state.turn.priority_player_id = starting_player_id
        if state.turn.active_player_id == starting_player_id and state.turn.step == Step.UPKEEP:
            state.turn.priority_player_id = starting_player_id
        if other_player not in state.players:
            raise ValueError("Invalid player setup")
        return state, events

    def legal_actions(self, state: GameState, player_id: int) -> list[GameAction]:
        if state.game_over:
            return []
        if player_id != state.turn.priority_player_id:
            return []
        legal: list[GameAction] = []

        if state.turn.step == Step.DECLARE_ATTACKERS:
            if player_id != state.turn.active_player_id:
                return legal
            legal.extend(self._legal_declare_attackers(state, player_id))
            return legal
        if state.turn.step == Step.DECLARE_BLOCKERS:
            if player_id == state.turn.active_player_id:
                return legal
            legal.extend(self._legal_declare_blockers(state, player_id))
            return legal

        legal.append(PassPriorityAction(actor_id=player_id))
        legal.extend(self._legal_land_plays(state, player_id))
        if self.config.expose_mana_actions:
            legal.extend(self._legal_mana_abilities(state, player_id))
        legal.extend(self._legal_cast_spells(state, player_id))
        return legal

    def apply_action(self, state: GameState, action: GameAction) -> list[GameEvent]:
        if state.game_over:
            return []
        if action.actor_id != state.turn.priority_player_id:
            raise ValueError("Action actor does not currently have priority.")

        events: list[GameEvent] = [self._event(state, "action", {"action_type": action.action_type, "actor_id": action.actor_id})]
        if isinstance(action, ConcedeAction):
            state.game_over = True
            state.winner = 1 - action.actor_id
            state.result = f"player_{state.winner}_wins"
            events.append(self._event(state, "game_over", {"reason": "concede", "winner": state.winner}))
            return events
        if isinstance(action, PassPriorityAction):
            self._apply_pass_priority(state, events)
            return events
        if isinstance(action, PlayLandAction):
            self._apply_play_land(state, action, events)
            return events
        if isinstance(action, ActivateManaAbilityAction):
            self._apply_activate_mana(state, action, events)
            return events
        if isinstance(action, CastSpellAction):
            self._apply_cast_spell(state, action, events)
            return events
        if isinstance(action, DeclareAttackersAction):
            self._apply_declare_attackers(state, action, events)
            return events
        if isinstance(action, DeclareBlockersAction):
            self._apply_declare_blockers(state, action, events)
            return events
        raise ValueError(f"Unsupported action: {type(action)}")

    def _event(self, state: GameState, kind: str, payload: dict[str, object]) -> GameEvent:
        state.event_index += 1
        return GameEvent(index=state.event_index, kind=kind, payload=payload)

    def _handle_mulligans(
        self,
        state: GameState,
        mulligan_decider: MulliganDecider | None,
        bottom_chooser: BottomChooser | None,
    ) -> None:
        for player_id in (0, 1):
            while state.players[player_id].mulligans_taken < self.config.max_mulligans:
                hand_names = [state.card_instances[cid].definition_name for cid in state.players[player_id].hand]
                wants_mulligan = mulligan_decider(player_id, hand_names, state.players[player_id].mulligans_taken) if mulligan_decider else False
                if not wants_mulligan:
                    break
                player = state.players[player_id]
                player.library.extend(player.hand)
                player.hand.clear()
                self._random.shuffle(player.library)
                player.mulligans_taken += 1
                for _ in range(self.config.opening_hand_size):
                    self._draw_card(state, player_id)

        for player_id in (0, 1):
            player = state.players[player_id]
            to_bottom = player.mulligans_taken
            if to_bottom <= 0:
                continue
            hand = list(player.hand)
            if bottom_chooser:
                chosen = list(bottom_chooser(player_id, [state.card_instances[cid].definition_name for cid in hand], to_bottom))
                if len(chosen) != to_bottom:
                    raise ValueError("Bottom chooser returned incorrect card count.")
                for card_id in chosen:
                    if card_id not in player.hand:
                        raise ValueError("Bottom chooser selected unknown card.")
                    player.hand.remove(card_id)
                    player.library.insert(0, card_id)
            else:
                for _ in range(to_bottom):
                    card_id = player.hand.pop()
                    player.library.insert(0, card_id)

    def _draw_card(self, state: GameState, player_id: int) -> None:
        player = state.players[player_id]
        if not player.library:
            state.pending_draw_loss.add(player_id)
            return
        player.hand.append(player.library.pop())

    def _check_sba(self, state: GameState, events: list[GameEvent]) -> None:
        applied = apply_state_based_actions(state, self.registry)
        for entry in applied:
            events.append(self._event(state, "state_based_action", {"detail": entry}))

    def _legal_land_plays(self, state: GameState, player_id: int) -> list[PlayLandAction]:
        player = state.players[player_id]
        if player_id != state.turn.active_player_id:
            return []
        if state.turn.step not in MAIN_PHASE_STEPS:
            return []
        if state.stack:
            return []
        if player.lands_played_this_turn >= 1:
            return []
        legal: list[PlayLandAction] = []
        for card_id in player.hand:
            definition = self.registry.get(state.card_instances[card_id].definition_name)
            if definition.is_land:
                legal.append(PlayLandAction(actor_id=player_id, card_id=card_id))
        return legal

    def _legal_mana_abilities(self, state: GameState, player_id: int) -> list[ActivateManaAbilityAction]:
        legal: list[ActivateManaAbilityAction] = []
        for permanent_id in state.players[player_id].battlefield:
            if permanent_id not in state.permanents:
                continue
            permanent = state.permanents[permanent_id]
            if permanent.tapped:
                continue
            definition = self.registry.get(state.card_instances[permanent.card_id].definition_name)
            if definition.has_mana_ability:
                legal.append(ActivateManaAbilityAction(actor_id=player_id, permanent_id=permanent_id))
        return legal

    def _legal_cast_spells(self, state: GameState, player_id: int) -> list[CastSpellAction]:
        legal: list[CastSpellAction] = []
        player = state.players[player_id]
        untapped_mountains = [
            permanent_id
            for permanent_id in player.battlefield
            if permanent_id in state.permanents
            and not state.permanents[permanent_id].tapped
            and self.registry.get(state.card_instances[state.permanents[permanent_id].card_id].definition_name).has_mana_ability
        ]
        for card_id in player.hand:
            definition = self.registry.get(state.card_instances[card_id].definition_name)
            if definition.is_land:
                continue
            if definition.is_creature and (player_id != state.turn.active_player_id or state.turn.step not in MAIN_PHASE_STEPS or state.stack):
                continue
            total_available = player.mana_pool_red + len(untapped_mountains)
            if total_available < definition.mana_cost.red:
                continue

            mana_options = [tuple()]
            if definition.mana_cost.red > player.mana_pool_red:
                needed = definition.mana_cost.red - player.mana_pool_red
                mana_options = [tuple([source]) for source in untapped_mountains[:needed]]
                if not mana_options:
                    continue

            if definition.name == "Lightning Bolt":
                targets: list[TargetRef] = [TargetRef(kind="player", id=1 - player_id)]
                for permanent_id in state.permanents:
                    targets.append(TargetRef(kind="permanent", id=permanent_id))
                for target in targets:
                    if not self._is_legal_bolt_target(state, target):
                        continue
                    for mana_sources in mana_options:
                        legal.append(
                            CastSpellAction(
                                actor_id=player_id,
                                card_id=card_id,
                                targets=(target,),
                                tap_mana_sources=mana_sources,
                            )
                        )
            else:
                for mana_sources in mana_options:
                    legal.append(
                        CastSpellAction(
                            actor_id=player_id,
                            card_id=card_id,
                            targets=tuple(),
                            tap_mana_sources=mana_sources,
                        )
                    )
        return legal

    def _legal_declare_attackers(self, state: GameState, player_id: int) -> list[DeclareAttackersAction]:
        candidates: list[str] = []
        for permanent_id in state.players[player_id].battlefield:
            if permanent_id not in state.permanents:
                continue
            permanent = state.permanents[permanent_id]
            definition = self.registry.get(state.card_instances[permanent.card_id].definition_name)
            if not definition.is_creature or permanent.tapped:
                continue
            if "haste" not in definition.keywords and permanent.entered_turn_number == state.turn.turn_number:
                continue
            candidates.append(permanent_id)

        if len(candidates) <= 8:
            legal: list[DeclareAttackersAction] = []
            for size in range(len(candidates) + 1):
                for subset in itertools.combinations(candidates, size):
                    legal.append(DeclareAttackersAction(actor_id=player_id, attacker_ids=tuple(subset)))
            return legal

        legal = [DeclareAttackersAction(actor_id=player_id, attacker_ids=tuple())]
        for attacker in candidates:
            legal.append(DeclareAttackersAction(actor_id=player_id, attacker_ids=(attacker,)))
        legal.append(DeclareAttackersAction(actor_id=player_id, attacker_ids=tuple(candidates)))
        return legal

    def _legal_declare_blockers(self, state: GameState, player_id: int) -> list[DeclareBlockersAction]:
        attackers = [attacker for attacker in state.combat.attackers if attacker in state.permanents]
        blockers = []
        for permanent_id in state.players[player_id].battlefield:
            if permanent_id not in state.permanents:
                continue
            permanent = state.permanents[permanent_id]
            definition = self.registry.get(state.card_instances[permanent.card_id].definition_name)
            if definition.is_creature and not permanent.tapped:
                blockers.append(permanent_id)

        legal: list[DeclareBlockersAction] = [DeclareBlockersAction(actor_id=player_id, blocks=tuple())]
        if len(attackers) * max(1, len(blockers)) > 12:
            greedy_blocks: list[tuple[str, str]] = []
            for attacker_id, blocker_id in zip(attackers, blockers):
                greedy_blocks.append((attacker_id, blocker_id))
            if greedy_blocks:
                legal.append(DeclareBlockersAction(actor_id=player_id, blocks=tuple(greedy_blocks)))
            return legal

        def rec(
            idx: int,
            available_blockers: list[str],
            acc: list[tuple[str, str]],
        ) -> None:
            if idx >= len(attackers):
                if acc:
                    legal.append(DeclareBlockersAction(actor_id=player_id, blocks=tuple(acc)))
                return
            rec(idx + 1, available_blockers, acc)
            attacker_id = attackers[idx]
            for blocker_id in list(available_blockers):
                new_available = [b for b in available_blockers if b != blocker_id]
                acc.append((attacker_id, blocker_id))
                rec(idx + 1, new_available, acc)
                acc.pop()

        rec(0, blockers, [])
        return legal

    def _apply_pass_priority(self, state: GameState, events: list[GameEvent]) -> None:
        state.turn.passed_priority_in_row += 1
        if state.turn.passed_priority_in_row < len(state.players):
            state.turn.priority_player_id = 1 - state.turn.priority_player_id
            events.append(self._event(state, "priority_passed", {"next_priority_player_id": state.turn.priority_player_id}))
            return

        state.turn.passed_priority_in_row = 0
        if state.stack:
            self._resolve_top_of_stack(state, events)
            if state.game_over:
                return
            self._check_sba(state, events)
            if state.game_over:
                return
            state.turn.priority_player_id = state.turn.active_player_id
            events.append(self._event(state, "priority_reset", {"priority_player_id": state.turn.priority_player_id}))
            return

        self._advance_step(state, events)

    def _apply_play_land(self, state: GameState, action: PlayLandAction, events: list[GameEvent]) -> None:
        legal_card_ids = {act.card_id for act in self._legal_land_plays(state, action.actor_id)}
        if action.card_id not in legal_card_ids:
            raise ValueError("Illegal land play.")
        player = state.players[action.actor_id]
        player.hand.remove(action.card_id)
        permanent_id = f"perm{self._next_permanent_id}"
        self._next_permanent_id += 1
        permanent = PermanentState(
            permanent_id=permanent_id,
            card_id=action.card_id,
            controller_id=action.actor_id,
            tapped=False,
            damage_marked=0,
            entered_turn_number=state.turn.turn_number,
        )
        state.permanents[permanent_id] = permanent
        player.battlefield.append(permanent_id)
        player.lands_played_this_turn += 1
        state.turn.passed_priority_in_row = 0
        state.turn.priority_player_id = action.actor_id
        events.append(self._event(state, "land_played", {"player_id": action.actor_id, "card_id": action.card_id, "permanent_id": permanent_id}))

    def _apply_activate_mana(self, state: GameState, action: ActivateManaAbilityAction, events: list[GameEvent]) -> None:
        legal_permanents = {act.permanent_id for act in self._legal_mana_abilities(state, action.actor_id)}
        if action.permanent_id not in legal_permanents:
            raise ValueError("Illegal mana ability activation.")
        permanent = state.permanents[action.permanent_id]
        permanent.tapped = True
        state.players[action.actor_id].mana_pool_red += 1
        state.turn.passed_priority_in_row = 0
        state.turn.priority_player_id = action.actor_id
        events.append(self._event(state, "mana_added", {"player_id": action.actor_id, "permanent_id": action.permanent_id, "red_added": 1}))

    def _apply_cast_spell(self, state: GameState, action: CastSpellAction, events: list[GameEvent]) -> None:
        legal_actions = self._legal_cast_spells(state, action.actor_id)
        serialized = {
            (act.card_id, tuple(act.tap_mana_sources), tuple(t.to_key() for t in act.targets))
            for act in legal_actions
        }
        key = (action.card_id, tuple(action.tap_mana_sources), tuple(t.to_key() for t in action.targets))
        if key not in serialized:
            raise ValueError("Illegal spell cast.")

        player = state.players[action.actor_id]
        definition = self.registry.get(state.card_instances[action.card_id].definition_name)
        for permanent_id in action.tap_mana_sources:
            permanent = state.permanents[permanent_id]
            permanent.tapped = True
            player.mana_pool_red += 1
        if player.mana_pool_red < definition.mana_cost.red:
            raise ValueError("Insufficient mana to pay cost.")
        player.mana_pool_red -= definition.mana_cost.red

        player.hand.remove(action.card_id)
        stack_object = StackObject(
            stack_id=f"stack{self._next_stack_id}",
            card_id=action.card_id,
            controller_id=action.actor_id,
            targets=action.targets,
        )
        self._next_stack_id += 1
        state.stack.append(stack_object)
        state.turn.passed_priority_in_row = 0
        state.turn.priority_player_id = 1 - action.actor_id
        events.append(self._event(state, "spell_cast", {"player_id": action.actor_id, "card_id": action.card_id, "stack_id": stack_object.stack_id}))

    def _apply_declare_attackers(self, state: GameState, action: DeclareAttackersAction, events: list[GameEvent]) -> None:
        legal = {tuple(act.attacker_ids) for act in self._legal_declare_attackers(state, action.actor_id)}
        if tuple(action.attacker_ids) not in legal:
            raise ValueError("Illegal attackers declaration.")
        state.combat = CombatState(attackers=list(action.attacker_ids), blocks={}, defending_player_id=1 - action.actor_id)
        for permanent_id in action.attacker_ids:
            state.permanents[permanent_id].tapped = True
        events.append(self._event(state, "attackers_declared", {"player_id": action.actor_id, "attackers": list(action.attacker_ids)}))
        state.turn.step = Step.DECLARE_BLOCKERS
        state.turn.priority_player_id = 1 - action.actor_id
        state.turn.passed_priority_in_row = 0
        events.append(self._event(state, "step_changed", {"step": state.turn.step.value}))

    def _apply_declare_blockers(self, state: GameState, action: DeclareBlockersAction, events: list[GameEvent]) -> None:
        legal = {tuple(act.blocks) for act in self._legal_declare_blockers(state, action.actor_id)}
        if tuple(action.blocks) not in legal:
            raise ValueError("Illegal blockers declaration.")
        blocks: dict[str, list[str]] = {}
        for attacker, blocker in action.blocks:
            blocks.setdefault(attacker, []).append(blocker)
        state.combat.blocks = blocks
        events.append(self._event(state, "blockers_declared", {"player_id": action.actor_id, "blocks": [list(b) for b in action.blocks]}))
        self._advance_step(state, events)

    def _resolve_top_of_stack(self, state: GameState, events: list[GameEvent]) -> None:
        stack_object = state.stack.pop()
        card_instance = state.card_instances[stack_object.card_id]
        definition = self.registry.get(card_instance.definition_name)
        controller = state.players[stack_object.controller_id]
        if definition.name == "Lightning Bolt":
            legal_targets = [target for target in stack_object.targets if self._is_legal_bolt_target(state, target)]
            if not legal_targets:
                controller.graveyard.append(stack_object.card_id)
                events.append(self._event(state, "spell_fizzled", {"card_id": stack_object.card_id, "stack_id": stack_object.stack_id}))
                return
            target = legal_targets[0]
            if target.kind == "player":
                state.players[int(target.id)].life -= 3
                events.append(self._event(state, "damage", {"target_player_id": int(target.id), "amount": 3}))
            else:
                permanent = state.permanents[str(target.id)]
                permanent.damage_marked += 3
                events.append(self._event(state, "damage", {"target_permanent_id": str(target.id), "amount": 3}))
            controller.graveyard.append(stack_object.card_id)
            events.append(self._event(state, "spell_resolved", {"card_id": stack_object.card_id, "stack_id": stack_object.stack_id}))
            return
        if definition.name == "Raging Goblin":
            permanent_id = f"perm{self._next_permanent_id}"
            self._next_permanent_id += 1
            permanent = PermanentState(
                permanent_id=permanent_id,
                card_id=stack_object.card_id,
                controller_id=stack_object.controller_id,
                tapped=False,
                damage_marked=0,
                entered_turn_number=state.turn.turn_number,
            )
            state.permanents[permanent_id] = permanent
            controller.battlefield.append(permanent_id)
            events.append(self._event(state, "spell_resolved", {"card_id": stack_object.card_id, "stack_id": stack_object.stack_id}))
            return
        raise ValueError(f"Unsupported resolving card: {definition.name}")

    def _is_legal_bolt_target(self, state: GameState, target: TargetRef) -> bool:
        if target.kind == "player":
            return int(target.id) in state.players
        permanent = state.permanents.get(str(target.id))
        if permanent is None:
            return False
        definition = self.registry.get(state.card_instances[permanent.card_id].definition_name)
        return definition.is_creature

    def _advance_step(self, state: GameState, events: list[GameEvent]) -> None:
        for player in state.players.values():
            player.mana_pool_red = 0
        current_index = TURN_STEPS.index(state.turn.step)
        if state.turn.step == Step.CLEANUP:
            state.turn.active_player_id = 1 - state.turn.active_player_id
            state.turn.turn_number += 1
            state.turn.step = Step.UNTAP
            state.combat = CombatState()
            events.append(self._event(state, "turn_changed", {"turn_number": state.turn.turn_number, "active_player_id": state.turn.active_player_id}))
            self._begin_step(state, events)
            return

        state.turn.step = TURN_STEPS[current_index + 1]
        state.combat.blocks = {}
        events.append(self._event(state, "step_changed", {"step": state.turn.step.value}))
        self._begin_step(state, events)

    def _begin_step(self, state: GameState, events: list[GameEvent]) -> None:
        step = state.turn.step
        active_player = state.players[state.turn.active_player_id]

        if step == Step.UNTAP:
            for permanent_id in list(active_player.battlefield):
                if permanent_id in state.permanents:
                    state.permanents[permanent_id].tapped = False
            active_player.lands_played_this_turn = 0
            state.turn.passed_priority_in_row = 0
            self._advance_step(state, events)
            return

        if step == Step.DRAW:
            if not (state.turn.turn_number == 1 and state.turn.active_player_id == state.turn.starting_player_id):
                self._draw_card(state, state.turn.active_player_id)
            self._check_sba(state, events)
            if state.game_over:
                return

        if step == Step.COMBAT_DAMAGE:
            self._resolve_combat_damage(state, events)
            self._check_sba(state, events)
            if state.game_over:
                return

        if step == Step.CLEANUP:
            while len(active_player.hand) > 7:
                discard = active_player.hand.pop()
                active_player.graveyard.append(discard)
                events.append(self._event(state, "discard", {"player_id": active_player.player_id, "card_id": discard}))
            for permanent in state.permanents.values():
                permanent.damage_marked = 0
            self._check_sba(state, events)
            if state.game_over:
                return
            self._advance_step(state, events)
            return

        if step == Step.DECLARE_ATTACKERS:
            state.turn.priority_player_id = state.turn.active_player_id
            state.turn.passed_priority_in_row = 0
            return
        if step == Step.DECLARE_BLOCKERS:
            state.turn.priority_player_id = 1 - state.turn.active_player_id
            state.turn.passed_priority_in_row = 0
            return

        state.turn.priority_player_id = state.turn.active_player_id
        state.turn.passed_priority_in_row = 0

    def _resolve_combat_damage(self, state: GameState, events: list[GameEvent]) -> None:
        if not state.combat.attackers:
            return
        defending_player_id = state.combat.defending_player_id
        if defending_player_id is None:
            return

        for attacker_id in list(state.combat.attackers):
            if attacker_id not in state.permanents:
                continue
            attacker = state.permanents[attacker_id]
            attacker_definition = self.registry.get(state.card_instances[attacker.card_id].definition_name)
            attacker_power = attacker_definition.power or 0
            blockers = [bid for bid in state.combat.blocks.get(attacker_id, []) if bid in state.permanents]
            if not blockers:
                state.players[defending_player_id].life -= attacker_power
                events.append(self._event(state, "combat_damage_player", {"attacker_id": attacker_id, "defending_player_id": defending_player_id, "amount": attacker_power}))
                continue
            first_blocker_id = blockers[0]
            blocker = state.permanents[first_blocker_id]
            blocker_definition = self.registry.get(state.card_instances[blocker.card_id].definition_name)
            blocker_power = blocker_definition.power or 0
            blocker.damage_marked += attacker_power
            attacker.damage_marked += blocker_power
            events.append(self._event(state, "combat_damage_creature", {"attacker_id": attacker_id, "blocker_id": first_blocker_id, "attacker_dealt": attacker_power, "blocker_dealt": blocker_power}))

