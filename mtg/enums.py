from enum import Enum, auto


class Phase(Enum):
    MULLIGAN = auto()
    BEGINNING = auto()
    PRECOMBAT_MAIN = auto()
    COMBAT = auto()
    POSTCOMBAT_MAIN = auto()
    ENDING = auto()


class Step(Enum):
    UNTAP = auto()
    UPKEEP = auto()
    DRAW = auto()
    PRECOMBAT_MAIN = auto()
    BEGIN_COMBAT = auto()
    DECLARE_ATTACKERS = auto()
    DECLARE_BLOCKERS = auto()
    COMBAT_DAMAGE = auto()
    END_COMBAT = auto()
    POSTCOMBAT_MAIN = auto()
    END_STEP = auto()
    CLEANUP = auto()


STEP_ORDER = [
    Step.UNTAP, Step.UPKEEP, Step.DRAW,
    Step.PRECOMBAT_MAIN,
    Step.BEGIN_COMBAT, Step.DECLARE_ATTACKERS, Step.DECLARE_BLOCKERS,
    Step.COMBAT_DAMAGE, Step.END_COMBAT,
    Step.POSTCOMBAT_MAIN,
    Step.END_STEP, Step.CLEANUP,
]

MAIN_PHASE_STEPS = {Step.PRECOMBAT_MAIN, Step.POSTCOMBAT_MAIN}

STEP_TO_PHASE = {
    Step.UNTAP: Phase.BEGINNING,
    Step.UPKEEP: Phase.BEGINNING,
    Step.DRAW: Phase.BEGINNING,
    Step.PRECOMBAT_MAIN: Phase.PRECOMBAT_MAIN,
    Step.BEGIN_COMBAT: Phase.COMBAT,
    Step.DECLARE_ATTACKERS: Phase.COMBAT,
    Step.DECLARE_BLOCKERS: Phase.COMBAT,
    Step.COMBAT_DAMAGE: Phase.COMBAT,
    Step.END_COMBAT: Phase.COMBAT,
    Step.POSTCOMBAT_MAIN: Phase.POSTCOMBAT_MAIN,
    Step.END_STEP: Phase.ENDING,
    Step.CLEANUP: Phase.ENDING,
}


class Zone(Enum):
    LIBRARY = auto()
    HAND = auto()
    BATTLEFIELD = auto()
    GRAVEYARD = auto()
    STACK = auto()


class CardType(Enum):
    LAND = auto()
    CREATURE = auto()
    INSTANT = auto()
    SORCERY = auto()
    ENCHANTMENT = auto()
    ARTIFACT = auto()


class ActionType(Enum):
    PASS_PRIORITY = auto()
    PLAY_LAND = auto()
    CAST_SPELL = auto()
    DECLARE_ATTACKERS = auto()
    DECLARE_BLOCKERS = auto()
    MULLIGAN_KEEP = auto()
    MULLIGAN_TAKE = auto()
    CHOOSE_BOTTOM_CARD = auto()
    CONCEDE = auto()


class AbilityKeyword(Enum):
    HASTE = auto()
    FIRST_STRIKE = auto()
    DOUBLE_STRIKE = auto()
    FLYING = auto()
    REACH = auto()
    TRAMPLE = auto()
    VIGILANCE = auto()
    DEATHTOUCH = auto()
    LIFELINK = auto()
    DEFENDER = auto()


class TargetType(Enum):
    PLAYER = auto()
    CREATURE = auto()


class GameResult(Enum):
    WIN_0 = 0
    WIN_1 = 1
    DRAW = 2
