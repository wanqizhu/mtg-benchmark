from __future__ import annotations
import random
from dataclasses import dataclass, field
from .game import Game, make_deck
from .agents import Agent
from .enums import GameResult


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    ea = expected_score(ra, rb)
    eb = 1.0 - ea
    new_ra = ra + k * (score_a - ea)
    new_rb = rb + k * ((1.0 - score_a) - eb)
    return new_ra, new_rb


@dataclass
class MatchResult:
    agent_a: str
    agent_b: str
    result: GameResult
    elo_a_after: float
    elo_b_after: float


class Tournament:
    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.agent_decks: dict[str, list[str]] = {}
        self.ratings: dict[str, float] = {}
        self.history: list[MatchResult] = []
        self.win_matrix: dict[str, dict[str, int]] = {}
        self.game_matrix: dict[str, dict[str, int]] = {}

    def register(self, name: str, agent: Agent, deck: list[str], initial_elo: float = 1000.0):
        self.agents[name] = agent
        self.agent_decks[name] = deck
        self.ratings[name] = initial_elo
        self.win_matrix[name] = {}
        self.game_matrix[name] = {}

    def play_match(
        self,
        name_a: str,
        name_b: str,
        starting_life: int = 20,
    ) -> GameResult:
        agent_a_cls = type(self.agents[name_a])
        agent_b_cls = type(self.agents[name_b])
        # Create fresh instances for stateless agents, or reuse for NN agents
        from .training import NNAgent
        if isinstance(self.agents[name_a], NNAgent):
            a = NNAgent(self.agents[name_a].q_net, epsilon=0.0)
        else:
            a = agent_a_cls()
        if isinstance(self.agents[name_b], NNAgent):
            b = NNAgent(self.agents[name_b].q_net, epsilon=0.0)
        else:
            b = agent_b_cls()

        game = Game(
            decks=[list(self.agent_decks[name_a]), list(self.agent_decks[name_b])],
            agents=[a, b],
            starting_life=starting_life,
            verbose=False,
        )
        result = game.run_game()

        score_a = 1.0 if result == GameResult.WIN_0 else (0.0 if result == GameResult.WIN_1 else 0.5)
        ra, rb = update_elo(self.ratings[name_a], self.ratings[name_b], score_a)
        self.ratings[name_a] = ra
        self.ratings[name_b] = rb

        self.history.append(MatchResult(name_a, name_b, result, ra, rb))

        self.win_matrix[name_a].setdefault(name_b, 0)
        self.win_matrix[name_b].setdefault(name_a, 0)
        self.game_matrix[name_a].setdefault(name_b, 0)
        self.game_matrix[name_b].setdefault(name_a, 0)
        self.game_matrix[name_a][name_b] += 1
        self.game_matrix[name_b][name_a] += 1
        if result == GameResult.WIN_0:
            self.win_matrix[name_a][name_b] += 1
        elif result == GameResult.WIN_1:
            self.win_matrix[name_b][name_a] += 1

        return result

    def round_robin(self, games_per_pair: int = 100, starting_life: int = 20):
        names = list(self.agents.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for _ in range(games_per_pair):
                    if random.random() < 0.5:
                        self.play_match(names[i], names[j], starting_life)
                    else:
                        self.play_match(names[j], names[i], starting_life)

    def report(self) -> str:
        lines = ["\n=== Tournament Results ===\n"]
        sorted_agents = sorted(self.ratings.items(), key=lambda x: -x[1])
        lines.append(f"{'Agent':<25} {'ELO':>8}")
        lines.append("-" * 35)
        for name, elo in sorted_agents:
            lines.append(f"{name:<25} {elo:>8.1f}")

        lines.append("\n\n=== Win Rate Matrix ===\n")
        names = [n for n, _ in sorted_agents]
        header = f"{'':>20}" + "".join(f"{n[:10]:>12}" for n in names)
        lines.append(header)
        for na in names:
            row = f"{na[:20]:>20}"
            for nb in names:
                if na == nb:
                    row += f"{'---':>12}"
                else:
                    wins = self.win_matrix.get(na, {}).get(nb, 0)
                    games = self.game_matrix.get(na, {}).get(nb, 0)
                    if games > 0:
                        row += f"{100 * wins / games:>10.1f}%"
                    else:
                        row += f"{'N/A':>12}"
            lines.append(row)

        return "\n".join(lines)
