from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .game import Game, make_deck, GameView, Action
from .agents import Agent, RandomAgent
from .enums import ActionType, GameResult
from .features import (
    state_to_features, action_to_index, get_legal_action_mask,
    index_to_action, STATE_DIM, ACTION_DIM,
)
from .network import QNetwork


class NNAgent(Agent):
    def __init__(self, q_net: QNetwork, epsilon: float = 0.0):
        self.q_net = q_net
        self.epsilon = epsilon
        self.last_state = None
        self.last_action_idx = None
        self.last_mask = None
        self.last_life = None
        self.last_opp_life = None
        self.player_idx = None
        self.transitions: list[tuple] = []

    def choose_action(self, view: GameView, legal_actions: list[Action]) -> Action:
        state = state_to_features(view)
        mask = get_legal_action_mask(legal_actions, view)
        action_idx = self.q_net.select_action(state, mask, self.epsilon)
        action = index_to_action(action_idx, legal_actions, view)

        if self.player_idx is None:
            self.player_idx = view.my_player

        # Reward shaping: small reward for life advantage changes
        shaped_reward = 0.0
        if self.last_state is not None:
            if self.last_life is not None:
                life_delta = (view.my_life - self.last_life) - (view.opp_life - self.last_opp_life)
                shaped_reward = life_delta * 0.02

            self.transitions.append((
                self.last_state, self.last_action_idx, shaped_reward,
                state, False, mask,
            ))

        self.last_state = state
        self.last_action_idx = action_idx
        self.last_mask = mask
        self.last_life = view.my_life
        self.last_opp_life = view.opp_life
        return action

    def game_over_callback(self, result: GameResult, player: int):
        if self.last_state is not None:
            reward = 1.0 if result.value == player else -1.0
            terminal_state = np.zeros(STATE_DIM, dtype=np.float32)
            terminal_mask = np.zeros(ACTION_DIM, dtype=np.bool_)
            self.transitions.append((
                self.last_state, self.last_action_idx, reward,
                terminal_state, True, terminal_mask,
            ))

    def reset(self):
        self.last_state = None
        self.last_action_idx = None
        self.last_mask = None
        self.last_life = None
        self.last_opp_life = None
        self.player_idx = None
        self.transitions = []


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(self, transition: tuple):
        self.buffer.append(transition)

    def push_many(self, transitions: list[tuple]):
        self.buffer.extend(transitions)

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNTrainer:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 256,
        target_update_freq: int = 500,
        buffer_capacity: int = 100_000,
        min_buffer: int = 2000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 30_000,
    ):
        self.q_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer = min_buffer
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.train_steps = 0
        self.episodes = 0
        self.losses: list[float] = []

    def get_epsilon(self) -> float:
        frac = min(1.0, self.episodes / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def run_episode(
        self,
        deck1: list[str],
        deck2: list[str],
        opponent: Agent | None = None,
        starting_life: int = 20,
    ) -> GameResult:
        epsilon = self.get_epsilon()
        agent0 = NNAgent(self.q_net, epsilon=epsilon)
        if opponent is None:
            agent1 = NNAgent(self.q_net, epsilon=epsilon)
        else:
            agent1 = opponent

        game = Game(
            decks=[list(deck1), list(deck2)],
            agents=[agent0, agent1],
            starting_life=starting_life,
            verbose=False,
        )
        result = game.run_game()

        self.replay_buffer.push_many(agent0.transitions)
        if isinstance(agent1, NNAgent):
            self.replay_buffer.push_many(agent1.transitions)

        if len(self.replay_buffer) >= self.min_buffer:
            loss = self.train_step()
            self.losses.append(loss)

        self.episodes += 1
        return result

    def train_step(self) -> float:
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
        dones = torch.tensor([t[4] for t in batch], dtype=torch.bool)
        next_masks = torch.tensor(np.array([t[5] for t in batch]), dtype=torch.bool)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q[~next_masks] = float("-inf")
            next_q_max = next_q.max(1)[0]
            next_q_max[dones] = 0.0
            targets = rewards + self.gamma * next_q_max

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes": self.episodes,
            "train_steps": self.train_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episodes = ckpt["episodes"]
        self.train_steps = ckpt["train_steps"]


def make_nn_agent(model_path: str, epsilon: float = 0.0) -> NNAgent:
    q_net = QNetwork()
    ckpt = torch.load(model_path, weights_only=True)
    q_net.load_state_dict(ckpt["q_net"])
    return NNAgent(q_net, epsilon=epsilon)
