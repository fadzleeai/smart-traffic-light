"""
dqn_agent.py
=============
A DQN (Deep Q-Network) agent for the traffic signal environment.

WHY DQN OVER TABULAR Q-LEARNING:
The tabular agent (q_agent.py) required bucketing queue lengths into
LOW/MED/HIGH to keep the state space finite, which throws away real
information — a queue of 6 and a queue of 14 both become "MED" and are
treated identically, even though they may call for different decisions.
DQN replaces the lookup table with a neural network that takes the RAW
queue lengths directly as input, no discretization needed, since a neural
network can generalize over continuous inputs instead of needing a finite
table of them.

THE THREE STANDARD DQN COMPONENTS, AND WHY EACH EXISTS:

1. Q-NETWORK: a small MLP mapping state (3 raw queue lengths) to 6 Q-values
   (one per action in the same fixed action menu the tabular agent used).
   This replaces the Q-table — instead of looking up Q(s,a) in a table,
   you compute it by passing s through the network.

2. EXPERIENCE REPLAY BUFFER: stores past (state, action, reward, next_state,
   done) transitions, and trains on randomly-sampled BATCHES of them
   instead of training immediately on each new experience as it happens.
   This matters because consecutive experiences in one episode are highly
   correlated (today's queue depends on yesterday's), and training directly
   on correlated data makes neural network training unstable. Random
   sampling breaks that correlation.

3. TARGET NETWORK: a second, separate copy of the Q-network, updated only
   periodically (not every step) and used to compute the TARGET value in
   the training loss. Without this, the network would be trying to hit a
   target that's defined by itself and shifting on every single update,
   a well-documented source of instability in DQN. The target network
   holds still for a while, giving the main network something stable to
   learn against.

Requires PyTorch (pip install torch). Verified working in this sandbox
with torch 2.x — if your machine has a different version, the core API
used here (nn.Module, nn.Linear, optim.Adam, MSELoss) is stable across
recent torch versions and should work unchanged.
"""

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from traffic_env import TrafficEnv, ACTIONS, NUM_ACTIONS


# ----------------------------------------------------------------------
# Q-NETWORK
# ----------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Small MLP: 3 inputs (raw queue lengths) -> hidden layers -> 6 outputs
    (one Q-value per action). Kept deliberately small — this problem has
    only 3 input features and 6 actions, a large network would just be
    slower to train with no real benefit here.
    """

    def __init__(self, state_dim=3, num_actions=NUM_ACTIONS, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# EXPERIENCE REPLAY BUFFER
# ----------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=20000, seed=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------------------------
# DQN AGENT
# ----------------------------------------------------------------------

class DQNAgent:
    """
    NOTE ON NORMALIZATION: raw queue lengths can grow into the hundreds
    over an episode, and rewards (negative queue sums) scale similarly.
    Neural networks train poorly on unnormalized, large-magnitude inputs
    and targets — gradients get dominated by scale rather than by the
    actual learning signal. This did NOT matter for the tabular agent
    (a lookup table has no concept of "magnitude"), but it matters a lot
    here. STATE_SCALE and REWARD_SCALE below divide raw values down to a
    roughly small, comparable range before they reach the network.
    """

    STATE_SCALE = 50.0    # divide raw queue lengths by this before feeding the network
    REWARD_SCALE = 50.0   # divide raw rewards by this before computing loss

    def __init__(self, state_dim=3, num_actions=NUM_ACTIONS, hidden_size=128,
                 lr=1e-3, gamma=0.99, buffer_capacity=20000,           # Increased gamma to 0.99
                 batch_size=64, target_update_every=800, seed=None):   # Slowed target update to 800
        if seed is not None:
            torch.manual_seed(seed)
        self.rng = random.Random(seed)

        self.q_network = QNetwork(state_dim, num_actions, hidden_size)
        self.target_network = QNetwork(state_dim, num_actions, hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() 

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # HUBER LOSS MODIFICATION: Prevents exploding gradients on massive traffic jams
        self.loss_fn = nn.SmoothL1Loss() 

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.num_actions = num_actions

        self.replay_buffer = ReplayBuffer(buffer_capacity, seed=seed)
        self.train_step_count = 0

    def _normalize_state(self, state):
        return [s / self.STATE_SCALE for s in state]

    def choose_action(self, state, epsilon):
        """Epsilon-greedy action selection, same logic as the tabular agent,
        but querying the network instead of a table for the greedy choice."""
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.num_actions)
        with torch.no_grad():
            normalized = self._normalize_state(state)
            state_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def best_action(self, state):
        """Greedy action, no exploration — the deployed policy."""
        return self.choose_action(state, epsilon=0.0)

    def remember(self, state, action, reward, next_state, done):
        # Normalize state and scale reward BEFORE storing, so everything
        # pulled from the replay buffer during training is already in a
        # network-friendly range.
        normalized_state = self._normalize_state(state)
        normalized_next_state = self._normalize_state(next_state)
        scaled_reward = reward / self.REWARD_SCALE
        self.replay_buffer.push(normalized_state, action, scaled_reward, normalized_next_state, done)

    def train_step(self):
        """
        Sample a batch from replay, compute the DQN loss, and update the
        Q-network. Returns the loss value (for monitoring), or None if
        there isn't yet enough data in the buffer to sample a full batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Current Q-value estimates for the actions actually taken
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DDQN MODIFICATION: 
        # 1. Main network selects the best next action
        # 2. Target network evaluates the Q-value of that selected action
        with torch.no_grad():
            best_next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_network(next_states).gather(1, best_next_actions).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()