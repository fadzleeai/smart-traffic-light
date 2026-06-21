"""
traffic_env.py
================
A Q-learning training environment for the 3-lane traffic signal problem.
This module ONLY defines the environment (state, actions, reward, step
mechanics) — no learning happens here. The Q-learning agent itself lives
in a separate file and interacts with this environment through reset()
and step(), the standard RL interface pattern.

DESIGN DECISIONS, AND WHY:

State representation: queue lengths are discretized into 3 buckets per
lane (LOW/MED/HIGH), giving 3^3 = 27 possible states. Raw continuous queue
counts would make the state space infinite, impossible for tabular
Q-learning (which needs a finite table of state-action values). 27 states
is small enough to learn in a reasonable number of episodes, while still
capturing the thing that actually matters: which lanes are relatively
busier than others.

Action space: 6 discrete green-time splits (not Monte Carlo's full ~36-55
candidate search), covering "favor lane 0", "favor lane 1", "favor lane 2",
and a few balanced options. A smaller, fixed action set is standard
practice for tabular Q-learning, since the Q-table size is
(num_states x num_actions), and we want this small enough to actually
converge.

Reward: negative of "cars left waiting" after the cycle (so the agent
is rewarded for LOWER wait, i.e. maximizing reward = minimizing wait).
Same scoring philosophy as the rest of this project (stats_comparison.py,
full_tier_sweep.py), for consistency.

No dependencies beyond the Python standard library.
"""

import random

# ----------------------------------------------------------------------
# CONFIG — reused/consistent with the rest of the project
# ----------------------------------------------------------------------

NUM_LANES = 3
CYCLE_BUDGET_SECONDS = 60
SECONDS_PER_CAR = 2.0
MIN_GREEN_SECONDS = 5

# How finely the action space is sliced. Smaller step = more possible
# splits = finer control, more actions for the network to choose between.
# step=10 -> 15 actions, step=5 -> 55 actions, step=2 -> 276 actions.
ACTION_STEP_SECONDS = 5

# Queue bucket thresholds: a queue length below LOW_MAX is "LOW", up to
# MED_MAX is "MED", anything above is "HIGH". These define the 27-state
# space (3 buckets ^ 3 lanes) — only used by the bucketed/tabular state
# mode, DQN's raw state mode ignores these entirely.
LOW_MAX = 5
MED_MAX = 15


def generate_actions(budget=CYCLE_BUDGET_SECONDS, min_green=MIN_GREEN_SECONDS,
                      step=ACTION_STEP_SECONDS, num_lanes=NUM_LANES):
    """
    Algorithmically generate every legal [lane0, lane1, lane2] green-time
    split summing to `budget`, each lane getting at least `min_green`,
    stepping by `step`. Same generation logic Monte Carlo uses elsewhere
    in this project (see full_tier_sweep.py's generate_candidate_splits),
    so the action space size is a config choice, not a hand-picked list.
    """
    actions = []
    remaining = budget - (min_green * num_lanes)
    if remaining < 0:
        raise ValueError("Budget too small for min_green on every lane")
    for extra_a in range(0, remaining + 1, step):
        for extra_b in range(0, remaining - extra_a + 1, step):
            extra_c = remaining - extra_a - extra_b
            if extra_c < 0:
                continue
            actions.append([min_green + extra_a, min_green + extra_b, min_green + extra_c])
    return actions


ACTIONS = generate_actions()
NUM_ACTIONS = len(ACTIONS)


def poisson_sample(lam, rng):
    """Same Knuth's-algorithm Poisson sampler used throughout this project."""
    L = pow(2.718281828, -lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def bucket(queue_length):
    """Discretize a raw queue length into LOW / MED / HIGH."""
    if queue_length <= LOW_MAX:
        return "LOW"
    elif queue_length <= MED_MAX:
        return "MED"
    else:
        return "HIGH"


def state_from_queues(queues):
    """
    Convert raw queue lengths (e.g. [3, 17, 6]) into a discrete state
    the Q-table can index, e.g. ('LOW', 'HIGH', 'LOW').
    """
    return tuple(bucket(q) for q in queues)


def all_possible_states():
    """All 27 possible discrete states, for initializing/inspecting the Q-table."""
    buckets = ["LOW", "MED", "HIGH"]
    states = []
    for a in buckets:
        for b in buckets:
            for c in buckets:
                states.append((a, b, c))
    return states


class TrafficEnv:
    """
    A minimal RL environment for the 3-lane traffic signal problem.

    Usage (tabular / bucketed state):
        env = TrafficEnv(arrival_rate_per_lane=[0.15, 0.30, 0.20], seed=42)
        state = env.reset()  # -> ('LOW', 'MED', 'HIGH')

    Usage (DQN / raw state):
        env = TrafficEnv(arrival_rate_per_lane=[0.15, 0.30, 0.20], seed=42,
                          use_raw_state=True)
        state = env.reset()  # -> [4.0, 2.0, 3.0] (raw queue lengths)
    """

    def __init__(self, arrival_rate_per_lane, seed=None, episode_length=30, use_raw_state=False):
        self.arrival_rate_per_lane = arrival_rate_per_lane
        self.rng = random.Random(seed)
        self.episode_length = episode_length
        self.use_raw_state = use_raw_state
        self.queues = None
        self.cycle_count = 0

    def _current_state(self):
        if self.use_raw_state:
            return list(self.queues)  # raw, continuous — DQN reads this directly
        return state_from_queues(self.queues)  # bucketed — for tabular Q-learning

    def reset(self):
        """Start a new episode: fresh queues, fresh cycle counter."""
        self.queues = [4, 2, 3]  # same starting point used elsewhere in the project
        self.cycle_count = 0
        return self._current_state()

    def step(self, action_index):
        """
        Apply the chosen action (a green-time split), advance one cycle,
        and return (next_state, reward, done).
        """
        if not (0 <= action_index < NUM_ACTIONS):
            raise ValueError(f"action_index must be 0-{NUM_ACTIONS - 1}, got {action_index}")

        split = ACTIONS[action_index]

        new_queues = []
        for lane in range(NUM_LANES):
            queue = self.queues[lane]
            max_cleared = split[lane] / SECONDS_PER_CAR
            cleared = min(queue, max_cleared)
            after_clearing = queue - cleared

            expected_arrivals = self.arrival_rate_per_lane[lane] * CYCLE_BUDGET_SECONDS
            new_arrivals = poisson_sample(expected_arrivals, self.rng)

            new_queues.append(after_clearing + new_arrivals)

        total_left_waiting = sum(new_queues)
        reward = -total_left_waiting  # maximize reward = minimize cars left waiting

        self.queues = new_queues
        self.cycle_count += 1
        done = self.cycle_count >= self.episode_length

        return self._current_state(), reward, done