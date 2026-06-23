"""
train_dqn.py
=============
Trains the DQN agent on the traffic signal environment, using randomized
arrival rates per episode (same fix that mattered for the tabular agent —
exposes the network to a wide range of situations rather than one fixed
pattern), with periodic evaluation against a fixed always-balanced
baseline so you can see whether learning is actually happening as
training progresses, not just at the very end.

Run:
    python3 train_dqn.py

Requires: pip install torch
"""

import random
import time

from traffic_env import TrafficEnv, ACTIONS
from dqn_agent import DQNAgent

TIERS = {"LOW": 0.15, "MED": 0.30, "HIGH": 0.45}
TIER_NAMES = list(TIERS.keys())

NUM_EPISODES = 8000
EPSILON_START = 1.0
EPSILON_END = 0.01          # Dropped from 0.15 to stop late-stage self-sabotage
EPSILON_DECAY_EPISODES = 2000
EVAL_EVERY = 200          # run an evaluation pass every N episodes
EVAL_SCENARIOS = [        # fixed scenarios used for every evaluation, for a consistent comparison over time
    [0.15, 0.30, 0.20],
    [0.45, 0.15, 0.15],
    [0.30, 0.30, 0.30],
    [0.15, 0.45, 0.30],
]


def epsilon_for_episode(episode_num):
    if episode_num >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    progress = episode_num / EPSILON_DECAY_EPISODES
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def run_episode_baseline(rates, seed, action_index=0):
    """Always-take-the-same-action baseline (default: action 0, balanced
    [20,20,20] split), used as the comparison point during evaluation."""
    env = TrafficEnv(arrival_rate_per_lane=rates, seed=seed, episode_length=30, use_raw_state=True)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state, reward, done = env.step(action_index)
        total_reward += reward
    return total_reward


def run_episode_agent(agent, rates, seed):
    """One episode with the agent acting GREEDILY (epsilon=0) — this is
    the actual deployed-policy performance, not exploration-noisy."""
    env = TrafficEnv(arrival_rate_per_lane=rates, seed=seed, episode_length=30, use_raw_state=True)
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.best_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
    return total_reward


def evaluate(agent):
    """Run the agent (greedy) and the baseline on the same fixed scenarios
    with the same seed, for a fair, repeatable comparison."""
    agent_total = 0
    baseline_total = 0
    for rates in EVAL_SCENARIOS:
        agent_total += run_episode_agent(agent, rates, seed=999)
        baseline_total += run_episode_baseline(rates, seed=999)
    return agent_total / len(EVAL_SCENARIOS), baseline_total / len(EVAL_SCENARIOS)


def main():
    rng = random.Random(7)
    agent = DQNAgent(seed=1)

    print(f"Training DQN for {NUM_EPISODES} episodes...")
    print(f"(Arrival rates randomized per episode across LOW/MED/HIGH tiers,")
    print(f" starting queues also randomized per episode — fixes a real gap found")
    print(f" via the 27-combination sweep: always starting at [4,2,3] meant the")
    print(f" agent rarely practiced handling near-empty or already-heavy queues)\n")

    start_time = time.time()
    recent_losses = []

    for episode_num in range(NUM_EPISODES):
        rates = [TIERS[rng.choice(TIER_NAMES)] for _ in range(3)]
        # Randomize starting queues each episode: 0-60 per lane covers
        # everything from empty to heavily congested, so the agent gets
        # real practice across the full range, not just whatever naturally
        # builds up starting from one fixed small queue.
        starting_queues = [rng.randint(0, 60) for _ in range(3)]
        env = TrafficEnv(arrival_rate_per_lane=rates, seed=None, episode_length=30, use_raw_state=True)
        state = env.reset(initial_queues=starting_queues)
        epsilon = epsilon_for_episode(episode_num)

        done = False
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                recent_losses.append(loss)
            state = next_state

        if (episode_num + 1) % EVAL_EVERY == 0:
            agent_avg, baseline_avg = evaluate(agent)
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else float("nan")
            recent_losses = []
            beats_baseline = agent_avg > baseline_avg
            print(f"  Episode {episode_num + 1}/{NUM_EPISODES} | epsilon={epsilon:.3f} | "
                  f"avg loss={avg_loss:.1f} | agent={agent_avg:.0f} vs baseline={baseline_avg:.0f} "
                  f"| beats baseline: {beats_baseline}")

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")

    print("\nFinal evaluation, scenario by scenario:")
    for rates in EVAL_SCENARIOS:
        agent_r = run_episode_agent(agent, rates, seed=999)
        baseline_r = run_episode_baseline(rates, seed=999)
        print(f"  Rates {rates}: agent={agent_r:.0f}  baseline={baseline_r:.0f}  "
              f"agent better: {agent_r > baseline_r}")

    # Save the trained model so it can be reloaded later without retraining
    import torch
    torch.save(agent.q_network.state_dict(), "dqn_trained.pt")
    print("\nSaved trained model to dqn_trained.pt")

    return agent


if __name__ == "__main__":
    main()