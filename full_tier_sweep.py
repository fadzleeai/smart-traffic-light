"""
Full 27-Combination Traffic Tier Sweep — Monte Carlo vs Fixed Timer
=====================================================================
Tests Monte Carlo against Fixed Timer across EVERY combination of
Low / Medium / High traffic on each of the 3 lanes independently.
3 tiers ^ 3 lanes = 27 distinct traffic scenarios, each run across
multiple random seeds for robustness, not just one lucky/unlucky draw.

WHERE THE TIER RATES COME FROM (not arbitrary):
Hang, Zhou & Wang (2020), "Modeling Traffic Function Reliability of
Signalized Intersections with Control Delay", Advances in Civil
Engineering, cites a saturation flow rate of 1,800 veh/h/lane as a
standard traffic engineering reference value, and tests volume-to-
capacity (v/c) ratios from 0.3 (matching their real measured off-peak
intersection) up to 0.9 (matching their real measured peak-hour
intersection). Multiplying the cited capacity by these v/c ratios
gives a defensible, sourced arrival rate for each tier:

    LOW    (v/c=0.3, off-peak)        -> 540 veh/h  -> 0.15 cars/sec/lane
    MEDIUM (v/c=0.6, moderate)        -> 1080 veh/h -> 0.30 cars/sec/lane
    HIGH   (v/c=0.9, peak congestion) -> 1620 veh/h -> 0.45 cars/sec/lane

Poisson sampling (used throughout this simulation, consistent with
the cited paper's own arrival-process assumption) turns each tier's
average rate into realistic randomly-varying per-cycle arrival counts.

No dependencies beyond the Python standard library.

Run:
    python3 full_tier_sweep.py
"""

import random
import itertools
import time

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

NUM_LANES = 3
CYCLE_BUDGET_SECONDS = 60
MIN_GREEN_SECONDS = 5
SECONDS_PER_CAR = 2.0
TRIALS_PER_CANDIDATE = 200
SEARCH_STEP_SECONDS = 5
NUM_CYCLES_PER_RUN = 30

# Sourced traffic tiers (see module docstring for derivation).
TIERS = {
    "LOW": 0.15,
    "MED": 0.30,
    "HIGH": 0.45,
}
TIER_NAMES = list(TIERS.keys())

# How many independent random seeds per combination, so a single
# lucky/unlucky run doesn't decide the result for that scenario.
SEEDS_PER_COMBO = [101, 202, 303, 404, 505]

STARTING_QUEUES = [4, 2, 3]


# ----------------------------------------------------------------------
# SHARED SIMULATION PRIMITIVES (identical algorithm to stats_comparison.py
# and robust_comparison.py — same Monte Carlo logic throughout this project)
# ----------------------------------------------------------------------

def poisson_sample(lam, rng):
    L = pow(2.718281828, -lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def generate_candidate_splits(num_lanes, budget, min_green, step):
    candidates = []
    remaining = budget - (min_green * num_lanes)
    if remaining < 0:
        raise ValueError("Budget too small for min_green on every lane")
    for extra_a in range(0, remaining + 1, step):
        for extra_b in range(0, remaining - extra_a + 1, step):
            extra_c = remaining - extra_a - extra_b
            if extra_c < 0:
                continue
            candidates.append([min_green + extra_a, min_green + extra_b, min_green + extra_c])
    return candidates


def pregenerate_trial_arrivals(num_trials, rng, arrival_rate_per_lane):
    """Common-random-numbers technique (see stats_comparison.py for full
    rationale): generate once, reuse across every candidate split this cycle,
    so only actual clearing performance differs between candidates, not
    independent random noise."""
    trials = []
    for _ in range(num_trials):
        trial_arrivals = [
            poisson_sample(arrival_rate_per_lane[lane] * CYCLE_BUDGET_SECONDS, rng)
            for lane in range(NUM_LANES)
        ]
        trials.append(trial_arrivals)
    return trials


def simulate_one_trial(queue_lengths, green_split, precomputed_arrivals):
    total_remaining = 0
    for lane in range(NUM_LANES):
        queue = queue_lengths[lane]
        max_cleared = green_split[lane] / SECONDS_PER_CAR
        cleared = min(queue, max_cleared)
        after_clearing = queue - cleared
        total_remaining += after_clearing + precomputed_arrivals[lane]
    return total_remaining


def score_candidate(queue_lengths, green_split, shared_trials):
    total = 0
    for trial_arrivals in shared_trials:
        total += simulate_one_trial(queue_lengths, green_split, trial_arrivals)
    return total / len(shared_trials)


def monte_carlo_best_split(queue_lengths, rng, arrival_rate_per_lane):
    candidates = generate_candidate_splits(
        NUM_LANES, CYCLE_BUDGET_SECONDS, MIN_GREEN_SECONDS, SEARCH_STEP_SECONDS
    )
    shared_trials = pregenerate_trial_arrivals(TRIALS_PER_CANDIDATE, rng, arrival_rate_per_lane)
    best_split, best_score = None, float("inf")
    for split in candidates:
        score = score_candidate(queue_lengths, split, shared_trials)
        if score < best_score:
            best_score, best_split = score, split
    return best_split


def fixed_split():
    each = (CYCLE_BUDGET_SECONDS // NUM_LANES // SEARCH_STEP_SECONDS) * SEARCH_STEP_SECONDS
    remainder = CYCLE_BUDGET_SECONDS - each * (NUM_LANES - 1)
    return [each] * (NUM_LANES - 1) + [remainder]


def apply_cycle(queue_lengths, green_split, rng, arrival_rate_per_lane):
    new_queues = []
    total_cleared = 0
    total_after = 0
    for lane in range(NUM_LANES):
        queue = queue_lengths[lane]
        max_cleared = green_split[lane] / SECONDS_PER_CAR
        cleared = min(queue, max_cleared)
        after_clearing = queue - cleared
        expected_arrivals = arrival_rate_per_lane[lane] * CYCLE_BUDGET_SECONDS
        new_arrivals = poisson_sample(expected_arrivals, rng)
        new_queue = after_clearing + new_arrivals
        new_queues.append(new_queue)
        total_cleared += cleared
        total_after += new_queue
    return new_queues, total_cleared, total_after


# ----------------------------------------------------------------------
# RUN ONE FULL (strategy, arrival_rate_combo, seed) COMBINATION
# ----------------------------------------------------------------------

def run_one_combination(strategy_is_montecarlo, arrival_rate_per_lane, seed):
    decision_rng = random.Random(seed * 7919)
    outcome_rng = random.Random(seed)
    queues = list(STARTING_QUEUES)

    total_left_waiting = 0
    total_cleared = 0
    max_left_waiting = 0

    for _ in range(NUM_CYCLES_PER_RUN):
        if strategy_is_montecarlo:
            split = monte_carlo_best_split(queues, decision_rng, arrival_rate_per_lane)
        else:
            split = fixed_split()

        new_queues, cleared, left_waiting = apply_cycle(queues, split, outcome_rng, arrival_rate_per_lane)
        total_left_waiting += left_waiting
        total_cleared += cleared
        max_left_waiting = max(max_left_waiting, left_waiting)
        queues = new_queues

    return {
        "avg_left_waiting": total_left_waiting / NUM_CYCLES_PER_RUN,
        "total_cleared": total_cleared,
        "max_left_waiting": max_left_waiting,
    }


# ----------------------------------------------------------------------
# THE FULL 27-COMBINATION SWEEP
# ----------------------------------------------------------------------

def run_full_sweep():
    """
    Generate every (lane0_tier, lane1_tier, lane2_tier) combination —
    3 tiers ^ 3 lanes = 27 total — and run both strategies across
    SEEDS_PER_COMBO random seeds for each one.
    """
    all_combos = list(itertools.product(TIER_NAMES, repeat=NUM_LANES))
    assert len(all_combos) == 27, f"Expected 27 combinations, got {len(all_combos)}"

    results = []

    for combo in all_combos:
        arrival_rate_per_lane = [TIERS[tier] for tier in combo]

        mc_waits = []
        fixed_waits = []
        for seed in SEEDS_PER_COMBO:
            mc_result = run_one_combination(True, arrival_rate_per_lane, seed)
            fixed_result = run_one_combination(False, arrival_rate_per_lane, seed)
            mc_waits.append(mc_result["avg_left_waiting"])
            fixed_waits.append(fixed_result["avg_left_waiting"])

        mc_avg = sum(mc_waits) / len(mc_waits)
        fixed_avg = sum(fixed_waits) / len(fixed_waits)
        improvement = ((fixed_avg - mc_avg) / fixed_avg * 100) if fixed_avg > 0 else 0

        results.append({
            "combo": combo,
            "mc_avg_wait": mc_avg,
            "fixed_avg_wait": fixed_avg,
            "improvement_pct": improvement,
        })

    return results


# ----------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------

def print_report(results):
    print("=" * 86)
    print("  FULL 27-COMBINATION TRAFFIC TIER SWEEP — Monte Carlo vs Fixed Timer")
    print("  Tiers sourced from Hang et al. (2020): LOW=0.15, MED=0.30, HIGH=0.45")
    print("  cars/sec/lane (1,800 veh/h/lane saturation flow rate x v/c ratio)")
    print("=" * 86)
    total_simulated_cycles = 27 * len(SEEDS_PER_COMBO) * 2 * NUM_CYCLES_PER_RUN
    print(f"  27 combinations x {len(SEEDS_PER_COMBO)} seeds x 2 strategies x "
          f"{NUM_CYCLES_PER_RUN} cycles = {total_simulated_cycles} total cycles simulated")
    print()

    header = f"{'#':>3} | {'Lane0':>5} {'Lane1':>5} {'Lane2':>5} | {'MC wait':>9} | {'Fixed wait':>11} | {'Improve':>8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        l0, l1, l2 = r["combo"]
        print(f"{i:>3} | {l0:>5} {l1:>5} {l2:>5} | {r['mc_avg_wait']:>9.2f} | "
              f"{r['fixed_avg_wait']:>11.2f} | {r['improvement_pct']:>7.1f}%")

    print("-" * len(header))

    improvements = [r["improvement_pct"] for r in results]
    overall_avg = sum(improvements) / len(improvements)
    overall_min = min(improvements)
    overall_max = max(improvements)
    num_mc_wins = sum(1 for x in improvements if x > 0)

    print(f"\n  SUMMARY ACROSS ALL 27 COMBINATIONS:")
    print(f"    Monte Carlo outperformed Fixed Timer in {num_mc_wins}/27 combinations")
    print(f"    Average improvement: {overall_avg:.1f}%")
    print(f"    Range: {overall_min:.1f}% to {overall_max:.1f}%")

    # Highlight the most/least favorable scenarios for context
    best = max(results, key=lambda r: r["improvement_pct"])
    worst = min(results, key=lambda r: r["improvement_pct"])
    print(f"\n    Best case for Monte Carlo: {best['combo']} -> {best['improvement_pct']:.1f}% improvement")
    print(f"    Worst case for Monte Carlo: {worst['combo']} -> {worst['improvement_pct']:.1f}% improvement")


def main():
    start = time.time()
    results = run_full_sweep()
    elapsed = time.time() - start
    print_report(results)
    print(f"\n  (Full sweep computed in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()