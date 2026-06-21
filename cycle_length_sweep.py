"""
Cycle Length Sweep — Does a Shorter Decision Cycle Actually Help?
====================================================================
Tests whether re-deciding more often (shorter CYCLE_BUDGET_SECONDS) improves
Monte Carlo's real-world advantage over Fixed Timer, or whether the added
overhead from more frequent yellow-light transitions cancels it out.

THE FAIRNESS PROBLEM THIS SCRIPT SOLVES:
Naively running "30 cycles" at different cycle lengths is NOT a fair
comparison — 30 cycles at 60s covers 1800 simulated seconds, but 30 cycles
at 20s only covers 600 seconds. A shorter-cycle config would look better
purely because it ran for less total time, not because it's actually
better. This script instead fixes TOTAL_SIMULATED_SECONDS and computes how
many cycles of each length fit into that same real-world time window, so
every cycle-length config is compared over an identical span of time.

THE TRANSITION-TIME PROBLEM THIS SCRIPT SOLVES:
The original tier sweep (full_tier_sweep.py) had no concept of yellow-light
transition time — green time converted directly to cars cleared, with zero
cost for switching between lanes. That's a real blind spot: more cycles
means more lane switches, and each switch costs real time where no cars
clear. Without modeling this, shorter cycles would look artificially better
than they'd actually be. This script explicitly subtracts
YELLOW_TRANSITION_SECONDS x NUM_LANES from each cycle's usable green time,
so the real cost of cycling faster is actually represented.

Tier rates: see full_tier_sweep.py for full sourcing
(Hang et al. 2020 — LOW=0.15, MED=0.30, HIGH=0.45 cars/sec/lane).

No dependencies beyond the Python standard library.

Run:
    python3 cycle_length_sweep.py
"""

import random
import itertools
import time

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

NUM_LANES = 3
MIN_GREEN_SECONDS = 5
SECONDS_PER_CAR = 2.0
TRIALS_PER_CANDIDATE = 200
SEARCH_STEP_SECONDS = 5

# Real-world cost of switching the active lane (matches the 3s yellow phase
# already used in the project's T-junction visual sim). Charged once per
# lane-switch, NUM_LANES times per full cycle (every lane gets a turn).
YELLOW_TRANSITION_SECONDS = 3

# Every cycle-length config is compared over this SAME total simulated
# time window, not the same number of cycles (see fairness note above).
TOTAL_SIMULATED_SECONDS = 1800  # 30 minutes

# The cycle lengths being tested. NOTE: 20s was excluded — at 3 lanes, with
# MIN_GREEN_SECONDS=5/lane and YELLOW_TRANSITION_SECONDS=3/lane, the minimum
# required budget is 15s and transition overhead alone is 9s, leaving only
# 11s usable at a 20s cycle, below the 15s floor. 25s is the shortest
# cycle length that's actually mathematically feasible here.
CYCLE_LENGTHS_TO_TEST = [25, 30, 45, 60]

# A representative subset of tier combinations, not all 27 — this sweep
# already multiplies out across 4 cycle lengths, so a smaller, deliberately
# chosen set keeps runtime reasonable while still covering the cases that
# matter: fully symmetric (no asymmetry to exploit), mildly asymmetric,
# and strongly asymmetric.
TIERS = {"LOW": 0.15, "MED": 0.30, "HIGH": 0.45}
TIER_COMBOS_TO_TEST = [
    ("MED", "MED", "MED"),    # symmetric — Monte Carlo's hardest case
    ("LOW", "MED", "HIGH"),   # strongly asymmetric — Monte Carlo's best case
    ("LOW", "LOW", "MED"),    # mildly asymmetric, lighter traffic
    ("HIGH", "HIGH", "LOW"),  # mildly asymmetric, heavier traffic
]

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


def pregenerate_trial_arrivals(num_trials, rng, arrival_rate_per_lane, cycle_budget_seconds):
    """Common-random-numbers technique (see stats_comparison.py for full
    rationale): generate once, reuse across every candidate split this cycle,
    so only actual clearing performance differs between candidates, not
    independent random noise."""
    trials = []
    for _ in range(num_trials):
        trial_arrivals = [
            poisson_sample(arrival_rate_per_lane[lane] * cycle_budget_seconds, rng)
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


def usable_green_budget(cycle_budget_seconds):
    """
    The real allocatable green time after subtracting yellow-transition
    overhead. Every lane gets one transition into its green phase, so the
    cost is YELLOW_TRANSITION_SECONDS x NUM_LANES per full cycle. This is
    the mechanism that makes shorter cycles genuinely cost something: the
    same fixed transition overhead eats a larger FRACTION of a short cycle
    than a long one.
    """
    return cycle_budget_seconds - (YELLOW_TRANSITION_SECONDS * NUM_LANES)


def monte_carlo_best_split(queue_lengths, rng, arrival_rate_per_lane, cycle_budget_seconds):
    usable_budget = usable_green_budget(cycle_budget_seconds)
    candidates = generate_candidate_splits(
        NUM_LANES, usable_budget, MIN_GREEN_SECONDS, SEARCH_STEP_SECONDS
    )
    shared_trials = pregenerate_trial_arrivals(
        TRIALS_PER_CANDIDATE, rng, arrival_rate_per_lane, cycle_budget_seconds
    )
    best_split, best_score = None, float("inf")
    for split in candidates:
        score = score_candidate(queue_lengths, split, shared_trials)
        if score < best_score:
            best_score, best_split = score, split
    return best_split


def fixed_split(cycle_budget_seconds):
    usable_budget = usable_green_budget(cycle_budget_seconds)
    each = (usable_budget // NUM_LANES // SEARCH_STEP_SECONDS) * SEARCH_STEP_SECONDS
    remainder = usable_budget - each * (NUM_LANES - 1)
    return [each] * (NUM_LANES - 1) + [remainder]


def apply_cycle(queue_lengths, green_split, rng, arrival_rate_per_lane, cycle_budget_seconds):
    new_queues = []
    total_cleared = 0
    total_after = 0
    for lane in range(NUM_LANES):
        queue = queue_lengths[lane]
        max_cleared = green_split[lane] / SECONDS_PER_CAR
        cleared = min(queue, max_cleared)
        after_clearing = queue - cleared
        # Arrivals still happen over the FULL cycle (including yellow time),
        # since real cars don't stop arriving just because lights are
        # transitioning — only the CLEARING capacity above is reduced.
        expected_arrivals = arrival_rate_per_lane[lane] * cycle_budget_seconds
        new_arrivals = poisson_sample(expected_arrivals, rng)
        new_queue = after_clearing + new_arrivals
        new_queues.append(new_queue)
        total_cleared += cleared
        total_after += new_queue
    return new_queues, total_cleared, total_after


# ----------------------------------------------------------------------
# RUN ONE FULL (strategy, arrival_rate_combo, seed) COMBINATION
# ----------------------------------------------------------------------

def run_one_combination(strategy_is_montecarlo, arrival_rate_per_lane, seed, cycle_budget_seconds):
    decision_rng = random.Random(seed * 7919)
    outcome_rng = random.Random(seed)
    queues = list(STARTING_QUEUES)

    # Number of cycles is derived from the FIXED total time window, not a
    # fixed cycle count, so every cycle-length config covers the same real
    # span of simulated time (see fairness note in module docstring).
    num_cycles = round(TOTAL_SIMULATED_SECONDS / cycle_budget_seconds)

    total_left_waiting = 0
    total_cleared = 0
    max_left_waiting = 0

    for _ in range(num_cycles):
        if strategy_is_montecarlo:
            split = monte_carlo_best_split(queues, decision_rng, arrival_rate_per_lane, cycle_budget_seconds)
        else:
            split = fixed_split(cycle_budget_seconds)

        new_queues, cleared, left_waiting = apply_cycle(
            queues, split, outcome_rng, arrival_rate_per_lane, cycle_budget_seconds
        )
        total_left_waiting += left_waiting
        total_cleared += cleared
        max_left_waiting = max(max_left_waiting, left_waiting)
        queues = new_queues

    return {
        "avg_left_waiting": total_left_waiting / num_cycles,
        "total_cleared": total_cleared,
        "max_left_waiting": max_left_waiting,
        "num_cycles": num_cycles,
    }


# ----------------------------------------------------------------------
# THE FULL 27-COMBINATION SWEEP
# ----------------------------------------------------------------------

def run_full_sweep():
    """
    For each tested cycle length, run every representative tier combination
    across SEEDS_PER_COMBO random seeds for both strategies. Each cycle
    length gets the SAME total simulated time window (see fairness note),
    so results are directly comparable across cycle lengths.
    """
    results = []

    for cycle_length in CYCLE_LENGTHS_TO_TEST:
        for combo in TIER_COMBOS_TO_TEST:
            arrival_rate_per_lane = [TIERS[tier] for tier in combo]

            mc_waits = []
            fixed_waits = []
            for seed in SEEDS_PER_COMBO:
                mc_result = run_one_combination(True, arrival_rate_per_lane, seed, cycle_length)
                fixed_result = run_one_combination(False, arrival_rate_per_lane, seed, cycle_length)
                mc_waits.append(mc_result["avg_left_waiting"])
                fixed_waits.append(fixed_result["avg_left_waiting"])

            mc_avg = sum(mc_waits) / len(mc_waits)
            fixed_avg = sum(fixed_waits) / len(fixed_waits)
            improvement = ((fixed_avg - mc_avg) / fixed_avg * 100) if fixed_avg > 0 else 0

            results.append({
                "cycle_length": cycle_length,
                "combo": combo,
                "mc_avg_wait": mc_avg,
                "fixed_avg_wait": fixed_avg,
                "improvement_pct": improvement,
                "num_cycles": mc_result["num_cycles"],
            })

    return results


# ----------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------

def print_report(results):
    print("=" * 90)
    print("  CYCLE LENGTH SWEEP — Does deciding more often actually help?")
    print(f"  Every cycle length compared over the SAME {TOTAL_SIMULATED_SECONDS}s simulated")
    print(f"  time window, with a {YELLOW_TRANSITION_SECONDS}s x {NUM_LANES} lanes yellow-transition")
    print("  cost charged per cycle (more cycles = more transition overhead)")
    print("=" * 90)
    print()

    header = (f"{'Cycle':>6} | {'#cyc':>5} | {'Pattern':>16} | {'MC wait':>9} | "
              f"{'Fixed wait':>11} | {'Improve':>8}")
    print(header)
    print("-" * len(header))

    for cycle_length in CYCLE_LENGTHS_TO_TEST:
        rows = [r for r in results if r["cycle_length"] == cycle_length]
        for r in rows:
            pattern_str = "/".join(r["combo"])
            print(f"{cycle_length:>5}s | {r['num_cycles']:>5} | {pattern_str:>16} | "
                  f"{r['mc_avg_wait']:>9.2f} | {r['fixed_avg_wait']:>11.2f} | {r['improvement_pct']:>7.1f}%")
        print("-" * len(header))

    # Per-cycle-length average across all tested patterns, the key answer
    # to "does shorter cycle length help overall?"
    print(f"\n  AVERAGE IMPROVEMENT BY CYCLE LENGTH (across all {len(TIER_COMBOS_TO_TEST)} tested patterns):")
    for cycle_length in CYCLE_LENGTHS_TO_TEST:
        rows = [r for r in results if r["cycle_length"] == cycle_length]
        avg_improvement = sum(r["improvement_pct"] for r in rows) / len(rows)
        avg_num_cycles = rows[0]["num_cycles"]
        print(f"    {cycle_length:>3}s cycles ({avg_num_cycles} cycles in the window): "
              f"{avg_improvement:>6.1f}% avg improvement")

    # Same comparison, but per traffic pattern, to see if the answer to
    # "does shorter help" depends on how asymmetric the traffic is.
    print(f"\n  SAME COMPARISON, BROKEN DOWN BY TRAFFIC PATTERN:")
    for combo in TIER_COMBOS_TO_TEST:
        pattern_str = "/".join(combo)
        print(f"\n    Pattern {pattern_str}:")
        for cycle_length in CYCLE_LENGTHS_TO_TEST:
            r = next(r for r in results if r["cycle_length"] == cycle_length and r["combo"] == combo)
            print(f"      {cycle_length:>3}s -> {r['improvement_pct']:>6.1f}% improvement")


def main():
    start = time.time()
    results = run_full_sweep()
    elapsed = time.time() - start
    print_report(results)
    print(f"\n  (Full sweep computed in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()