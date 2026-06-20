"""
monte_carlo_core.py
====================
The real Monte Carlo decision logic, adapted from monte_carlo_traffic.py
for use as an importable backend module rather than a standalone demo
script. The algorithm itself is UNCHANGED — same candidate generation,
same trial simulation, same averaging-over-trials approach. The only
difference is that config values are passed as parameters instead of
being read from module-level globals, so a Flask server can safely call
this from multiple requests with different settings.

This file has NO Flask dependency and no I/O — it's pure logic, so it
can be tested and verified completely standalone before any web server
code touches it.
"""

import random


def poisson_sample(lam):
    """
    Minimal Poisson sampler using only the standard library (Knuth's
    algorithm). Same implementation as monte_carlo_traffic.py.
    """
    L = pow(2.718281828, -lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def generate_candidate_splits(num_lanes, budget, min_green, step):
    """
    Generate every way to split `budget` seconds across `num_lanes` lanes,
    each lane getting at least `min_green`, stepping by `step`.
    Identical logic to monte_carlo_traffic.py.
    """
    candidates = []
    remaining_after_mins = budget - (min_green * num_lanes)
    if remaining_after_mins < 0:
        raise ValueError("Budget too small for min_green on every lane")

    for extra_a in range(0, remaining_after_mins + 1, step):
        for extra_b in range(0, remaining_after_mins - extra_a + 1, step):
            extra_c = remaining_after_mins - extra_a - extra_b
            if extra_c < 0:
                continue
            candidates.append([
                min_green + extra_a,
                min_green + extra_b,
                min_green + extra_c,
            ])
    return candidates


def simulate_one_trial(queue_lengths, green_split, arrival_rate_per_lane,
                        cycle_budget_seconds, seconds_per_car, num_lanes):
    """
    Simulate ONE random outcome for a given split: cars clear during green
    time, new cars arrive randomly over the full cycle. Identical logic to
    monte_carlo_traffic.py's simulate_one_trial, just parameterized instead
    of reading module-level globals.
    """
    total_remaining = 0
    for lane in range(num_lanes):
        queue = queue_lengths[lane]
        green_time = green_split[lane]
        max_cars_cleared = green_time / seconds_per_car
        cars_cleared = min(queue, max_cars_cleared)
        queue_after_clearing = queue - cars_cleared

        expected_arrivals = arrival_rate_per_lane[lane] * cycle_budget_seconds
        new_arrivals = poisson_sample(expected_arrivals)

        total_remaining += queue_after_clearing + new_arrivals
    return total_remaining


def score_candidate(queue_lengths, green_split, num_trials, arrival_rate_per_lane,
                     cycle_budget_seconds, seconds_per_car, num_lanes):
    """
    Average `num_trials` random simulations for this candidate split.
    Identical logic to monte_carlo_traffic.py's score_candidate.
    """
    total_score = 0
    for _ in range(num_trials):
        total_score += simulate_one_trial(
            queue_lengths, green_split, arrival_rate_per_lane,
            cycle_budget_seconds, seconds_per_car, num_lanes
        )
    return total_score / num_trials


def monte_carlo_best_split(
    queue_lengths,
    cycle_budget_seconds=60,
    min_green_seconds=5,
    seconds_per_car=2.0,
    arrival_rate_per_lane=None,
    trials_per_candidate=300,
    search_step_seconds=5,
):
    """
    Given current queue lengths (a list of length num_lanes), search over
    candidate green-time splits and return the one with the lowest average
    simulated wait. Same algorithm as monte_carlo_traffic.py's function of
    the same name, just with explicit parameters instead of module globals,
    so each request can safely use its own settings.

    Returns: (best_split: list[int], best_score: float)
    """
    num_lanes = len(queue_lengths)
    if arrival_rate_per_lane is None:
        arrival_rate_per_lane = [0.05] * num_lanes
    if len(arrival_rate_per_lane) != num_lanes:
        raise ValueError("arrival_rate_per_lane must match length of queue_lengths")

    candidates = generate_candidate_splits(
        num_lanes, cycle_budget_seconds, min_green_seconds, search_step_seconds
    )

    best_split = None
    best_score = float("inf")

    for split in candidates:
        avg_score = score_candidate(
            queue_lengths, split, trials_per_candidate, arrival_rate_per_lane,
            cycle_budget_seconds, seconds_per_car, num_lanes
        )
        if avg_score < best_score:
            best_score = avg_score
            best_split = split

    return best_split, best_score


def fixed_split(num_lanes, cycle_budget_seconds=60, search_step_seconds=5):
    """
    Equal split across all lanes, rounded down to the nearest step,
    remainder given to the last lane. Same approach as the JS version's
    fixedSplit(), included here for completeness even though the Flask
    server may not need to be called for fixed-timer mode at all (it has
    no randomness, so it could just as easily live client-side).
    """
    each = (cycle_budget_seconds // num_lanes // search_step_seconds) * search_step_seconds
    remainder = cycle_budget_seconds - each * (num_lanes - 1)
    return [each] * (num_lanes - 1) + [remainder]