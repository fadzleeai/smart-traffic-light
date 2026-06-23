"""
stats_comparison_hourly.py
=============================
The comparison script that actually uses hourly_rates.py — every previous
comparison script (stats_comparison.py, stats_comparison_3.py,
robust_comparison.py, full_tier_sweep.py) used a FLAT, constant arrival
rate held for the entire run. This one simulates a full 24-hour day:
each cycle maps to a specific hour, and the arrival rate for that cycle
comes directly from the capacity-checked hourly_rates.py table — rates
genuinely change cycle to cycle, following the real pagi/petang rush
pattern, not a fixed number.

One cycle = one hour of simulated time (24 cycles = 1 full day). This is
coarser than the 60-second CYCLE_BUDGET_SECONDS used elsewhere, but it's
the simplest, most direct way to map "one row of the hourly table" to
"one decision cycle" without inventing a finer scheme — see the note in
main() if you want multiple decision-cycles per hour instead.

Run:
    python3 stats_comparison_hourly.py
"""

import random
from hourly_rates import build_hourly_table, check_capacity

NUM_LANES = 3
CYCLE_BUDGET_SECONDS = 60
MIN_GREEN_SECONDS = 5
SECONDS_PER_CAR = 2.0
TRIALS_PER_CANDIDATE = 200
SEARCH_STEP_SECONDS = 5
RANDOM_SEED = 42

LANE_NAMES = ["East", "West", "North"]  # matches hourly_rates.py's table keys


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
    for extra_a in range(0, remaining + 1, step):
        for extra_b in range(0, remaining - extra_a + 1, step):
            extra_c = remaining - extra_a - extra_b
            if extra_c < 0:
                continue
            candidates.append([min_green + extra_a, min_green + extra_b, min_green + extra_c])
    return candidates


def pregenerate_trial_arrivals(num_trials, rng, arrival_rate_per_lane):
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
    """
    REALISTIC CAVEAT: Monte Carlo is given THIS HOUR's true rate to
    simulate against — it does not know the next hour's rate in advance
    (no oracle knowledge of the future curve). This matches a real
    camera-based system: it estimates current demand and decides, finding
    out the next hour's true demand only when it measures again.
    """
    candidates = generate_candidate_splits(NUM_LANES, CYCLE_BUDGET_SECONDS, MIN_GREEN_SECONDS, SEARCH_STEP_SECONDS)
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
    total_arrived = 0
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
        total_arrived += new_arrivals
    return new_queues, total_cleared, total_arrived


CYCLES_PER_HOUR = 60  # re-decide every simulated minute (60min / 60),
                       # a realistic per-minute decision rate. Each cycle still
                       # draws arrivals from that hour's rate (the hourly
                       # table has no sub-hour resolution), but the system
                       # gets to react every cycle, not just once an hour.


def run_strategy(strategy_name, decide_split_fn, rng_seed, hourly_table, cycles_per_hour, verbose=True):
    decision_rng = random.Random(rng_seed * 7919)
    outcome_rng = random.Random(rng_seed)
    queues = [4, 2, 3]
    history = []

    if verbose:
        print(f"\n{'='*100}")
        print(f"  RUNNING: {strategy_name} (full 24-hour day, {cycles_per_hour} cycles/hour)")
        print(f"{'='*100}")
        print(f"{'Hr.Cyc':>7} | {'Rates (E/W/N)':>16} | {'Queues before':>18} | {'Split':>16} | {'Cleared':>7} | {'Arrived':>7} | {'Left waiting':>12}")
        print("-" * 100)

    for row in hourly_table:
        hour = row["hour"]
        rates = [row["east"], row["west"], row["north"]]

        for sub_cycle in range(cycles_per_hour):
            split = decide_split_fn(queues, decision_rng, rates)
            new_queues, cleared, arrived = apply_cycle(queues, split, outcome_rng, rates)
            left_waiting = sum(new_queues)

            history.append({
                "hour": hour, "sub_cycle": sub_cycle, "rates": rates,
                "queues_before": list(queues), "split": split,
                "cleared": cleared, "arrived": arrived, "left_waiting": left_waiting,
            })

            if verbose and sub_cycle % max(1, cycles_per_hour // 2) == 0:
                rates_str = "/".join(f"{r:.2f}" for r in rates)
                queues_str = "/".join(str(round(q, 1)) for q in queues)
                split_str = "/".join(str(s) for s in split)
                label = f"{hour}.{sub_cycle}"
                print(f"{label:>7} | {rates_str:>16} | {queues_str:>18} | {split_str:>16} | "
                      f"{cleared:>7.0f} | {arrived:>7} | {left_waiting:>12.1f}")

            queues = new_queues

    return history


def summarize(history):
    total_cleared = sum(h["cleared"] for h in history)
    total_arrived = sum(h["arrived"] for h in history)
    avg_left_waiting = sum(h["left_waiting"] for h in history) / len(history)
    max_left_waiting = max(h["left_waiting"] for h in history)
    return {
        "total_cleared": total_cleared, "total_arrived": total_arrived,
        "avg_left_waiting": avg_left_waiting, "max_left_waiting": max_left_waiting,
        "final_backlog": history[-1]["left_waiting"],
    }


def print_comparison_table(mc_summary, fixed_summary):
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON — Full 24-Hour Day")
    print(f"{'='*70}")
    print(f"{'Metric':<28} | {'Monte Carlo':>16} | {'Fixed Timer':>16}")
    print("-" * 70)
    rows = [
        ("Total cars cleared", "total_cleared", "{:.0f}"),
        ("Total cars arrived", "total_arrived", "{:.0f}"),
        ("Avg cars left waiting/hr", "avg_left_waiting", "{:.2f}"),
        ("Worst hour (cars waiting)", "max_left_waiting", "{:.1f}"),
        ("Final backlog (end of day)", "final_backlog", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        mc_val = fmt.format(mc_summary[key])
        fx_val = fmt.format(fixed_summary[key])
        print(f"{label:<28} | {mc_val:>16} | {fx_val:>16}")
    print("-" * 70)
    improvement = (
        (fixed_summary["avg_left_waiting"] - mc_summary["avg_left_waiting"])
        / fixed_summary["avg_left_waiting"] * 100
        if fixed_summary["avg_left_waiting"] > 0 else 0
    )
    print(f"\n  Monte Carlo reduces avg cars left waiting by {improvement:.1f}% vs fixed timer")
    print(f"  (over a full simulated 24-hour day, identical random arrival sequence for both)")


def main():
    hourly_table = build_hourly_table()
    violations = check_capacity(hourly_table)
    if violations:
        print("ABORTING: hourly rate table failed its own capacity check.")
        print("Fix hourly_rates.py before running this comparison.")
        return

    print("MONTE CARLO vs FIXED TIMER — COMPOUNDING SWEEP")
    print("(Asymmetric East/West curve x multiple decision frequencies)")
    print(f"{NUM_LANES} lanes (East/West/North), East peak=0.155 West peak=0.190 (asymmetric),")
    print(f"seed={RANDOM_SEED}, capacity check passed\n")

    frequencies_to_test = [1, 6, 15, 30, 60]
    results = []

    for cph in frequencies_to_test:
        mc_history = run_strategy(
            "MONTE CARLO",
            lambda queues, rng, rates: monte_carlo_best_split(queues, rng, rates),
            RANDOM_SEED, hourly_table, cph, verbose=False,
        )
        fixed_history = run_strategy(
            "FIXED TIMER",
            lambda queues, rng, rates: fixed_split(),
            RANDOM_SEED, hourly_table, cph, verbose=False,
        )
        mc_summary = summarize(mc_history)
        fixed_summary = summarize(fixed_history)
        improvement = (
            (fixed_summary["avg_left_waiting"] - mc_summary["avg_left_waiting"])
            / fixed_summary["avg_left_waiting"] * 100
            if fixed_summary["avg_left_waiting"] > 0 else 0
        )
        results.append({
            "cph": cph, "mc_avg": mc_summary["avg_left_waiting"],
            "fixed_avg": fixed_summary["avg_left_waiting"], "improvement": improvement,
        })
        print(f"  cycles/hour={cph:>3} | MC avg wait={mc_summary['avg_left_waiting']:>7.2f} | "
              f"Fixed avg wait={fixed_summary['avg_left_waiting']:>7.2f} | improvement={improvement:>6.1f}%")

    print(f"\n{'='*70}")
    print("  SUMMARY: improvement vs decision frequency, WITH lane asymmetry")
    print(f"{'='*70}")
    for r in results:
        bar = "#" * int(r["improvement"])
        print(f"  {r['cph']:>3} cycles/hr: {r['improvement']:>5.1f}%  {bar}")


if __name__ == "__main__":
    main()