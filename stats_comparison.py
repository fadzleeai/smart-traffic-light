"""
Monte Carlo vs Fixed-Timer — Multi-Cycle Stats Comparison
============================================================
Runs both control strategies over the same sequence of cycles (same
random arrivals applied to both, for a fair comparison) and reports
live per-cycle numbers plus a final summary table.

This is the script to run if you want HARD NUMBERS backing up a claim
like "Monte Carlo reduces wait time by X% vs fixed timers" rather than
just asserting it.

No dependencies beyond the Python standard library.

Run:
    python3 stats_comparison.py
"""

import random

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

NUM_LANES = 3
CYCLE_BUDGET_SECONDS = 60
MIN_GREEN_SECONDS = 5
SECONDS_PER_CAR = 2.0
ARRIVAL_RATE_PER_LANE = [0.12, 0.18, 0.08]   # avg new cars/sec, per lane
TRIALS_PER_CANDIDATE = 200                    # Monte Carlo trials per candidate split
SEARCH_STEP_SECONDS = 5

NUM_CYCLES_TO_RUN = 30          # how many signal cycles to simulate
RANDOM_SEED = 42                # fixed seed so both strategies see identical arrivals


# ----------------------------------------------------------------------
# SHARED SIMULATION PRIMITIVES
# ----------------------------------------------------------------------

def poisson_sample(lam, rng):
    """Knuth's algorithm, stdlib-only Poisson sampler."""
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


def pregenerate_trial_arrivals(num_trials, rng):
    """
    Pre-generate the random 'new arrivals' outcomes for a batch of trials,
    ONCE, to be reused identically across every candidate split being
    compared this cycle.

    WHY THIS MATTERS: arrivals happen over the full cycle regardless of
    which lane is green, which is realistic, cars don't stop arriving just
    because their light is red. But if each candidate split draws its OWN
    independent random arrivals, that random noise (same total magnitude
    for every candidate) swamps the actual signal we're trying to measure,
    which is how well each split clears the EXISTING queue. Two candidates
    that clear the queue very differently can end up with similar-looking
    scores purely because they got different random arrival luck, not
    because they're actually similar in quality.

    The fix is a standard variance-reduction technique called "common
    random numbers": generate the random arrivals once, then reuse that
    exact same set of outcomes for every candidate. Now the only thing
    that differs between candidates' scores is how well they clear the
    queue, which is the thing we actually want to optimize.
    """
    trials = []
    for _ in range(num_trials):
        trial_arrivals = [
            poisson_sample(ARRIVAL_RATE_PER_LANE[lane] * CYCLE_BUDGET_SECONDS, rng)
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


def monte_carlo_best_split(queue_lengths, rng):
    candidates = generate_candidate_splits(
        NUM_LANES, CYCLE_BUDGET_SECONDS, MIN_GREEN_SECONDS, SEARCH_STEP_SECONDS
    )
    # Generate the shared random outcomes ONCE for this decision, reused
    # across every candidate split being compared (see pregenerate_trial_arrivals).
    shared_trials = pregenerate_trial_arrivals(TRIALS_PER_CANDIDATE, rng)
    best_split, best_score = None, float("inf")
    for split in candidates:
        score = score_candidate(queue_lengths, split, shared_trials)
        if score < best_score:
            best_score, best_split = score, split
    return best_split


def fixed_split():
    """Equal split, rounded down to the nearest step, remainder to the last lane."""
    each = (CYCLE_BUDGET_SECONDS // NUM_LANES // SEARCH_STEP_SECONDS) * SEARCH_STEP_SECONDS
    remainder = CYCLE_BUDGET_SECONDS - each * (NUM_LANES - 1)
    return [each] * (NUM_LANES - 1) + [remainder]


def apply_cycle(queue_lengths, green_split, rng):
    """
    Actually 'run' one real cycle (not a trial): cars clear per the given
    split, then real new arrivals happen (the actual outcome, not a simulated
    guess at one). Returns the new queue lengths and stats for this cycle.
    """
    new_queues = []
    total_cleared = 0
    total_after = 0
    total_arrived = 0
    for lane in range(NUM_LANES):
        queue = queue_lengths[lane]
        max_cleared = green_split[lane] / SECONDS_PER_CAR
        cleared = min(queue, max_cleared)
        after_clearing = queue - cleared
        expected_arrivals = ARRIVAL_RATE_PER_LANE[lane] * CYCLE_BUDGET_SECONDS
        new_arrivals = poisson_sample(expected_arrivals, rng)
        new_queue = after_clearing + new_arrivals
        new_queues.append(new_queue)
        total_cleared += cleared
        total_after += new_queue
        total_arrived += new_arrivals
    return new_queues, total_cleared, total_after, total_arrived


# ----------------------------------------------------------------------
# RUN ONE FULL STRATEGY (Monte Carlo or Fixed) ACROSS MANY CYCLES
# ----------------------------------------------------------------------

def run_strategy(strategy_name, decide_split_fn, rng_seed):
    """
    Run NUM_CYCLES_TO_RUN cycles using `decide_split_fn(queues, decision_rng) -> split`
    to choose green times each cycle. Returns a list of per-cycle stat dicts.

    IMPORTANT: two separate random number generators are used:
    - `decision_rng` is consumed by the decision strategy itself (Monte Carlo
      burns many random draws running its internal trial simulations; Fixed
      timer burns none). How much randomness a strategy consumes internally
      must NOT affect the real-world outcome that follows.
    - `outcome_rng`, seeded identically for both strategies, is the ONLY
      source of randomness for the actual applied cycle (apply_cycle). This
      guarantees Monte Carlo and Fixed Timer face the literal same sequence
      of real car arrivals, which is what makes the comparison fair.
    """
    decision_rng = random.Random(rng_seed * 7919)  # distinct stream, arbitrary salt
    outcome_rng = random.Random(rng_seed)           # THE shared, comparable stream
    queues = [4, 2, 3]  # same starting queues every run, for fair comparison
    history = []

    print(f"\n{'='*86}")
    print(f"  RUNNING: {strategy_name}")
    print(f"{'='*86}")
    print(f"{'Cycle':>5} | {'Queues before':>16} | {'Green split':>16} | {'Cleared':>7} | {'Arrived':>7} | {'Left waiting':>12}")
    print("-" * 86)

    for cycle_num in range(1, NUM_CYCLES_TO_RUN + 1):
        split = decide_split_fn(queues, decision_rng)
        new_queues, cleared, left_waiting, arrived = apply_cycle(queues, split, outcome_rng)

        history.append({
            "cycle": cycle_num,
            "queues_before": list(queues),
            "split": split,
            "cleared": cleared,
            "arrived": arrived,
            "left_waiting": left_waiting,
        })

        queues_str = "/".join(str(q) for q in [round(q) for q in queues])
        split_str = "/".join(str(s) for s in split)
        print(f"{cycle_num:>5} | {queues_str:>16} | {split_str:>16} | {cleared:>7.0f} | {arrived:>7} | {left_waiting:>12.1f}")

        queues = new_queues

    return history


# ----------------------------------------------------------------------
# SUMMARY TABLE
# ----------------------------------------------------------------------

def summarize(history):
    total_cleared = sum(h["cleared"] for h in history)
    total_arrived = sum(h["arrived"] for h in history)
    avg_left_waiting = sum(h["left_waiting"] for h in history) / len(history)
    max_left_waiting = max(h["left_waiting"] for h in history)
    final_queue_total = sum(history[-1]["queues_before"]) if history else 0
    return {
        "total_cleared": total_cleared,
        "total_arrived": total_arrived,
        "avg_left_waiting": avg_left_waiting,
        "max_left_waiting": max_left_waiting,
        "final_backlog": history[-1]["left_waiting"],
    }


def print_comparison_table(mc_summary, fixed_summary):
    print(f"\n{'='*70}")
    print("  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<28} | {'Monte Carlo':>16} | {'Fixed Timer':>16}")
    print("-" * 70)

    rows = [
        ("Total cars cleared", "total_cleared", "{:.0f}"),
        ("Total cars arrived", "total_arrived", "{:.0f}"),
        ("Avg cars left waiting/cycle", "avg_left_waiting", "{:.2f}"),
        ("Worst cycle (cars waiting)", "max_left_waiting", "{:.1f}"),
        ("Final backlog", "final_backlog", "{:.1f}"),
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
    print(f"  (over {NUM_CYCLES_TO_RUN} cycles, identical random arrival sequence for both)")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print("MONTE CARLO vs FIXED TIMER — STATS COMPARISON")
    print(f"{NUM_LANES} lanes, {CYCLE_BUDGET_SECONDS}s cycle budget, {NUM_CYCLES_TO_RUN} cycles, seed={RANDOM_SEED}")

    mc_history = run_strategy(
        "MONTE CARLO",
        lambda queues, rng: monte_carlo_best_split(queues, rng),
        RANDOM_SEED,
    )
    fixed_history = run_strategy(
        "FIXED TIMER",
        lambda queues, rng: fixed_split(),
        RANDOM_SEED,  # same seed = same arrival sequence, fair comparison
    )

    mc_summary = summarize(mc_history)
    fixed_summary = summarize(fixed_history)
    print_comparison_table(mc_summary, fixed_summary)


if __name__ == "__main__":
    main()