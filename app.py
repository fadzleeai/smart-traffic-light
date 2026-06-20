"""
app.py
======
Flask server for the T-junction traffic sim. Serves the frontend HTML and
exposes the real Python Monte Carlo logic as a JSON API endpoint.

Architecture:
- Backend (this file + monte_carlo_core.py): the ONLY place Monte Carlo
  decisions are computed. Real Python, real randomness, real algorithm.
- Frontend (templates/index.html): pure rendering/animation in JS. It has
  NO Monte Carlo logic of its own — every decision is fetched from this
  server via POST /api/decide.

Run:
    python3 app.py
Then open:
    http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
import monte_carlo_core as mc

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/decide", methods=["POST"])
def decide():
    """
    Request body (JSON):
        {
            "queues": [int, int, int],       # required, current queue length per approach
            "cycle_budget_seconds": int,      # optional, default 60
            "arrival_rate_per_lane": [f,f,f], # optional, default [0.05]*3
            "trials_per_candidate": int       # optional, default 300
        }

    Response body (JSON):
        {
            "split": [int, int, int],   # chosen green-time seconds per approach
            "score": float               # avg cars left waiting under the chosen split
        }
    """
    data = request.get_json(silent=True)
    if not data or "queues" not in data:
        return jsonify({"error": "Request must include a 'queues' field, e.g. {\"queues\": [3, 4, 2]}"}), 400

    queues = data["queues"]
    if not isinstance(queues, list) or len(queues) == 0:
        return jsonify({"error": "'queues' must be a non-empty list of numbers"}), 400
    if not all(isinstance(q, (int, float)) and q >= 0 for q in queues):
        return jsonify({"error": "'queues' must contain only non-negative numbers"}), 400

    cycle_budget = data.get("cycle_budget_seconds", 60)
    arrival_rates = data.get("arrival_rate_per_lane")
    trials = data.get("trials_per_candidate", 300)

    if arrival_rates is not None and len(arrival_rates) != len(queues):
        return jsonify({"error": "'arrival_rate_per_lane' length must match 'queues' length"}), 400

    try:
        split, score = mc.monte_carlo_best_split(
            queue_lengths=queues,
            cycle_budget_seconds=cycle_budget,
            arrival_rate_per_lane=arrival_rates,
            trials_per_candidate=trials,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"split": split, "score": round(score, 2)})


@app.route("/api/fixed-split", methods=["POST"])
def fixed_split_endpoint():
    """
    Same request shape as /api/decide, but returns the (non-random) equal
    split instead. Included for completeness so Fixed Timer mode can also
    go through the backend if desired, though it has no real need to since
    there's no randomness involved.
    """
    data = request.get_json(silent=True)
    if not data or "queues" not in data:
        return jsonify({"error": "Request must include a 'queues' field"}), 400

    num_lanes = len(data["queues"])
    cycle_budget = data.get("cycle_budget_seconds", 60)
    split = mc.fixed_split(num_lanes, cycle_budget_seconds=cycle_budget)
    return jsonify({"split": split})


if __name__ == "__main__":
    app.run(debug=True, port=5000)