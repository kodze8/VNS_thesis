import ast
import os
from dataclasses import asdict

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from data_parser import load_instance
from vns import vns


def _clean(obj):
    #recursively convert numpy types to native json
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
STATIC_DIR = os.path.join(HERE, "static")



OPTIMALS = {
    "eil51": 426, "eil76": 538, "rat99": 1211, "eil101": 629,
    "ch150": 6528, "kroA100": 21282, "kroA150": 26524,
    "berlin52": 7542, "kroB100": 22141, "kroA200": 29368, "lin318": 41345,
}

# per-instance time / iteration tiers (matches the thesis setup)
TIME_LIMITS = {
    "eil51": 30, "berlin52": 30, "eil76": 45,
    "rat99": 60, "eil101": 60, "kroA100": 60, "kroB100": 60,
    "ch150": 120, "kroA150": 120,
    "kroA200": 180, "lin318": 300,
}
MAX_ITERATIONS = {
    "eil51": 500, "berlin52": 500, "eil76": 500,
    "rat99": 300, "eil101": 300, "kroA100": 300, "kroB100": 300,
    "ch150": 200, "kroA150": 200,
    "kroA200": 150, "lin318": 100,
}

ALLOWED = {
    "instances": list(OPTIMALS.keys()),
    "k_max": [2, 3, 5],
    "max_no_improve": [50, 100, 200],
    "init_method": ["nearest_neighbor", "random"],
    "seed": [42, 43, 44, 45, 46],
}

# instances flagged as slow so the UI can warn (n >= ~150)
SLOW_INSTANCES = ["ch150", "kroA150", "kroA200", "lin318"]

app = Flask(__name__, static_folder=None)

# small cache so repeated instance loads are cheap
_instance_cache = {}


def get_instance(name):
    if name not in _instance_cache:
        path = os.path.join(DATA_DIR, f"{name}.tsp")
        _instance_cache[name] = load_instance(path)
    return _instance_cache[name]


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)


@app.route("/api/options")
def options():
    """All allowed dropdown values + per-instance tier limits + slow flags."""
    return jsonify({
        "instances": ALLOWED["instances"],
        "k_max": ALLOWED["k_max"],
        "max_no_improve": ALLOWED["max_no_improve"],
        "init_method": ALLOWED["init_method"],
        "seed": ALLOWED["seed"],
        "time_limits": TIME_LIMITS,
        "max_iterations": MAX_ITERATIONS,
        "slow_instances": SLOW_INSTANCES,
        "optimals": OPTIMALS,
    })


@app.route("/api/instance/<name>")
def instance_coords(name):
    if name not in OPTIMALS:
        return jsonify({"error": "unknown instance"}), 400
    inst = get_instance(name)
    return jsonify({
        "name": name,
        "dimension": inst["dimension"],
        "coordinates": inst["coordinates"],
        "optimal": OPTIMALS[name],
    })


def _validate(payload):
    try:
        instance = str(payload["instance"])
        k_max = int(payload["k_max"])
        max_no_improve = int(payload["max_no_improve"])
        init_method = str(payload["init_method"])
        seed = int(payload["seed"])
    except (KeyError, ValueError, TypeError):
        return None, "missing or malformed parameters"

    if instance not in ALLOWED["instances"]:
        return None, f"instance not allowed: {instance}"
    if k_max not in ALLOWED["k_max"]:
        return None, f"k_max not allowed: {k_max}"
    if max_no_improve not in ALLOWED["max_no_improve"]:
        return None, f"max_no_improve not allowed: {max_no_improve}"
    if init_method not in ALLOWED["init_method"]:
        return None, f"init_method not allowed: {init_method}"
    if seed not in ALLOWED["seed"]:
        return None, f"seed not allowed: {seed}"

    return {
        "instance": instance,
        "k_max": k_max,
        "max_no_improve": max_no_improve,
        "init_method": init_method,
        "seed": seed,
    }, None


def _parse_tour(tour_str):
    try:
        return ast.literal_eval(tour_str)
    except (ValueError, SyntaxError):
        return []


@app.route("/api/run", methods=["POST"])
def run():
    payload = request.get_json(silent=True) or {}
    params, err = _validate(payload)
    if err:
        return jsonify({"error": err}), 400

    inst = get_instance(params["instance"])
    # time/iteration limits are tier-locked per instance (not user-chosen),
    # so the live run matches the thesis methodology.
    time_limit = TIME_LIMITS[params["instance"]]
    max_iter = MAX_ITERATIONS[params["instance"]]

    best_tour, best_cost, logger = vns(
        instance=inst,
        k_max=params["k_max"],
        max_iterations=max_iter,
        max_no_improve=params["max_no_improve"],
        time_limit=time_limit,
        seed=params["seed"],
        init_method=params["init_method"],
    )

    # serialise the per-step log parse tour strings back to lists
    steps = []
    for e in logger.entries:
        d = asdict(e)
        d["current_tour"] = _parse_tour(d["current_tour"])
        d["best_tour"] = _parse_tour(d["best_tour"])
        steps.append(d)

    # neighbourhood usage / contribution summary
    from logger import NEIGHBORHOOD_NAMES
    usage = {name: 0 for name in NEIGHBORHOOD_NAMES.values()}
    contrib = {name: 0 for name in NEIGHBORHOOD_NAMES.values()}
    for e in logger.entries:
        if e.neighborhood in usage:
            usage[e.neighborhood] += 1
        if e.improved and e.neighborhood in contrib:
            contrib[e.neighborhood] += 1

    optimal = OPTIMALS[params["instance"]]
    final_best = logger.entries[-1].best_cost if logger.entries else best_cost
    gap = round((final_best - optimal) / optimal * 100, 4) if optimal else None

    summary = {
        "instance": params["instance"],
        "optimal": optimal,
        "best_cost": final_best,
        "gap": gap,
        "total_iters": len(logger.entries),
        "improvements": sum(1 for e in logger.entries if e.improved),
        "time": logger.entries[-1].time_elapsed if logger.entries else 0,
        "usage": usage,
        "contribution": contrib,
        "params": params,
    }

    return jsonify(_clean({
        "coordinates": inst["coordinates"],
        "optimal": optimal,
        "steps": steps,
        "summary": summary,
    }))


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
