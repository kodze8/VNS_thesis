"""
run_test.py
-----------
Phase 2: held-out evaluation.
Runs the SINGLE chosen configuration on the TEST instances only, once.
Edit BEST_CONFIG to the configuration selected from the tuning results
BEFORE running this. Do not run multiple configs here.
"""

from run_tuning import run_experiments, SEEDS

TEST_INSTANCES = {
    "berlin52": 7542,
    "kroB100":  22141,
    "kroA200":  29368,
    "lin318":   41345,
}

# add the test instances' limits to TIME_LIMITS / MAX_ITERATIONS in run_tuning.py,
# or override here by importing and updating those dicts before running.

# The configuration chosen on the tuning set. EDIT THIS after analysing tuning.
BEST_CONFIG = {"k_max": 5, "max_no_improve": 100, "init_method": "nearest_neighbor"}


if __name__ == "__main__":
    run_experiments(
        instances=TEST_INSTANCES,
        configs=[BEST_CONFIG],          # exactly one config
        summary_path="results/test_summary.csv",
        results_subdir="results",
    )