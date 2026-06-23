from run_tuning import run_experiments, SEEDS, TIME_LIMITS, MAX_ITERATIONS

TEST_INSTANCES = {
    "berlin52": 7542,
    "kroB100":  22141,
    "kroA200":  29368,
    "lin318":   41345,
}

# the configuration chosen on the tuning set
BEST_CONFIG = {"k_max": 5, "max_no_improve": 200, "init_method": "nearest_neighbor"}

if __name__ == "__main__":
    # fail early with a clear message if any test instance lacks limits
    missing = [name for name in TEST_INSTANCES
               if name not in TIME_LIMITS or name not in MAX_ITERATIONS]
    if missing:
        raise KeyError(
            f"Missing TIME_LIMITS/MAX_ITERATIONS for: {missing}. "
            f"Add them to run_tuning.py before running the test phase."
        )

    print(f"Running held-out test with config: {BEST_CONFIG}")
    print(f"Test instances: {list(TEST_INSTANCES.keys())}\n")

    run_experiments(
        instances=TEST_INSTANCES,
        configs=[BEST_CONFIG],          # exactly one config
        summary_path="results/test_summary.csv",
        results_subdir="results",
    )