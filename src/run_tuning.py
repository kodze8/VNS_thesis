import csv
import os
import time
from data_parser import load_instance
from vns import vns


TUNING_INSTANCES = {
    "eil51": 426,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "ch150": 6528,
    "kroA100": 21282,
    "kroA150": 26524,
}

DATA_DIR = "../data"

TIME_LIMITS = {
    # small
    "eil51": 30,
    "berlin52": 30,
    "eil76": 45,
    # medium
    "rat99": 60,
    "eil101": 60,
    "kroA100": 60,
    "kroB100": 60,
    # large
    "ch150": 120,
    "kroA150": 120,
    # x-large
    "kroA200": 180,
    "lin318": 300,
    # xx-large
    "rat575": 600,
}

MAX_ITERATIONS = {
    # small
    "eil51": 500,
    "berlin52": 500,
    "eil76": 500,
    # medium
    "rat99": 300,
    "eil101": 300,
    "kroA100": 300,
    "kroB100": 300,
    # large
    "ch150": 200,
    "kroA150": 200,
    # x-large
    "kroA200": 150,
    "lin318": 100,
    # xx-large
    "rat575": 100,
}


SEEDS = [42, 43, 44, 45, 46]

CONFIGS = [
    {"k_max": 2, "max_no_improve": 100, "init_method": "nearest_neighbor"},
    {"k_max": 3, "max_no_improve": 100, "init_method": "nearest_neighbor"},
    {"k_max": 5, "max_no_improve": 100, "init_method": "nearest_neighbor"},
    {"k_max": 5, "max_no_improve": 100, "init_method": "random"},
    {"k_max": 5, "max_no_improve": 50,  "init_method": "nearest_neighbor"},
    {"k_max": 5, "max_no_improve": 200, "init_method": "nearest_neighbor"},
]


def run_experiments(instances, configs, summary_path, results_subdir):
    summary_rows = []
    total = len(instances) * len(configs) * len(SEEDS)
    completed = 0
    failed = 0

    print("---------------------------------------")
    print(f"  Instances : {len(instances)}")
    print(f"  Configs   : {len(configs)}")
    print(f"  Seeds     : {len(SEEDS)}")
    print(f"  Total runs: {total}")
    print("---------------------------------------\n")

    for instance_name, optimal in instances.items():
        filepath = os.path.join(DATA_DIR, f"{instance_name}.tsp")
        if not os.path.exists(filepath):
            print(f"PROBLEM_{instance_name} not found in data directory")
            continue

        instance = load_instance(filepath)
        time_limit = TIME_LIMITS[instance_name]
        max_iter = MAX_ITERATIONS[instance_name]

        print(f"\n--- {instance_name} | n={instance['dimension']} | optimal={optimal} | "
              f"time_limit={time_limit}s | max_iter={max_iter} ---")

        for config in configs:
            for seed in SEEDS:
                completed += 1
                print(f"  [{completed}/{total}] "
                      f"k_max={config['k_max']} | "
                      f"no_improve={config['max_no_improve']} | "
                      f"init={config['init_method']} | seed={seed} ... ",
                      end="", flush=True)
                try:
                    start = time.time()
                    best_tour, best_cost, logger = vns(
                        instance=instance,
                        k_max=config["k_max"],
                        max_iterations=max_iter,
                        max_no_improve=config["max_no_improve"],
                        time_limit=time_limit,
                        seed=seed,
                        init_method=config["init_method"],
                    )
                    elapsed = round(time.time() - start, 2)
                    if optimal:
                        gap = round((best_cost - optimal) / optimal * 100, 4)
                        gap = 0.0 if abs(gap) < 1e-6 else gap
                        print(f"cost={best_cost} | gap={gap}% | time={elapsed}s")
                    else:
                        print(f"cost={best_cost} | time={elapsed}s")
                    summary_rows.append(logger.to_summary_row())
                except Exception as e:
                    failed += 1
                    print(f"[ERROR] {e} : skipping this run")
                    continue

    save_summary(summary_rows, summary_path)
    print(f"\n  Completed : {len(summary_rows)}")
    print(f"  Failed    : {failed}")


def save_summary(rows, filepath):
    if not rows:
        print("No results to save.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary saved -> {filepath}")


if __name__ == "__main__":
    run_experiments(
        instances=TUNING_INSTANCES,
        configs=CONFIGS,
        summary_path="results/tuning_summary.csv",
        results_subdir="results",
    )