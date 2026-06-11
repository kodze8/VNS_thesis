import random
import time

from vns_utils.local_search import local_search
from vns_utils.or_opt import small_shake
from vns_utils.double_bridge import double_bridge, double_double_bridge
from vns_utils.initial_state import nearest_neighbor, random_tour
from logger import Logger

KNOWN_OPTIMALS = {
    "eil51": 426,
    "eil76": 538,
    "rat99": 1211,
    "eil101": 629,
    "ch150": 6528,
    "kroA100": 21282,
    "kroA150": 26524,
    "berlin52": 7542,
    "kroB100": 22141,
    "kroA200": 29368,
    "lin318": 41345,
}


def shake(tour, dist, k):
    if k == 1:
        return small_shake(tour, dist, chain_length=1)
    elif k == 2:
        return small_shake(tour, dist, chain_length=2)
    elif k == 3:
        return small_shake(tour, dist, chain_length=3)
    elif k == 4:
        return double_bridge(tour, dist)
    elif k == 5:
        return double_double_bridge(tour, dist)
    else:
        return small_shake(tour, dist, chain_length=1)


def vns(instance, k_max=5, max_iterations=500, max_no_improve=100, time_limit=60.0, seed=42,
        init_method="nearest_neighbor"):
    random.seed(seed)
    dist = instance["distances"]

    logger = Logger(
        instance_name=instance["name"],
        run_id=seed,
        optimal=KNOWN_OPTIMALS.get(instance["name"]),
        k_max=k_max,
        max_no_improve=max_no_improve,
        init_method=init_method,
    )

    # --- initialize ---
    if init_method == "nearest_neighbor":
        current_tour = nearest_neighbor(instance)
    elif init_method == "random":
        current_tour = random_tour(instance)
    else:
        raise ValueError(f"Unknown init_method: {init_method}. Use 'nearest_neighbor' or 'random'.")

    current_tour, current_cost = local_search(current_tour, dist)

    best_tour = current_tour[:]  # copy
    best_cost = current_cost

    iteration = 0
    no_improve_count = 0
    start_time = time.time()

    print(f"START VNS - Initial cost: {best_cost}")

    # --- main loop ---
    while (
            iteration < max_iterations
            and no_improve_count < max_no_improve
            and (time.time() - start_time) < time_limit
    ):
        k = 1
        while k <= k_max:

            # shaking
            shaken_tour, shaken_cost = shake(current_tour, dist, k)

            # local search on shaken solution
            improved_tour, improved_cost = local_search(shaken_tour, dist)

            # log
            logger.log(
                iteration=iteration,
                k=k,
                cost_before=current_cost,
                cost_after_shake=shaken_cost,
                cost_after=improved_cost,
                best_cost=best_cost,
                improved=improved_cost < current_cost,
                shaking_occurred=True,
                time_elapsed=round(time.time() - start_time, 4),
                current_tour=improved_tour,
                best_tour=best_tour,
            )

            if improved_cost < current_cost:
                # accept new solution and reset k
                current_tour = improved_tour[:]
                current_cost = improved_cost

                if improved_cost < best_cost:
                    best_tour = improved_tour[:]
                    best_cost = improved_cost
                    no_improve_count = 0
                    print(f"  Iteration {iteration:4d} | k={k} | New best: {best_cost:.2f}")

                k = 1  # reset to first neighborhood

            else:
                k += 1  # shake more if no imprpvement

        iteration += 1
        no_improve_count += 1

    elapsed = round(time.time() - start_time, 2)
    print(f"\nVNS finished - Best cost: {best_cost} - Iterations: {iteration} - Time: {elapsed}")

    logger.save()
    logger.summary()

    return best_tour, best_cost, logger
