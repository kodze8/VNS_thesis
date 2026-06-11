import random

from src.vns_utils.helpers import tour_cost


def small_shake(tour, dist, chain_length=1):
    """
    Or-opt shaking:
    randomly pick a chain of cities and insert it
    at a random different position in the tour.
    """
    n = len(tour)
    new_tour = tour[:]

    # pick a random starting position for the chain
    i = random.randint(0, n - chain_length - 1)
    chain = new_tour[i:i + chain_length]

    remaining = new_tour[:i] + new_tour[i + chain_length:]

    insert_at = random.randint(0, len(remaining) - 1)

    new_tour = remaining[:insert_at] + chain + remaining[insert_at:]

    return new_tour, tour_cost(new_tour, dist)
