import random
from vns_utils.helpers import tour_cost


def _double_bridge_once(tour):
    n = len(tour)
    # three distinct interior cut points -> four segments
    a, b, c = sorted(random.sample(range(1, n), 3))
    A, B, C, D = tour[:a], tour[a:b], tour[b:c], tour[c:]
    return A + C + B + D


def double_bridge(tour, dist):
    """N4: single double bridge."""
    new_tour = _double_bridge_once(tour)
    return new_tour, tour_cost(new_tour, dist)


def double_double_bridge(tour, dist):
    """
    N5: two successive double-bridge moves. Each move changes four
    edges, so applying two consecutively perturbs the tour more than a
    single double bridge, giving a strictly stronger shake than N4.
    """
    new_tour = _double_bridge_once(tour)
    new_tour = _double_bridge_once(new_tour)
    return new_tour, tour_cost(new_tour, dist)