import random
from vns_utils.helpers import tour_cost


def _double_bridge_once(tour):
    """
    One double-bridge (4-opt) move: cut at 3 interior points into
    four segments A, B, C, D and reconnect as A + C + B + D.
    This is the canonical non-sequential 4-opt move (Martin, Otto &
    Felten, 1991); it changes exactly four edges and cannot be undone
    by a single 2-opt or 3-opt move.
    """
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