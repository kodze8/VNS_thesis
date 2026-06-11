from src.vns_utils.helpers import tour_cost

def local_search(tour, dist):
    """
    2-opt local search.
    tries all possible edge swaps and keeps improving until no better swap exists.
    """
    n = len(tour)
    best = tour[:]
    best_cost = tour_cost(best, dist)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[(j + 1) % n]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                if delta < 0:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    best_cost += delta
                    improved = True
    return best, best_cost