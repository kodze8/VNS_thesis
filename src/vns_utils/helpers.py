def tour_cost(tour, dist):
    n = len(tour)
    return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))