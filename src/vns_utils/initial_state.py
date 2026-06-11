import random

def nearest_neighbor(instance):
    # build an initial tour using nearest neighbor
    n = instance["dimension"]
    dist = instance["distances"]

    unvisited = list(range(1, n))  # all cities except start
    tour = [0]  # start from city 0

    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda x: dist[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)

    return tour


def random_tour(instance):
    n = instance["dimension"]
    tour = list(range(n))
    random.shuffle(tour)
    return tour

