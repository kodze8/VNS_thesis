import math
import numpy as np


def load_instance(filepath) -> dict:
    coordinates = []
    name = "unknown"
    dimension = 0

    with open(filepath, "r") as f:
        lines = f.readlines()

    in_coords = False
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue
        if line == "NODE_COORD_SECTION":
            in_coords = True
            continue

        if in_coords:
            parts = line.split()
            coordinates.append((float(parts[1]), float(parts[2])))
        elif line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())

    dist = build_matrix(coordinates)

    return {"name": name, "dimension": dimension, "coordinates": coordinates, "distances": dist}


def build_matrix(coordinates):
    n = len(coordinates)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = round(math.sqrt((coordinates[i][0]-coordinates[j][0])**2 + (coordinates[i][1]-coordinates[j][1])**2))
            #diagonally equal
            dist[i][j] = distance
            dist[j][i] = distance
    return dist


# if __name__ == "__main__":
#     instance = load_instance("../data/eil51.tsp")
#     print(instance)