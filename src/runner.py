from src.data_parser import load_instance
from src.vns import vns

if __name__ == "__main__":
    instance = load_instance("../data/eil51.tsp")
    best_tour, best_cost, logger = vns(instance)

    print(f"\nBest tour cost : {best_cost}")
    print(f"Best tour      : {best_tour}")