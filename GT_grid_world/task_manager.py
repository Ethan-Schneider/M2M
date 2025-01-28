from typing import Tuple, List

from src.graph import Graph
import numpy as np


# def generate_tasks(G : Graph, num_tasks : int, input_output_ratio : float, frequencey : float) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
#     return

# def save_tasks(tasks : List[Tuple[int, Tuple[int, int], Tuple[int, int]]], file_name : str) -> None:
#     pass

# def load_tasks(file_name : str) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
#     pass

def main():
    map = "data/maps/symbotic_small"
    num_robots = 20
    DOF = 4
    initial_inventory_amount = 25.
    
    np.random.seed(0)
    
    G = Graph(num_robots, map, DOF, True, "uniform", initial_inventory_amount)
    
    print(G.warehouse.printInventory())
    print(G.driveway.printInventory())
    

if __name__ == "__main__":
    main()
