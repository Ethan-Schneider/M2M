import numpy as np
from .path_finding_algorithms.GT_algorithms.cbs import cbs

from src.path_finding_algorithms.external_algorithms.EECBS import eecbs

def pathPlan(G, Rs : list, Ra : list, J : set, path_planning_strategy : str) -> list:
    if path_planning_strategy == "cbs":
        # solution = cbs()
        print(Rs)
        print(Ra)
        print(J)
        states = [robot[1] for robot in Rs]
        
        goal_locations = states.copy()
        for task in J:
            goal_locations[task[0]] = task[1]
            
        print("States; ", states)
        print("goal_locations: ", goal_locations)
        
        print(eecbs.test_cpp_func("GT_grid_world/src/maps/symbotic_small", len(Rs), 60, 1.2, states, goal_locations))