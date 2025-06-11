from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs

from .agent import *
from .utils import *
from .analysis.statistics import *

def pathPlan(map : str, Rs : AgentLoader, J : set, path_planning_strategy : str, S : Stats) -> AgentLoader:
    states = [agent.state for agent in Rs.agents]

    goal_locations = []
    for agent in Rs.agents:
        # If robot is going to pickup, set goal location to the task's start location
        if agent.status == 1:
            # Get current assigned task's start location
            goal_locations.append(get_task_start_location(J, agent.task_sequence[0]))
            
        # If robot is going to delivery, set goal location to the task's goal location
        elif agent.status == 2:
            goal_locations.append(get_task_goal_location(J, agent.task_sequence[0]))
            
        # If robot is a free_agent, set goal location to current state
        else:
            goal_locations.append(agent.state)

    sequences = []
    w = 1.2
    
    print(f"Start Locations: {states}")
    print(f"Goal Locations: {goal_locations}")
    
    latch = False
    while not sequences:
        # Execute the path planning algorithm
        if path_planning_strategy == "pbs":
            sequences = pbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        elif path_planning_strategy == "ecbs":
            sequences = eecbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        if sequences == []:
            print("+++++++++++++++++++Execution Failed with w = ", w)
            print(sequences)
            S.update_num_path_plan_fail()
            
        # If a solution cannot be found with a higher suboptimality bound, break
        if w >= 11.2:
            if latch:
                break
            for i, agent in enumerate(Rs.agents):
                if agent.path_sequence == []:
                    goal_locations[i] = agent.home
            latch = True
        w += 5.0

    if not sequences:
        return Rs
    
    # Remove first item in sequences, as they are the robot's current location
    for i, agent in enumerate(Rs.agents):
        temp_sequence = sequences[i][1:]
        agent.path_sequence = temp_sequence
    
    return Rs