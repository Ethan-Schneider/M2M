from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs

from .agent import *
from .utils import *
from .analysis.statistics import *
from .simulate import STATUS_PICKING, STATUS_PLACING
from typing import Dict, Tuple

BLOCKED_REPLAN_THRESHOLD = 5


def needs_path_plan(Rs: AgentLoader, blocked_threshold: int = BLOCKED_REPLAN_THRESHOLD) -> bool:
    for agent in Rs.agents:
        if agent.path_sequence == []:
            return True
        if agent.blocked_ticks >= blocked_threshold:
            return True
    return False


def clear_all_paths(Rs: AgentLoader) -> None:
    for agent in Rs.agents:
        agent.path_sequence = []
        agent.blocked_ticks = 0


def pathPlan(map : str, Rs : AgentLoader, path_planning_strategy : str, S : Stats) -> AgentLoader:
    states = [agent.state for agent in Rs.agents]

    

    goal_locations = []
    for agent in Rs.agents:
        # If robot is going to pickup, set goal location to the task's start location
        if agent.status == 1:
            # Get current assigned task's start location
            goal_locations.append(agent.task_sequence[0][1])
            
        # If robot is going to delivery, set goal location to the task's goal location
        elif agent.status == 2:
            goal_locations.append(agent.task_sequence[0][2])

        # Pick/place: hold position while waiting at pickup or delivery
        elif agent.status in (STATUS_PICKING, STATUS_PLACING):
            print(f"Agent {agent.id} status: {agent.status} with state: {agent.state}")
            goal_locations.append(agent.state)
            
        # If robot is a free_agent, set goal location to current state
        else:
            goal_locations.append(agent.home)

    sequences = []
    w = 1.2

    # If two agents have the same goal location, set the goal location of the agnet with no task to its home location
    # Check if any goal location is found more than once
    if len(goal_locations) != len(set(goal_locations)):
        # raise ValueError(f"Agents have duplicate goal locations: {goal_locations} with agent states: {states}")
        for i, agent in enumerate(Rs.agents):
            if goal_locations.count(goal_locations[i]) > 1 and agent.status == 0:
                goal_locations[i] = agent.home

    print(f"Goal locations: {goal_locations}")
    # print(f"Number of goal locations: {len(goal_locations)}")
    # print(f"Number of unique goal locations: {len(set(goal_locations))}")
    
    latch = False
    while not sequences:
        # Execute the path planning algorithm
        if path_planning_strategy == "pbs":
            if len(goal_locations) != len(set(goal_locations)):
                print(f"Duplicate goal locations: {goal_locations}")
            else:
                print(f"No duplicate goal locations.")
            if len(Rs.get_agent_states()) != len(set(Rs.get_agent_states())):
                print(f"Duplicate agent states: {Rs.get_agent_states()}")
            else:
                print(f"No duplicate agent states.")
            sequences = pbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        elif path_planning_strategy == "ecbs":
            sequences = eecbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        if sequences == []:
            print("+++++++++++++++++++Execution Failed with w = ", w)
            S.update_num_path_plan_fail()
            
        # If a solution cannot be found with a higher suboptimality bound, break
        if w >= 1.2:
            if latch:
                break
            for i, agent in enumerate(Rs.agents):
                print(f"Agent {agent.id} home location: {agent.home} with assigned goal location: {goal_locations[i]}")
                if agent.path_sequence == []:
                    if agent.status in (STATUS_PICKING, STATUS_PLACING):
                        goal_locations[i] = agent.state
                    else:
                        goal_locations[i] = agent.home
            latch = True
        w += 5.0

    if not sequences:
        clear_all_paths(Rs)
        return Rs
    
    # Remove first item in sequences, as they are the robot's current location
    for i, agent in enumerate(Rs.agents):
        temp_sequence = sequences[i][1:]
        agent.path_sequence = temp_sequence
        agent.blocked_ticks = 0
    
    return Rs