from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs

from .agent import *
from .utils import *
from .analysis.statistics import *
from typing import Dict, Tuple

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
            
        # If robot is a free_agent, set goal location to current state
        else:
            goal_locations.append(agent.home)

    sequences = []
    w = 1.2

    # Guarantee unique goal locations before path planning. PBS asserts a
    # single permanent goal per cell (ConstraintTable::insert2CAT); two agents
    # sharing a goal cell abort the planner. The baseline allocators never
    # emit duplicate goals, but crM2M shuffle tasks can place a pickup/drop-off
    # cell that coincides with a concurrent real task's cell, so any residual
    # collision must be resolved here. Keep the first occurrence of each cell
    # and defer the rest: idle agents (status 0) have no real goal so they go
    # to their home; task-bearing agents wait in place at their current state
    # this tick (a one-tick deferral -- they re-plan toward the real goal next
    # tick, the task is not dropped) and fall back to home if the state is also
    # taken. This loop is a no-op when goals are already unique, so baseline
    # and LNS-PBS routing behaviour is unchanged.
    if len(goal_locations) != len(set(goal_locations)):
        used = set()
        for i, agent in enumerate(Rs.agents):
            goal = goal_locations[i]
            if goal not in used:
                used.add(goal)
                continue
            fallbacks = (
                [agent.home, agent.state]
                if agent.status == 0
                else [agent.state, agent.home]
            )
            new_goal = next((c for c in fallbacks if c not in used), goal)
            goal_locations[i] = new_goal
            used.add(new_goal)
        if len(goal_locations) != len(set(goal_locations)):
            print(f"[WARN] router could not fully de-duplicate goal locations: {goal_locations}")

    print(f"Goal locations: {goal_locations}")
    # print(f"Number of goal locations: {len(goal_locations)}")
    # print(f"Number of unique goal locations: {len(set(goal_locations))}")
    
    latch = False
    while not sequences:
        # Execute the path planning algorithm
        if path_planning_strategy == "pbs":
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