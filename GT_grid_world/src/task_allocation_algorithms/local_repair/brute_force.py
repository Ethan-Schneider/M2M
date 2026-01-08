import numpy as np
from typing import Tuple, List, Dict

from itertools import product

from sortedcontainers import SortedList

from ...agent import AgentLoader, Agent
from ...graph import Graph
from ...analysis.statistics import Stats

from ...path_finding_algorithms.external_algorithms.PBS import pbs

def brute_force(Rs : AgentLoader, G : Graph, S : Stats, J : set, reallocation_group : list, map_name : str) -> AgentLoader:
            
    print(f"Reallocation Group: {reallocation_group}")
    col_locs = []
    print(f"Current Agent Goal Locations")
    for agent in Rs.agents:
        if agent.task_sequence and agent.id in reallocation_group:
            if agent.status == 1:
                col_locs.append(agent.task_sequence[0][1][1])
                print(f"Agent {agent.id} with current goal {agent.task_sequence[0][1]}")
            elif agent.status == 2:
                col_locs.append(agent.task_sequence[0][2][1])
                agent.task_sequence[0][2][1]
                print(f"Agent {agent.id} with current goal {agent.task_sequence[0][2]}")         
            else:
                print(f"Error ...")
    
    if len(set(col_locs)) > 1:
        print(f"col locs: {col_locs}")
        exit()
        
    print(f"Agent Task Sequences: {[agent.task_sequence[0] for agent in Rs.agents if agent.id in reallocation_group]}")
    print(f"Col Locs: {col_locs}")
    col_loc = col_locs[0]
    print(f"Column # {col_loc}")
    
    allocated_locations = set()
    for agent in Rs.agents:
        if agent.task_sequence:
            for task in agent.task_sequence:
                allocated_locations.add(task[1])  # Start location
                allocated_locations.add(task[2])  # End location
            
    # print(f"Allocated Locations: {allocated_locations}")

    # print(f"Reallocation group: {reallocation_group}")
    
    agent_locs = {}
    
    for agent_id in reallocation_group:
        agent_status = Rs.get_agent(agent_id).status
        
        print(f"Agent status: {agent_status}")
        
        if Rs.get_agent(agent_id).task_sequence:
            task_id = Rs.get_agent(agent_id).task_sequence[0][0]
        else:
            print(f"Agent has no task sequence ...")
            exit()
        
        # print(f"Task id: {task_id}")
        # print(f"Task locations: {J[task_id]}")
        # print(f"Task Start Locations: {J[task_id][0]}")
        # print(f"Task Goal Locations: {J[task_id][1]}")
        
        # If status is 1, get locations of current task's pickup location
        if agent_status == 1:
            agent_locs[agent_id] = J[task_id][0]
            allocated_locations.remove(Rs.get_agent(agent_id).task_sequence[0][1])
        # If status is 2, get locations of current task's dropoff location
        elif agent_status == 2:
            agent_locs[agent_id] = J[task_id][1]
            allocated_locations.remove(Rs.get_agent(agent_id).task_sequence[0][2])
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")
            
    # print(f"allocated locations {allocated_locations}")
    
    # print(f"Agent Locations: {agent_locs}")
    
    for agent_id, locs in agent_locs.items():
        agent_locs[agent_id] = locs - allocated_locations
        
        for loc in locs:
            # if loc[1] == col_loc - 2 or loc[1] == col_loc or loc[1] == col_loc + 2:
            if loc[1] == col_loc:
                continue
            else:
                agent_locs[agent_id] -= set(loc)
    print(f"Fixed agent locations: {agent_locs}")
    
    
    allocations = [list(c) for c in product(*(agent_locs[k] for k in sorted(agent_locs)))]
    sorted_reallocation_list = reallocation_group.copy()
    sorted_reallocation_list.sort()

    
    # print(f"possible allocations: {allocations}")
    # print(f"sorted agent list: {sorted_reallocation_list}")
    
    # exit()
    
    mapf_cost_allocations = SortedList()
    estimated_cost_allocations = SortedList()
    
    for allocation in allocations:
        # print(f"Allocation: {allocation}")
        
        if len(set(allocation)) < len(allocation):
            continue
        
        mapf_cost_allocations.add((mapf_cost(Rs, allocation, map_name, sorted_reallocation_list), allocation))
        # estimated_cost_allocations.add((estimated_cost(Rs, G, allocation, sorted_reallocation_list), allocation))
        
    # print(f"Estimated costs: {estimated_cost_allocations} \n Mapf costs: {mapf_cost_allocations}")
    # print(f"Best assignment using estimated cost: {estimated_cost_allocations[0]} \n Best assignment using mapf costs: {mapf_cost_allocations[0]}")
    
    # for agent_id in sorted_reallocation_list:
    #     print(f"Agent {agent_id} with state {Rs.get_agent(agent_id).state}")
        
    for i, loc in enumerate(mapf_cost_allocations[0][1]):
        agent_id = sorted_reallocation_list[i]
        
        if Rs.get_agent(agent_id).status == 1:
            Rs.get_agent(agent_id).task_sequence[0] = (Rs.get_agent(agent_id).task_sequence[0][0], loc, Rs.get_agent(agent_id).task_sequence[0][2], Rs.get_agent(agent_id).task_sequence[0][3])
        elif agent.status == 2:
            Rs.get_agent(agent_id).task_sequence[0] = (Rs.get_agent(agent_id).task_sequence[0][0], Rs.get_agent(agent_id).task_sequence[0][1], loc, Rs.get_agent(agent_id).task_sequence[0][3])
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")
    # exit()
    
    print(f"New Agent Goal Locations")
    for agent in Rs.agents:
        if agent.task_sequence and agent.id in reallocation_group:
            if agent.status == 1:
                print(f"Agent {agent.id} with current goal {agent.task_sequence[0][1]}")
            elif agent.status == 2:
                print(f"Agent {agent.id} with current goal {agent.task_sequence[0][2]}")         
            else:
                print(f"Error ...")
    
    return Rs
    
    # Compute all possible assignment tuples


def estimated_cost(Rs : AgentLoader, G : Graph, assignment : List[Tuple[int, int]], sorted_reallocation_list : list) -> float:
    cost = 0
    
    for i, loc in enumerate(assignment): 
        cost += G.get_distance(Rs.get_agent(sorted_reallocation_list[i]).state, loc)
        
    return cost


def mapf_cost(Rs : AgentLoader, assignments : List[Tuple[int, int]], map_name : str, sorted_reallocation_list : list) -> float:
    """
    assignments: [(agent_id, loc_idx), ...]
    locations: [(col, row), ...]
    """
    goal_locations = []
    agent_states = []

    # print(f"Assignments: {assignments}")
    
    for i, loc in enumerate(assignments):
        goal_locations.append(loc)
        agent_states.append(Rs.get_agent(sorted_reallocation_list[i]).state)

    sequences = []
    w = 1.2
    
    latch = False
    while not sequences:
        # Execute the path planning algorithm
        sequences = pbs.test_cpp_func(map_name, len(assignments), 1, w, agent_states, goal_locations)
            
        # If a solution cannot be found with a higher suboptimality bound, break
        if w >= 1.2:
            if latch:
                break
            latch = True
        w += 5.0

    cost = 0
    if sequences:
        for sequence in sequences:
            cost += len(sequence)
    else:
        cost = 9999

    return cost