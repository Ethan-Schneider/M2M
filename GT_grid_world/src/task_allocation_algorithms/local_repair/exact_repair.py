import numpy as np
from munkres import Munkres, print_matrix, DISALLOWED

from typing import Tuple, List, Dict

from ...agent import AgentLoader, Agent
from ...graph import Graph
from ...analysis.statistics import Stats

def HA_exact_repair(Rs : AgentLoader, G : Graph, S : Stats, J : set, reallocation_group : list, t_key : float) -> AgentLoader:
    
    # print(f"Initial Agent Task Locations:")
    # for agent in Rs.agents:
    #     print(f"Agent {agent.id} with status {agent.status} with location {agent.state} and task sequence {agent.task_sequence}")
    
    allocated_locations = set()
    for agent in Rs.agents:
        if agent.task_sequence:
            for task in agent.task_sequence:
                # print(f"task {task}")
                allocated_locations.add(task[1])  # Start location
                allocated_locations.add(task[2])  # End location
                # exit()
            
    # print(f"Allocated Locations: {allocated_locations}")
            
    locations = set()
    current_goals = {}
    
    # print(f"Reallocation group: {reallocation_group}")
    
    for agent_id in reallocation_group:
        agent_status = Rs.get_agent(agent_id).status
        
        # print(f"Agent status: {agent_status}")
        
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
            current_goal = Rs.get_agent(agent_id).task_sequence[0][1]
            locations = locations.union(J[task_id][0])
            allocated_locations.remove(current_goal)
        # If status is 2, get locations of current task's dropoff location
        elif agent_status == 2:
            current_goal = Rs.get_agent(agent_id).task_sequence[0][2]
            locations = locations.union(J[task_id][1])
            allocated_locations.remove(current_goal)
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")

        # Always keep the agent's currently assigned goal as a candidate column,
        # even if J no longer lists it for this task/status.
        locations.add(current_goal)

        current_goals[agent_id] = current_goal
    locations = list(locations)

    S.reallocation_data[t_key]["num_candidate_goal_locations"].append(len(locations))

    cost_matrix = []
    
    for agent_id in reallocation_group:
        task_id = Rs.get_agent(agent_id).task_sequence[0][0]
        agent_status = Rs.get_agent(agent_id).status
        
        if agent_status == 1:
            task_loc_set = J[task_id][0]
        elif agent_status == 2:
            task_loc_set = J[task_id][1]
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")

        task_loc_set = set(task_loc_set) | {current_goals[agent_id]}
        
        row = []
        for loc in locations:
            if loc in allocated_locations:
                row.append(DISALLOWED)
            elif loc not in task_loc_set:
                row.append(DISALLOWED)
            else:
                row.append(G.get_distance(Rs.get_agent(agent_id).state, loc))
        cost_matrix.append(row)
    
    # print(f"Cost Matrix: {cost_matrix}")

    # Check if any row is just DISALLOWED
    if any(all(cell == DISALLOWED for cell in row) for row in cost_matrix):
        print(f"Found a row with all DISALLOWED values")
        # Handle this case as needed
        return Rs

    m = Munkres()
    indexes = m.compute(cost_matrix)
    
    # print(f"Indexes: {indexes}")
    for index in indexes:
        # print(f"Agent {reallocation_group[index[0]]} allocated location {locations[index[1]]}")
        agent_id = reallocation_group[index[0]]
        agent = Rs.get_agent(agent_id)
        loc = locations[index[1]]
        
        if agent.status == 1:
            agent.task_sequence[0] = (agent.task_sequence[0][0], loc, agent.task_sequence[0][2], agent.task_sequence[0][3])
        elif agent.status == 2:
            agent.task_sequence[0] = (agent.task_sequence[0][0], agent.task_sequence[0][1], loc, agent.task_sequence[0][3])
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")
            
    # print(f"Final Agent Task Locations:")
    # for agent in Rs.agents:
    #     print(f"Agent {agent.id} with location {agent.state} and task sequence {agent.task_sequence}")
    
    return Rs
        