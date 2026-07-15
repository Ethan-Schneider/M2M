import numpy as np
from typing import Tuple
from ...agent import AgentLoader
from ...graph import Graph

def duration_difference(Rs : AgentLoader, G : Graph) -> Tuple[dict, dict]:
    agents_to_reallocate = []
    percent_difference = {}


    for agent in Rs.agents:
        if agent.task_sequence:
            if agent.status == 1:
                goal_loc = agent.task_sequence[0][1]
            elif agent.status == 2:
                goal_loc = agent.task_sequence[0][2]
            else:
                continue

            state = agent.state

            estimated_cost = G.get_distance(state, goal_loc)
            actual_cost = len(agent.path_sequence)

            agent_percent_difference = np.abs((actual_cost - estimated_cost) / estimated_cost) * 100

            if agent_percent_difference >= 10:
                agents_to_reallocate.append(agent)
                percent_difference[agent.id] = agent_percent_difference
                
    aisle_groups = {}
    
    if agents_to_reallocate:
        for agent_ in agents_to_reallocate:
            # get agent_id's current goal location column
            if Rs.get_agent(agent_.id).status == 1:
                goal_aisle = Rs.get_agent(agent_.id).task_sequence[0][1][1]
            elif Rs.get_agent(agent_.id).status == 2:
                goal_aisle = Rs.get_agent(agent_.id).task_sequence[0][2][1]
            else:
                print(f"Agent {agent_.id}")
            
            if goal_aisle not in aisle_groups.keys():
                aisle_groups[goal_aisle] = [agent_]
            
            for comparison_agent in Rs.agents:
                if comparison_agent.id != agent_.id and comparison_agent.task_sequence:
                    if comparison_agent.status == 1:
                        comp_aisle = comparison_agent.task_sequence[0][1][1]
                    elif comparison_agent.status == 2:
                        comp_aisle = comparison_agent.task_sequence[0][2][1]
                    else:
                        continue
                    if goal_aisle == comp_aisle and comparison_agent not in aisle_groups[goal_aisle]:
                        aisle_groups[goal_aisle].append(comparison_agent)

    return aisle_groups, percent_difference