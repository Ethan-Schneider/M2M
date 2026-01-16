import numpy as np
from ...agent import AgentLoader


def detect_backtracking(Rs : AgentLoader) -> dict:
    agents_to_reallocate = []
    for agent_ in Rs.agents:
        if agent_.path_sequence:
            if agent_.status == 1:
                current_target_location = agent_.task_sequence[0][1]
            elif agent_.status == 2:
                current_target_location = agent_.task_sequence[0][2]
            else:
                continue
            
            current_loc = agent_.state
            
            # Backtracking detection: Defined as increasing the L1 distance to the target location along their path sequence
            for loc in agent_.path_sequence[1:]:
                d1 = np.linalg.norm(np.array(current_target_location) - np.array(current_loc), ord=1)
                d2 = np.linalg.norm(np.array(current_target_location) - np.array(loc), ord=1)
                
                # print(f"Agent {agent_.id} at location {current_loc} with target location {current_target_location}")
                # print(f"Distance to target from current location: {d1}, distance to target from next location: {d2}")
                
                if d2 > d1:
                    agents_to_reallocate.append(agent_)
                    break
                current_loc = loc
            # print(f"Agent {agent_.id} at location {agent_.state} and status {agent_.status} with current target location {current_target_location}")
            # print(f"Agent {agent_.id} path sequence: {agent_.path_sequence}")
    
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

    return aisle_groups