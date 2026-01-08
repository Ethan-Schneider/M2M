import math
from typing import Dict, List, Tuple

from ...agent import AgentLoader, Agent
from ...graph import Graph
from ...analysis.statistics import Stats

from ...path_finding_algorithms.external_algorithms.PBS import pbs

INF = float("inf")

class MAPF_QAP_BnB:
    def __init__(self, base_cost, reallocation_group : list, 
                 locations : list, J : set, Rs : AgentLoader, S : Stats, 
                 map : str):
        """
        base_cost[i][g]: base MAPF cost for agent-task i to goal g
        """
        self.J = J
        self.Rs = Rs
        self.S = S

        self.map_name = map

        self.locations = locations
        self.reallocation_group = reallocation_group

        self.base_cost = base_cost

        self.n_agents = len(base_cost)
        self.n_goals = len(base_cost[0])

        self.best_cost = INF
        self.best_assignment = None

    def solve(self):
        agents = list(range(self.n_agents))
        goals = list(range(self.n_goals))
        self._branch({}, agents, goals, 0.0)
        return self.best_assignment, self.best_cost

    def _lower_bound(self, assigned, remaining_agents, remaining_goals):
        lb = 0.0
        for i in remaining_agents:
            best = INF
            for g in remaining_goals:
                best = min(best, self.base_cost[i][g])
            lb += best
        return lb

    def _branch(self, assigned, remaining_agents, remaining_goals, current_cost):
        # Bounding
        lb = current_cost + self._lower_bound(
            assigned, remaining_agents, remaining_goals
        )
        if lb >= self.best_cost:
            return

        # If complete
        if not remaining_agents:
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_assignment = dict(assigned)
            return

        # Choose next agent (simple heuristic: most constrained)
        i = min(
            remaining_agents,
            key=lambda a: min(self.base_cost[a][g] for g in remaining_goals)
        )

        for g in remaining_goals:
            if self.base_cost[i][g] == INF:
                continue

            new_assigned = dict(assigned)
            new_assigned[i] = g

            # Compute MAPF cost incrementally
            new_cost = self.mapf_cost(new_assigned)

            if new_cost >= self.best_cost:
                continue

            self._branch(
                new_assigned,
                [a for a in remaining_agents if a != i],
                [h for h in remaining_goals if h != g],
                new_cost
            )

    def mapf_cost(self, assignments: Dict[int, int]) -> float:
        """
        assignments: agent -> goal
        """
        goal_locations = []
        agent_states = []

        print(f"Assignments: {assignments}")

        for assignment in assignments.items():
            goal_locations.append(self.locations[assignment[1]])
            agent_states.append(self.Rs.get_agent(self.reallocation_group[assignment[0]]).state)

        sequences = []
        w = 1.2

        print(f"Number of agent assignments: {len(assignments)}")
        
        latch = False
        while not sequences:
            # Execute the path planning algorithm
            sequences = pbs.test_cpp_func(self.map_name, len(assignments), 1, w, agent_states, goal_locations)
            if sequences == []:
                print("+++++++++++++++++++Execution Failed with w = ", w)
                
            # If a solution cannot be found with a higher suboptimality bound, break
            if w >= 1.2:
                if latch:
                    break
                latch = True
            w += 5.0

        cost = 0
        for sequence in sequences:
            cost += len(sequence)

        return cost

def run(Rs : AgentLoader, G : Graph, S : Stats, J : set, reallocation_group : list, map : str) -> AgentLoader:
    locations = set()

    print(f"Current Agent Tasks")
    for agent in Rs.agents:
        if agent.task_sequence:
            print(f"Agent {agent.id} with current task {agent.task_sequence[0]}")
    
    allocated_locations = set()
    for agent in Rs.agents:
        if agent.task_sequence:
            for task in agent.task_sequence:
                allocated_locations.add(task[1])  # Start location
                allocated_locations.add(task[2])  # End location
            
    print(f"Allocated Locations: {allocated_locations}")

    print(f"Reallocation group: {reallocation_group}")
    
    for agent_id in reallocation_group:
        agent_status = Rs.get_agent(agent_id).status
        
        print(f"Agent status: {agent_status}")
        
        if Rs.get_agent(agent_id).task_sequence:
            task_id = Rs.get_agent(agent_id).task_sequence[0][0]
        else:
            print(f"Agent has no task sequence ...")
            exit()
        
        print(f"Task id: {task_id}")
        print(f"Task locations: {J[task_id]}")
        print(f"Task Start Locations: {J[task_id][0]}")
        print(f"Task Goal Locations: {J[task_id][1]}")
        # If status is 1, get locations of current task's pickup location
        if agent_status == 1:
            locations = locations.union(J[task_id][0])
            allocated_locations.remove(Rs.get_agent(agent_id).task_sequence[0][1])
        # If status is 2, get locations of current task's dropoff location
        elif agent_status == 2:
            locations = locations.union(J[task_id][1])
            allocated_locations.remove(Rs.get_agent(agent_id).task_sequence[0][2])
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")
            
    locations = list(locations)

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
        
        row = []
        for loc in locations:
            if loc in allocated_locations:
                row.append(INF)
            elif loc not in task_loc_set:
                row.append(INF)
            else:
                row.append(G.get_distance(Rs.get_agent(agent_id).state, loc))
        cost_matrix.append(row)
    
    print(f"Cost Matrix: {cost_matrix}")

    BnB = MAPF_QAP_BnB(cost_matrix, reallocation_group, locations, J, Rs, S, map)
    best_solution, best_cost = BnB.solve()

    print(f"Best Solution: {best_solution} with cost {best_cost}")

    if best_solution is None:
        return Rs
    else:
        for agent_idx, goal_idx in best_solution.items():
            print(f"Agent {reallocation_group[agent_idx]} at location {Rs.get_agent(reallocation_group[agent_idx]).state} going with goal location {locations[goal_idx]}")
        
        agent_id = reallocation_group[agent_idx]
        if Rs.get_agent(agent_id).status == 1:
            Rs.get_agent(agent_id).task_sequence[0] = (Rs.get_agent(agent_id).task_sequence[0][0], locations[goal_idx], Rs.get_agent(agent_id).task_sequence[0][2], Rs.get_agent(agent_id).task_sequence[0][3])
        elif Rs.get_agent(agent_id).status == 2:
            Rs.get_agent(agent_id).task_sequence[0] = (Rs.get_agent(agent_id).task_sequence[0][0], Rs.get_agent(agent_id).task_sequence[0][1], locations[goal_idx], Rs.get_agent(agent_id).task_sequence[0][2])
        else:
            print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status} ...")

        print(f"Better Solution Found")
        # exit()

        return Rs