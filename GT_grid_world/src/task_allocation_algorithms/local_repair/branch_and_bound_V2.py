import numpy as np
import time
from typing import Tuple, Dict, List

from ...agent import AgentLoader, Agent
from ...graph import Graph
from ...analysis.statistics import Stats

from sortedcontainers import SortedList

from ...path_finding_algorithms.external_algorithms.PBS import pbs

class BnB:
    def __init__(self, Rs : AgentLoader, G : Graph, J : set, S : Stats, 
                 reallocation_group : list, map_name : str, t_key : float, initial_cost : float) -> None:

        S.reallocation_data[t_key]["path_planning_compute_time"].append(0)
        S.reallocation_data[t_key]["lower_bound_and_checks"].append(0)
        self.start_time = time.time()
        # time limit of 1 second
        self.time_limit = 1.0

        self.Rs = Rs
        self.G = G
        self.J = J
        self.S = S

        self.map_name = map_name

        self.t_key = t_key

        self.agents = reallocation_group

        self.allocated_locations = set()
        self.goals = set()
        self.current_goals = {}

        tasks = []
        goals_per_task = []
        
        for agent in self.Rs.agents:
            if agent.task_sequence:
                for task in agent.task_sequence:
                    # print(f"task {task}")
                    self.allocated_locations.add(task[1])  # Start location
                    self.allocated_locations.add(task[2])  # End location

        for agent_id in self.agents:
            agent_status = self.Rs.get_agent(agent_id).status
            
            # print(f"Agent status: {agent_status}")
            
            if Rs.get_agent(agent_id).task_sequence:
                task_id = Rs.get_agent(agent_id).task_sequence[0][0]
            else:
                print(f"Agent has no task sequence ...")
                exit()
                
            if agent_status == 1:
                current_goal = Rs.get_agent(agent_id).task_sequence[0][1]
                task_loc_set = set(J[task_id][0]) | {current_goal}
                self.goals = self.goals.union(task_loc_set)
                goals_per_task.append(len(task_loc_set))
                self.allocated_locations.remove(current_goal)
            # If status is 2, get locations of current task's dropoff location
            elif agent_status == 2:
                current_goal = Rs.get_agent(agent_id).task_sequence[0][2]
                task_loc_set = set(J[task_id][1]) | {current_goal}
                self.goals = self.goals.union(task_loc_set)
                goals_per_task.append(len(task_loc_set))
                self.allocated_locations.remove(current_goal)
            else:
                print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")
                continue

            # Always keep the agent's currently assigned goal as a candidate,
            # even if J no longer lists it for this task/status.
            self.current_goals[agent_id] = current_goal

        num_combinations = np.prod(goals_per_task) + (np.sum(goals_per_task)**2 - np.sum(np.square(goals_per_task)))/2 + np.sum(goals_per_task)
        self.S.reallocation_data[self.t_key]["possible_number_nodes"] = int(num_combinations)
                        
        # print(f"Number of combinations: {num_combinations}")
        # print(f"Goals: {self.goals} and num goals {len(self.goals)}")
        # print(f"Already Allocated Locations: {self.allocated_locations}")

        self.goals = self.goals - self.allocated_locations
        
        # print(f"Goals: {self.goals} and num goals {len(self.goals)}")

        self.original_goals = list(self.goals)

        self.S.reallocation_data[self.t_key]["num_candidate_goal_locations"].append(len(self.original_goals))

        self.best_cost = np.inf
        self.best_assignment = None

        self.cost_matrix = []
        
        for agent_id in reallocation_group:
            task_id = Rs.get_agent(agent_id).task_sequence[0][0]
            agent_status = Rs.get_agent(agent_id).status
            
            if agent_status == 1:
                task_loc_set = J[task_id][0]
            elif agent_status == 2:
                task_loc_set = J[task_id][1]
            else:
                print(f"Agent {agent_id} with status {Rs.get_agent(agent_id).status}")

            if agent_id in self.current_goals:
                task_loc_set = set(task_loc_set) | {self.current_goals[agent_id]}
            
            row = []
            for loc in self.goals:
                if loc in self.allocated_locations:
                    row.append(np.inf)
                elif loc not in task_loc_set:
                    row.append(np.inf)
                else:
                    row.append(G.get_distance(Rs.get_agent(agent_id).state, loc))
            self.cost_matrix.append(row)

        self.nodes = SortedList()

        self.pruned_nodes = 0
        self.expanded_nodes = 0

    def solve(self) -> AgentLoader:
        self._branch({}, self.agents, self.goals, 0.0)

        self.S.reallocation_data[self.t_key]["nodes_expanded"] = self.expanded_nodes
        self.S.reallocation_data[self.t_key]["nodes_pruned"] = self.pruned_nodes

        if np.abs(time.time() - self.start_time) >= self.time_limit:
            return self.Rs

        self._modify_Rs(self.best_assignment)
        
        # print(f"Best Assignment: {self.best_assignment} with cost {self.best_cost}")
        return self.Rs
    
    def _modify_Rs(self, assignment : dict) -> None:
        if assignment is None:
            return
        
        for agent_id, goal_idx in assignment.items():
            # print(f"Agent {agent_id} with status {self.Rs.get_agent(agent_id).status} assigned to goal loc {self.original_goals[goal_idx]}: SKU needed: {self.J[self.Rs.get_agent(agent_id).task_sequence[0][0]][3]}: SKU at Location: {self.G.warehouse.get_sku_at_location(self.original_goals[goal_idx])}")
            agent = self.Rs.get_agent(agent_id)
            goal_loc = self.original_goals[goal_idx]
            # print(f"Agent task sequence: {agent.task_sequence[0]}")
            # exit()
            # print(f"Agent {agent_id} with goal_loc {goal_loc}")
            if agent.status == 1:
                agent.task_sequence[0] = (agent.task_sequence[0][0], goal_loc, agent.task_sequence[0][2], agent.task_sequence[0][3])
            elif agent.status == 2:
                agent.task_sequence[0] = (agent.task_sequence[0][0], agent.task_sequence[0][1], goal_loc, agent.task_sequence[0][3])                
            else:
                print(f"Agent {agent.id} should not have status {agent.status} Exiting ...")
                exit()

    def _lower_bound(self, remaining_agents : list, remaining_goals : set) -> float:
        lb = 0.0
        for agent_id in remaining_agents:
            best = np.inf
            for goal in remaining_goals:
                best = min(best, self.cost_matrix[self.agents.index(agent_id)][self.original_goals.index(goal)])
            lb += best
        return lb

    
    def mapf_cost(self, assignments: Dict[int, int]) -> float:
        """
        assignments: agent -> goal
        """
        goal_locations = []
        agent_states = []
        
        # agents in self.agents but not in assignment, choose their best goal location from self.cost_matrix and add that to agent_states and goal_locations
        # for agent_id in self.agents:
        #     if agent_id not in assignments:
        #         best_goal = np.argmin(self.cost_matrix[self.agents.index(agent_id)])
        #         goal_locations.append(self.original_goals[best_goal])
        #         agent_states.append(self.Rs.get_agent(agent_id).state)

        for assignment in assignments.items():
            goal_locations.append(self.original_goals[assignment[1]])
            agent_states.append(self.Rs.get_agent(assignment[0]).state)
            
        # # For all agents not included in self.agents, add their state and goal locations to sequences
        # for agent in self.Rs.agents:
        #     # if agent.id not in list(assignments.keys()):
        #     if agent.id not in self.agents:
        #         agent_states.append(agent.state)
        #         # If robot is going to pickup, set goal location to the task's start location
        #         if agent.status == 1:
        #             # Get current assigned task's start location
        #             goal_locations.append(agent.task_sequence[0][1])
                    
        #         # If robot is going to delivery, set goal location to the task's goal location
        #         elif agent.status == 2:
        #             goal_locations.append(agent.task_sequence[0][2])
                    
        #         # If robot is a free_agent, set goal location to current state
        #         else:
        #             goal_locations.append(agent.state)

        sequences = []
        w = 1.2
        
        latch = False
        while not sequences:
            if np.abs(time.time() - self.start_time) >= self.time_limit:
                return np.inf
            
            # Execute the path planning algorithm
            sequences = pbs.test_cpp_func(self.map_name, len(agent_states), 1, w, agent_states, goal_locations)
            if sequences == []:
                print("+++++++++++++++++++Execution Failed with w = ", w)
                
            # If a solution cannot be found with a higher suboptimality bound, break
            if w >= 1.2:
                if latch:
                    break
                latch = True
            w += 5.0

        if not sequences:
            return np.inf

        cost = 0
        for sequence in sequences:
            cost += len(sequence[1:])

        return cost

    def _branch(self, assignment : dict, remaining_agents : list, remaining_goals : set, current_cost : float) -> bool:
        """
        Expands current node to append them to the sorted list of all nodes
        
        :param self: Description
        :param assignment: Description
        :type assignment: dict
        :param current_cost: Description
        :type current_cost: float
        """
        lb_and_check_tik = time.time()
        if np.abs(time.time() - self.start_time) >= self.time_limit:
            return
        
        lb = current_cost + self._lower_bound(remaining_agents, remaining_goals)

        # If lower bound is higher than best cost, prune node
        if lb > self.best_cost:
            self.pruned_nodes += 1
            return
        
        # If fully allocated
        
        if not remaining_agents:
            if current_cost < self.best_cost:
                self.best_cost = current_cost
                self.best_assignment = assignment
            return
        
        self.S.reallocation_data[self.t_key]["lower_bound_and_checks"][-1] += np.abs(time.time() - lb_and_check_tik)
        
        # Choose next agent

        i = -1
        min_number = np.inf

        multiple = False
        agents = []

        for agent_id in remaining_agents:
            agent = self.Rs.get_agent(agent_id)
            num_goals = np.inf
            if agent.status == 1:
                num_goals = len(remaining_goals - self.J[agent.task_sequence[0][0]][0])
            elif agent.status == 2:
                num_goals = len(remaining_goals - self.J[agent.task_sequence[0][0]][1])
            if num_goals < min_number:
                i = agent_id
                min_number = num_goals
                agents = [agent_id]
            elif num_goals == min_number:
                multiple = True
                agents.append(agent_id)
            else:
                continue

        if multiple:
            i = np.random.choice(agents)

        child_nodes = SortedList()

        for g in remaining_goals:
            if np.abs(time.time() - self.start_time) >= self.time_limit:
                return
            
            j = self.original_goals.index(g)
            
            if self.cost_matrix[self.agents.index(i)][j] == np.inf:
                continue

            if self.cost_matrix[self.agents.index(i)] == np.inf:
                continue

            child_assignment = assignment.copy()
            child_assignment[i] = j

            child_remaining_agents = remaining_agents.copy()
            child_remaining_agents.remove(i)

            child_remaining_goals = remaining_goals.copy()
            child_remaining_goals.remove(g)

            mapf_tik = time.time()
            mapf_cost = self.mapf_cost(child_assignment)
            self.S.reallocation_data[self.t_key]["path_planning_compute_time"][-1] += np.abs(time.time() - mapf_tik)

            # print(f"MAPF Cost: {mapf_cost}")
            # print(f"Child Nodes: {child_nodes}")

            tie_breaker = 0

            for child in child_nodes:
                if child[0] == mapf_cost:
                    tie_breaker += 1
            
            child_nodes.add((mapf_cost, tie_breaker, child_assignment, child_remaining_agents, child_remaining_goals))

        # print(f"Child Nodes: {child_nodes}")

        for node in child_nodes:
            if np.abs(time.time() - self.start_time) >= self.time_limit:
                return
            
            # print(f"Node cost {node[0]}")
            self.expanded_nodes += 1
            self._branch(node[2], node[3], node[4], node[0])