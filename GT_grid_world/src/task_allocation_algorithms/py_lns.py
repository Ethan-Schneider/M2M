import numpy as np
import time
from typing import Set, Tuple, List
from ..utils import manhattan_distance
from ..graph import Graph
from ..agent import AgentLoader
from ..analysis.statistics import Stats
from .initial_solutions.construct_cost_tensor import construct_cost_tensor
from .removal_operators.random_removal import random_removal
from .repair_operators.greedy_repair import greedy_repair

from .initial_solutions.greedy_allocation import greedy_call
from .initial_solutions.randomized_greedy import randomized_greedy_call
from .initial_solutions.FCF import FCF_call
from .initial_solutions.max_regret_FC import max_regret_FC_call
from .initial_solutions.randomized_max_regret_FC import randomized_max_regret_FC_call

class LNS:
    def __init__(self, S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
                 initial_task_assignment_strategy : str, time_limit: float = 1.0,
                 removal_size: int = 3, cost_calculation_method: str = "manhattan"):
        """
        Initialize Large Neighborhood Search algorithm.
        
        Args:
            S: Statistics object for tracking metrics
            G: Graph representing the warehouse
            Rs: AgentLoader containing all agents
            J: Set of tasks to be assigned
            initial_solution: Initial solution from another algorithm
            time_limit: Maximum time to run LNS in seconds
            removal_size: Number of allocations to remove in each iteration
            cost_calculation_method: Method to use for cost calculation ("manhattan" or "shortest_path")
        """
        self.S = S
        self.G = G
        self.Rs = Rs
        self.J = J
        self.initial_task_assignment_strategy = initial_task_assignment_strategy
        self.time_limit = time_limit
        self.removal_size = removal_size
        self.cost_calculation_method = cost_calculation_method
        
        # Store best solution found
        self.best_solution = None
        self.best_cost = float('inf')
        
        # Construct initial cost tensor
        self.cost_tensor, self.start_locs, self.goal_locs, self.idx_to_task_id = construct_cost_tensor(
            J, Rs, G, cost_calculation_method
        )
        
    def run(self) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
        """
        Run the LNS algorithm.
        
        Returns:
            Tuple containing:
            - Updated AgentLoader with the best solution found
            - List of allocations (agent_idx, task_idx, start_idx, goal_idx)
            - Best cost found
        """
        inital_task_assignment_time = 0.0
        removal_time = 0.0
        update_cost_tensor_time = 0.0
        repair_time = 0.0
        start_time = time.time()

        tik = time.time()
        if self.initial_task_assignment_strategy == "greedy":
            current_solution, allocations, _ = greedy_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "randomized_greedy":
            current_solution, allocations, _ = randomized_greedy_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "FCF":
            current_solution, allocations, _ = FCF_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "max_regret_FC":
            current_solution, allocations, _ = max_regret_FC_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "randomized_max_regret_FC":
            current_solution, allocations, _ = randomized_max_regret_FC_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        else:
            print("ERROR: Unknown initial task assignment strategy " + self.initial_task_assignment_strategy + ", please choose another one.")
            return self.Rs, [], float('inf')
        tok = time.time()
        inital_task_assignment_time += tok - tik
        current_cost = self._calculate_total_cost(current_solution)


        
        # Initialize best solution and allocations
        self.best_solution = current_solution
        self.best_cost = current_cost
        best_allocations = allocations.copy()

        print(f"Initial cost: {current_cost}")  
        
        iteration = 0
        while time.time() - start_time < self.time_limit:
            # Create a copy of current solution to modify
            temp_solution = self._copy_solution(current_solution)
            temp_allocations = allocations.copy()
            
            # Remove allocations
            tik = time.time()
            removed_allocations = random_removal(temp_solution, self.removal_size)
            tok = time.time()
            removal_time += tok - tik
            
            # Skip iteration if no allocations were removed
            if not removed_allocations:
                continue
                
            # Update allocations list by removing the removed allocations
            for agent_idx, task_idx, start_idx, goal_idx in removed_allocations:
                temp_allocations = [alloc for alloc in temp_allocations 
                                  if not (alloc[0] == agent_idx and alloc[1] == task_idx)]
                
            # Update cost tensor based on removed allocations
            tik = time.time()
            updated_tensor = self._update_cost_tensor(removed_allocations)
            tok = time.time()
            update_cost_tensor_time += tok - tik
            
            # Repair solution
            tik = time.time()
            new_solution = greedy_repair(self.S, self.G, updated_tensor, temp_solution, 
                                       self.start_locs, self.goal_locs, 
                                       self.idx_to_task_id, temp_allocations, self.cost_calculation_method)
            tok = time.time()
            repair_time += tok - tik
            print(f"Iteration {iteration} repair time: {tok-tik}")
            
            # Calculate new cost
            new_cost = self._calculate_total_cost(new_solution)
            
            # Update if better
            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
                allocations = temp_allocations
                
                # Update best if better
                if current_cost < self.best_cost:
                    self.best_solution = current_solution
                    self.best_cost = current_cost
                    best_allocations = allocations.copy()
            
            iteration += 1
            
        print(f"LNS completed {iteration} iterations in {time.time() - start_time:.2f} seconds")
        print(f"Best cost found: {self.best_cost}")

        print(f"Inital task assignment time: {inital_task_assignment_time}")
        print(f"Removal time: {removal_time}")
        print(f"Update cost tensor time: {update_cost_tensor_time}")
        print(f"Repair time: {repair_time}")
        
        return self.best_solution, best_allocations, self.best_cost
    
    def _copy_solution(self, solution: AgentLoader) -> AgentLoader:
        """Create a deep copy of the current solution."""
        new_solution = AgentLoader([])
        for agent in solution.agents:
            new_agent = agent.__class__(agent.id, agent.state)
            new_agent.task_sequence = agent.task_sequence.copy()
            new_agent.status = agent.status
            new_solution.agents.append(new_agent)
        return new_solution
    
    def _calculate_total_cost(self, solution: AgentLoader) -> float:
        """Calculate total cost of a solution."""
        total_cost = 0.0
        for agent in solution.agents:
            if len(agent.task_sequence) > 1:  # Skip if only current task
                for i in range(1, len(agent.task_sequence)):
                    prev_task = agent.task_sequence[i-1]
                    curr_task = agent.task_sequence[i]
                    
                    # Get locations
                    prev_goal = prev_task[2]  # goal location of previous task
                    curr_start = curr_task[1]  # start location of current task
                    curr_goal = curr_task[2]   # goal location of current task
                    
                    # Calculate costs
                    if self.cost_calculation_method == "manhattan":
                        goal_to_start = manhattan_distance(prev_goal, curr_start)
                        start_to_goal = manhattan_distance(curr_start, curr_goal)
                    else:  # shortest_path
                        goal_to_start = self.G.get_distance(prev_goal, curr_start)
                        start_to_goal = self.G.get_distance(curr_start, curr_goal)
                    
                    total_cost += goal_to_start + start_to_goal
        return total_cost
    
    def _update_cost_tensor(self, removed_allocations: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Update cost tensor based on removed allocations.
        
        Args:
            removed_allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples
            
        Returns:
            Updated cost tensor
        """
        updated_tensor = self.cost_tensor.copy()
        
        # Reset costs for removed allocations
        for agent_idx, task_idx, start_idx, goal_idx in removed_allocations:
            # Find the task index in the current cost tensor
            for tensor_task_idx, task_id in self.idx_to_task_id.items():
                if task_id == task_idx:
                    updated_tensor[agent_idx, tensor_task_idx, start_idx, goal_idx] = np.inf
                    break
            
        return updated_tensor

def py_lns_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
                initial_solution: AgentLoader, time_limit: float = 1.0,
                removal_size: int = 3, cost_calculation_method: str = "manhattan") -> AgentLoader:
    """
    Call the LNS algorithm with given parameters.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader object
        J: Set of tasks
        initial_solution: Initial solution from another algorithm
        time_limit: Maximum time to run LNS in seconds
        removal_size: Number of allocations to remove in each iteration
        cost_calculation_method: Method to use for cost calculation
        
    Returns:
        Updated AgentLoader with the best solution found
    """
    lns = LNS(S, G, Rs, J, initial_solution, time_limit, removal_size, cost_calculation_method)
    return lns.run()
