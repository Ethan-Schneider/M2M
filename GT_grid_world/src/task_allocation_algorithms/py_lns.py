import numpy as np
import time
from typing import Set, Tuple, List, Dict
from ..utils import manhattan_distance
from ..graph import Graph
from ..agent import AgentLoader
from ..analysis.statistics import Stats
from .initial_solutions.construct_cost_tensor import construct_cost_tensor
from .removal_operators.random_removal import random_removal
from .removal_operators.worst_removal import worst_removal
from .repair_operators.greedy_repair import greedy_repair

from .initial_solutions.random_allocation import random_call
from .initial_solutions.greedy_allocation import greedy_call
from .initial_solutions.randomized_greedy import randomized_greedy_call
from .initial_solutions.FCF import FCF_call
from .initial_solutions.max_regret_FC import max_regret_FC_call
from .initial_solutions.randomized_max_regret_FC import randomized_max_regret_FC_call
from .initial_solutions.fast_greedy import fast_greedy_call
from .initial_solutions.fast_FCF import fast_FCF_call

class LNS:
    def __init__(self, S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
                 initial_task_assignment_strategy : str, time_limit: float = 1.0,
                 removal_size: int = 3, cost_calculation_method: str = "manhattan",
                 removal_operator: str = "worst", repair_operator: str = "greedy"):
        """
        Initialize Large Neighborhood Search algorithm.
        
        Args:
            S: Statistics object for tracking metrics
            G: Graph representing the warehouse
            Rs: AgentLoader containing all agents
            J: Set of tasks to be assigned
            initial_task_assignment_strategy: Strategy for initial task assignment
            time_limit: Maximum time to run LNS in seconds
            removal_size: Number of allocations to remove in each iteration
            cost_calculation_method: Method to use for cost calculation ("manhattan" or "shortest_path")
            removal_operator: Removal operator to use ("random" or "worst")
            repair_operator: Repair operator to use ("greedy")
        """
        self.S = S
        self.G = G
        self.Rs = Rs
        self.J = J
        self.initial_task_assignment_strategy = initial_task_assignment_strategy
        self.time_limit = time_limit
        self.removal_size = removal_size
        self.cost_calculation_method = cost_calculation_method
        self.removal_operator = removal_operator
        self.repair_operator = repair_operator
        
        # Store best solution found
        self.best_solution = None
        self.best_cost = float('inf')
        
        # Construct initial cost tensor
        self.cost_tensor, self.cost_tensor_agent_start, self.start_locs, self.goal_locs, self.idx_to_task_id = construct_cost_tensor(
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
        if self.initial_task_assignment_strategy == "random":
            current_solution, allocations, _ = random_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "greedy":
            current_solution, allocations, _ = greedy_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "randomized_greedy":
            current_solution, allocations, _ = randomized_greedy_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "FCF":
            current_solution, allocations, _ = FCF_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "max_regret_FC":
            current_solution, allocations, _ = max_regret_FC_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "randomized_max_regret_FC":
            current_solution, allocations, _ = randomized_max_regret_FC_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "fast_greedy":
            current_solution, allocations, _ = fast_greedy_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
        elif self.initial_task_assignment_strategy == "fast_FCF":
            current_solution, allocations, _ = fast_FCF_call(self.S, self.G, self.Rs, self.J, self.cost_calculation_method)
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

        print(f"Current Solution:")
        for agent in current_solution.agents:
            print(f"Agent {agent.id} task sequence: {agent.task_sequence}")

        print(f"Initial cost: {current_cost}")  
        
        iteration = 0
        print(f"Time so far: {time.time() - start_time}")
        print(f"Time limit: {self.time_limit}")
        while time.time() - start_time < self.time_limit:
            # Create a copy of current solution to modify
            temp_solution = self._copy_solution(current_solution)
            temp_allocations = allocations.copy()
            
            # Remove allocations
            tik = time.time()
            if self.removal_operator == "random":
                temp_solution, temp_allocations, removed_allocations = random_removal(temp_solution, temp_allocations, self.removal_size)
            elif self.removal_operator == "worst":
                temp_solution, temp_allocations, removed_allocations = worst_removal(temp_solution, temp_allocations, self.removal_size, 
                                                                                   self.G, self.start_locs, self.goal_locs, self.cost_calculation_method)
            else:
                print(f"ERROR: Unknown removal operator {self.removal_operator}, using worst removal")
                temp_solution, temp_allocations, removed_allocations = worst_removal(temp_solution, temp_allocations, self.removal_size, 
                                                                                   self.G, self.start_locs, self.goal_locs, self.cost_calculation_method)
            tok = time.time()
            removal_time += tok - tik

            # Update cost tensor based on removed allocations
            tik = time.time()
            updated_tensor, updated_start_locs, updated_goal_locs, updated_idx_to_task_id = self._update_cost_tensor(removed_allocations, temp_solution)
            tok = time.time()
            update_cost_tensor_time += tok - tik
            
            # Repair solution
            tik = time.time()
            if self.repair_operator == "greedy":
                new_solution, temp_allocations, new_cost = greedy_repair(self.S, self.G, updated_tensor, self.cost_tensor_agent_start, temp_solution, 
                                           updated_start_locs, updated_goal_locs, 
                                           updated_idx_to_task_id, temp_allocations, self.cost_calculation_method)
            else:
                print(f"ERROR: Unknown repair operator {self.repair_operator}, using greedy repair")
                new_solution, temp_allocations, new_cost = greedy_repair(self.S, self.G, updated_tensor, self.cost_tensor_agent_start, temp_solution, 
                                           updated_start_locs, updated_goal_locs, 
                                           updated_idx_to_task_id, temp_allocations, self.cost_calculation_method)
            
            tok = time.time()
            repair_time += tok - tik
            
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
        print(f"New Best Task Allocation:")
        for agent in self.best_solution.agents:
            print(f"Agent {agent.id} task sequence: {agent.task_sequence}")

        print(f"Inital task assignment time: {inital_task_assignment_time}")
        print(f"Removal time: {removal_time}")
        print(f"Update cost tensor time: {update_cost_tensor_time}")
        print(f"Repair time: {repair_time}")

        for agent in self.best_solution.agents:
            if len(agent.task_sequence) == 0:
                agent.status = 0 # Set agent status to idle
        
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
            elif len(agent.task_sequence) == 1:
                total_cost += self.G.get_distance(agent.state, agent.task_sequence[0][1])
                total_cost += self.G.get_distance(agent.task_sequence[0][1], agent.task_sequence[0][2])
        return total_cost
    
    def _update_cost_tensor(self, removed_allocations: List[Tuple[int, int, int, int]], current_solution: AgentLoader) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, int]]:
        """
        Reconstruct cost tensor to include removed tasks as available for re-allocation.
        
        Args:
            removed_allocations: List of (agent_idx, task_idx, start_idx, goal_idx) tuples
            current_solution: Current solution state after removal
            
        Returns:
            Tuple containing:
            - Updated cost tensor
            - Updated start locations
            - Updated goal locations  
            - Updated idx to task_id mapping
        """
        # Reconstruct the cost tensor from scratch to include removed tasks
        from .initial_solutions.construct_cost_tensor import construct_cost_tensor
        
        # Reconstruct cost tensor with the current solution state
        updated_tensor, updated_agent_start, updated_start_locs, updated_goal_locs, updated_idx_to_task_id = construct_cost_tensor(
            self.J, current_solution, self.G, self.cost_calculation_method
        )
        
        # Update the agent start cost tensor as well
        self.cost_tensor_agent_start = updated_agent_start
        
        return updated_tensor, updated_start_locs, updated_goal_locs, updated_idx_to_task_id

def py_lns_call(S: Stats, G: Graph, Rs: AgentLoader, J: Set[Tuple], 
                initial_task_assignment_strategy: str, time_limit: float = 1.0,
                removal_size: int = 3, cost_calculation_method: str = "manhattan",
                removal_operator: str = "worst", repair_operator: str = "greedy") -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Call the LNS algorithm with given parameters.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader object
        J: Set of tasks
        initial_task_assignment_strategy: Strategy for initial task assignment
        time_limit: Maximum time to run LNS in seconds
        removal_size: Number of allocations to remove in each iteration
        cost_calculation_method: Method to use for cost calculation
        removal_operator: Removal operator to use ("random" or "worst")
        repair_operator: Repair operator to use ("greedy")
        
    Returns:
        Tuple containing:
        - Updated AgentLoader with the best solution found
        - List of allocations (agent_idx, task_idx, start_idx, goal_idx)
        - Best cost found
    """
    lns = LNS(S, G, Rs, J, initial_task_assignment_strategy, time_limit, removal_size, 
              cost_calculation_method, removal_operator, repair_operator)
    return lns.run()
