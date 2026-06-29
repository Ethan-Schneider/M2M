import numpy as np
import math
import random
import time
from typing import Set, Tuple, List, Dict
from ..utils import manhattan_distance
from ..graph import Graph
from ..agent import AgentLoader
from ..analysis.statistics import Stats
from .initial_solutions.construct_cost_elements import construct_cost_elements
from .removal_operators.random_removal import random_removal
from .removal_operators.worst_removal import worst_removal
from .removal_operators.shaw_removal import shaw_removal
from .repair_operators.greedy_repair import greedy_repair
from .repair_operators.fast_SCF_repair import fast_SCF_repair

from .initial_solutions.construct_cost_elements import (
    CRM2M_DEFAULT_LAMBDA,
    CRM2M_DEFAULT_DETOUR_CUTOFF,
)
from .initial_solutions.fast_greedy import fast_greedy_allocation
from .initial_solutions.fast_FCF import fast_FCF_allocation
from .initial_solutions.fast_SCF import fast_SCF_allocation

class LNS:
    def __init__(self, S: Stats, G: Graph, Rs: AgentLoader, J: Dict[int, Tuple], 
                 initial_task_assignment_strategy : str, time_limit: float = 1.0,
                 removal_size: int = 3, cost_calculation_method: str = "manhattan",
                 removal_operator: str = "worst", repair_operator: str = "greedy",
                 acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
                 current_time: int = 0,
                 base_cost_weight: float = 1.0,
                 deadline_weight: float = 0.0,
                 sku_distribution_weight: float = 0.0,
                 agent_unallocated_penalty: float = 0.0,
                 agent_task_sequence_limit : int = -1,
                 crm2m_lambda: float = CRM2M_DEFAULT_LAMBDA,
                 crm2m_detour_cutoff: float = CRM2M_DEFAULT_DETOUR_CUTOFF):
        """
        Initialize Large Neighborhood Search algorithm.
        
        Args:
            S: Statistics object for tracking metrics
            G: Graph representing the warehouse
            Rs: AgentLoader containing all agents
            J: Dict of tasks
            initial_task_assignment_strategy: Strategy for initial task assignment
            time_limit: Maximum time to run LNS in seconds
            removal_size: Number of allocations to remove in each iteration
            cost_calculation_method: Method to use for cost calculation ("manhattan" or "shortest_path")
            removal_operator: Removal operator to use ("random" or "worst" or "shaw")
            repair_operator: Repair operator to use ("greedy" or "fast_SCF")
            acceptance_function: Acceptance function to use ("greedy" or "simulated_annealing" etc.)
            T_0: Initial temperature for simulated annealing
            alpha: Cooling factor for simulated annealing
            current_time: Current timestep for deadline calculations
        """
        self.S = S
        self.G = G
        self.Rs = self._copy_solution(Rs)

        self.J = J
        self.initial_task_assignment_strategy = initial_task_assignment_strategy
        self.time_limit = time_limit
        self.removal_size = removal_size
        self.cost_calculation_method = cost_calculation_method
        self.removal_operator = removal_operator
        self.repair_operator = repair_operator
        self.acceptance_function = acceptance_function
        self.T_0 = T_0
        self.alpha = alpha
        self.current_time = current_time
        self.base_cost_weight = base_cost_weight
        self.deadline_weight = deadline_weight
        self.sku_distribution_weight = sku_distribution_weight
        self.agent_unallocated_penalty = agent_unallocated_penalty
        self.agent_task_sequence_limit = agent_task_sequence_limit
        self.crm2m_lambda = crm2m_lambda
        self.crm2m_detour_cutoff = crm2m_detour_cutoff
        # Store best solution found
        self.best_solution = None
        self.best_cost = np.inf

        self.initial_cost = np.inf

        # Cost lookup table to store allocation costs {(agent_idx, task_id, start_idx, goal_idx): cost}
        self.cost_lookup = {}
        
        # Construct initial cost elements
        (self.agent_start_cost_tensor, self.start_goal_dist, self.task_start_mask, self.task_goal_mask,
         self.start_locs, self.goal_locs, self.idx_to_task_id, self.task_id_to_idx,
         self.task_deadline_costs, self.inbound_sku_distribution_costs, self.outbound_sku_distribution_costs,
         self.rearrangement_sku_distribution_costs, self.agent_task_sequence_time
         ) = construct_cost_elements(self.J, self.Rs, self.G, self.current_time, self.cost_calculation_method)
        print(f"original start_locs: {self.start_locs}")
        
    def run(self, t: int = None) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
        """
        Run the LNS algorithm.
        
        Returns:
            Tuple containing:
            - Updated AgentLoader with the best solution found
            - List of allocations (agent_idx, task_idx, start_idx, goal_idx)
            - Best cost found
        """
        start_time = time.time()
        inital_task_assignment_time = 0.0
        removal_time = 0.0
        repair_time = 0.0
        cost_computation_time = 0.0

        inital_task_assignment_tik = time.time()

        solution_costs = []
        destroyed_solution_costs = []
        repaired_solution_costs = []
        number_of_tasks_removed = []

        # Initial task assignment
        if self.initial_task_assignment_strategy == "fast_greedy":
            current_solution, allocations, __, = fast_greedy_allocation(self.S, self.G, self.Rs, self.start_locs, 
                                                                        self.goal_locs, self.idx_to_task_id, self.J, 
                                                                        self.cost_calculation_method, 
                                                                        self.agent_start_cost_tensor, 
                                                                        self.start_goal_dist, self.task_start_mask, 
                                                                        self.task_goal_mask, cost_lookup=self.cost_lookup, task_deadline_costs=self.task_deadline_costs,
                                                                        inbound_sku_distribution_costs=self.inbound_sku_distribution_costs,
                                                                        outbound_sku_distribution_costs=self.outbound_sku_distribution_costs,
                                                                        rearrangement_sku_distribution_costs=self.rearrangement_sku_distribution_costs,
                                                                        base_cost_weight=self.base_cost_weight,
                                                                        deadline_weight=self.deadline_weight,
                                                                        sku_distribution_weight=self.sku_distribution_weight,
                                                                        agent_task_sequence_time=self.agent_task_sequence_time, 
                                                                        agent_task_sequence_limit=self.agent_task_sequence_limit,
                                                                        crm2m_lambda=self.crm2m_lambda,
                                                                        crm2m_detour_cutoff=self.crm2m_detour_cutoff)
        elif self.initial_task_assignment_strategy == "fast_FCF":
            current_solution, allocations, __ = fast_FCF_allocation(self.S, self.G, self.Rs, self.start_locs, self.goal_locs, self.idx_to_task_id, self.cost_calculation_method, self.agent_start_cost_tensor, self.start_goal_dist, self.task_start_mask, self.task_goal_mask, cost_lookup=self.cost_lookup)
        elif self.initial_task_assignment_strategy == "fast_SCF":
            current_solution, allocations, __ = fast_SCF_allocation(self.S, self.G, self.Rs, self.start_locs, self.goal_locs, self.idx_to_task_id, self.cost_calculation_method, self.agent_start_cost_tensor, self.start_goal_dist, self.task_start_mask, self.task_goal_mask, cost_lookup=self.cost_lookup)
        else:
            print("ERROR: Unknown initial task assignment strategy " + self.initial_task_assignment_strategy + ", please choose another one.")
            return self.Rs, [], np.inf
        
        inital_task_assignment_time += time.time() - inital_task_assignment_tik
        #Iterate over task assignment, if duplicate tasks are found, exit
        task_ids = []
        for agent in current_solution.agents:
            for task in agent.task_sequence:
                if task[0] in task_ids:
                    print(f"Duplicate task found: {task}")
                    exit(0)
                task_ids.append(task[0])

        # Initialize best and current solution and allocations
        self.best_solution = self._copy_solution(current_solution)
        self.best_cost = self._calculate_total_cost(self.cost_lookup, self.Rs)
        self.initial_cost = self.best_cost
        best_allocations = allocations.copy()
        
        current_solution = self._copy_solution(current_solution)
        current_allocations = allocations.copy()
        current_cost_lookup = self.cost_lookup.copy()
        current_cost = self.best_cost

        current_agent_start_cost_tensor = self.agent_start_cost_tensor.copy()
        current_start_goal_dist = self.start_goal_dist.copy()
        current_task_start_mask = self.task_start_mask.copy()
        current_task_goal_mask = self.task_goal_mask.copy()
        current_task_deadline_costs = self.task_deadline_costs.copy()
        current_agent_task_sequence_time = self.agent_task_sequence_time.copy()
        
        # LNS logging
        lns_log = {
            "timestep": t,
            "initial_cost": self.initial_cost,
            "acceptance_function": self.acceptance_function,
            "T_0": self.T_0,
            "alpha": self.alpha,
            "improvements": [
                {
                    "iteration": 0,
                    "wall_time": 0.0,
                    "cost": self.initial_cost
                }
            ],
            "final_best_cost": None,
            "total_iterations": None
        }
        
        T = self.T_0
        iteration = 0

        while time.time() - start_time < self.time_limit:
            # if iteration >= 100:
            #     print(f"Exiting at iteration {iteration}")
            #     break
            # print(f"Iteration: {iteration}")
            # Create a copy of the current solution to modify
            temp_solution = self._copy_solution(current_solution)
            temp_cost_lookup = current_cost_lookup.copy()
            temp_allocations = current_allocations.copy()

            temp_agent_start_cost_tensor = current_agent_start_cost_tensor.copy()
            temp_start_goal_dist = current_start_goal_dist.copy()
            temp_task_start_mask = current_task_start_mask.copy()
            temp_task_goal_mask = current_task_goal_mask.copy()
            temp_task_deadline_costs = current_task_deadline_costs.copy()
            temp_agent_task_sequence_time = current_agent_task_sequence_time.copy()

            num_before = len(temp_cost_lookup.keys())
            
            # print(f"Number tasks before removal: {num_before}: Iteration: {iteration}")

            # Remove allocations
            removal_tik = time.time()
            if self.removal_operator == "random":
                temp_solution, temp_allocations, temp_cost_lookup = random_removal(temp_solution, temp_allocations, self.removal_size, self.idx_to_task_id, self.start_locs, self.goal_locs, temp_cost_lookup,
                                                                                   G=self.G, method=self.cost_calculation_method, task_id_to_idx=self.task_id_to_idx,
                                                                                   task_start_mask=temp_task_start_mask, task_goal_mask=temp_task_goal_mask,
                                                                                   original_task_start_mask=self.task_start_mask, original_task_goal_mask=self.task_goal_mask,
                                                                                   agent_start_cost_tensor=temp_agent_start_cost_tensor, agent_task_sequence_time=temp_agent_task_sequence_time)
            elif self.removal_operator == "worst":
                temp_solution, temp_allocations, temp_cost_lookup = worst_removal(temp_solution, temp_allocations, self.removal_size, 
                                                                                   self.G, self.start_locs, self.goal_locs, self.cost_calculation_method, temp_cost_lookup,
                                                                                   task_id_to_idx=self.task_id_to_idx, task_start_mask=temp_task_start_mask,
                                                                                   task_goal_mask=temp_task_goal_mask, original_task_start_mask=self.task_start_mask,
                                                                                   original_task_goal_mask=self.task_goal_mask, agent_start_cost_tensor=temp_agent_start_cost_tensor,
                                                                                   agent_task_sequence_time=temp_agent_task_sequence_time)
            elif self.removal_operator == "shaw":
                temp_solution, temp_allocations, temp_cost_lookup, temp_task_start_mask, temp_task_goal_mask, temp_agent_start_cost_tensor, temp_agent_task_sequence_time = shaw_removal(temp_solution, temp_allocations, self.removal_size, 
                                                                                   self.G, self.start_locs, self.goal_locs, self.cost_calculation_method, temp_cost_lookup, self.task_id_to_idx,
                                                                                   agent_start_cost_tensor=temp_agent_start_cost_tensor,
                                                                                   start_goal_dist=temp_start_goal_dist,
                                                                                   task_start_mask=temp_task_start_mask,
                                                                                   task_goal_mask=temp_task_goal_mask,
                                                                                   original_task_start_mask=self.task_start_mask,
                                                                                   original_task_goal_mask=self.task_goal_mask,
                                                                                   agent_task_sequence_time=temp_agent_task_sequence_time)
            else:
                print(f"ERROR: Unknown removal operator {self.removal_operator}, using worst removal")
                temp_solution, temp_allocations, temp_cost_lookup = worst_removal(temp_solution, temp_allocations, self.removal_size, 
                                                                                   self.G, self.start_locs, self.goal_locs, self.cost_calculation_method, temp_cost_lookup,
                                                                                   task_id_to_idx=self.task_id_to_idx, task_start_mask=temp_task_start_mask,
                                                                                   task_goal_mask=temp_task_goal_mask, original_task_start_mask=self.task_start_mask,
                                                                                   original_task_goal_mask=self.task_goal_mask, agent_start_cost_tensor=temp_agent_start_cost_tensor,
                                                                                   agent_task_sequence_time=temp_agent_task_sequence_time)
            removal_time += time.time() - removal_tik

            num_after = len(temp_cost_lookup.keys())
            number_of_tasks_removed.append(num_before - num_after)
            
            # print(f"Number tasks after removal: {num_after}: Iteration: {iteration}")

            destroyed_solution_costs.append(self._calculate_total_cost(temp_cost_lookup, temp_solution))
            
            # Repair solution
            tik = time.time()
            if self.repair_operator == "greedy":
                new_solution, __, temp_cost_lookup, temp_task_start_mask, temp_task_goal_mask, temp_agent_start_cost_tensor, temp_agent_task_sequence_time = greedy_repair(self.S, self.G, temp_agent_start_cost_tensor, 
                                                                         temp_start_goal_dist, temp_task_start_mask, temp_task_goal_mask, 
                                                                         temp_solution, self.start_locs, self.goal_locs, 
                                                                         self.idx_to_task_id, temp_allocations, self.cost_calculation_method, self.J, temp_cost_lookup,
                                                                         inbound_sku_distribution_costs=self.inbound_sku_distribution_costs,
                                                                         outbound_sku_distribution_costs=self.outbound_sku_distribution_costs,
                                                                         rearrangement_sku_distribution_costs=self.rearrangement_sku_distribution_costs,
                                                                         task_deadline_costs=self.task_deadline_costs,
                                                                         base_cost_weight=self.base_cost_weight,
                                                                         deadline_weight=self.deadline_weight,
                                                                         sku_distribution_weight=self.sku_distribution_weight,
                                                                         agent_task_sequence_time=temp_agent_task_sequence_time,
                                                                         current_time=self.current_time, agent_task_sequence_limit=self.agent_task_sequence_limit,
                                                                         crm2m_lambda=self.crm2m_lambda, crm2m_detour_cutoff=self.crm2m_detour_cutoff)
                # Update the temp cost elements with the returned values
                temp_start_goal_dist = temp_start_goal_dist.copy()
                temp_task_deadline_costs = temp_task_deadline_costs.copy()
                temp_agent_task_sequence_time = temp_agent_task_sequence_time.copy()
            elif self.repair_operator == "fast_SCF":
                new_solution, temp_allocations, __, temp_cost_lookup, temp_task_start_mask, temp_task_goal_mask, temp_agent_start_cost_tensor = fast_SCF_repair(self.S, self.G, temp_agent_start_cost_tensor, 
                                                                         temp_start_goal_dist, temp_task_start_mask, temp_task_goal_mask, 
                                                                         temp_solution, self.start_locs, self.goal_locs, 
                                                                         self.idx_to_task_id, temp_allocations, self.cost_calculation_method, temp_cost_lookup,
                                                                         J=self.J, inbound_sku_distribution_costs=self.inbound_sku_distribution_costs,
                                                                         outbound_sku_distribution_costs=self.outbound_sku_distribution_costs,
                                                                         rearrangement_sku_distribution_costs=self.rearrangement_sku_distribution_costs,
                                                                         base_cost_weight=self.base_cost_weight, deadline_weight=self.deadline_weight,
                                                                         sku_distribution_weight=self.sku_distribution_weight,
                                                                         agent_task_sequence_time=temp_agent_task_sequence_time,
                                                                         current_time=self.current_time,
                                                                         crm2m_lambda=self.crm2m_lambda, crm2m_detour_cutoff=self.crm2m_detour_cutoff)
                # Update the temp cost elements with the returned values
                temp_start_goal_dist = temp_start_goal_dist.copy()
                temp_task_deadline_costs = temp_task_deadline_costs.copy()
                temp_agent_task_sequence_time = temp_agent_task_sequence_time.copy()
            else:
                print(f"ERROR: Unknown repair operator {self.repair_operator}, using greedy repair")
                new_solution, temp_allocations, __, temp_cost_lookup, temp_task_start_mask, temp_task_goal_mask, temp_agent_start_cost_tensor, temp_agent_task_sequence_time = greedy_repair(self.S, self.G, temp_agent_start_cost_tensor, temp_start_goal_dist, 
                                                                         temp_task_start_mask, temp_task_goal_mask, 
                                                                         temp_solution, self.start_locs, self.goal_locs, 
                                                                         self.idx_to_task_id, temp_allocations, self.cost_calculation_method, self.J, temp_cost_lookup,
                                                                         inbound_sku_distribution_costs=self.inbound_sku_distribution_costs,
                                                                         outbound_sku_distribution_costs=self.outbound_sku_distribution_costs,
                                                                         rearrangement_sku_distribution_costs=self.rearrangement_sku_distribution_costs,
                                                                         task_deadline_costs=self.task_deadline_costs,
                                                                         base_cost_weight=self.base_cost_weight, deadline_weight=self.deadline_weight,
                                                                         sku_distribution_weight=self.sku_distribution_weight,
                                                                         agent_task_sequence_time=temp_agent_task_sequence_time,
                                                                         current_time=self.current_time,
                                                                         crm2m_lambda=self.crm2m_lambda, crm2m_detour_cutoff=self.crm2m_detour_cutoff)
                # Update the temp cost elements with the returned values
                temp_start_goal_dist = temp_start_goal_dist.copy()
                temp_task_deadline_costs = temp_task_deadline_costs.copy()
                temp_agent_task_sequence_time = temp_agent_task_sequence_time.copy()
            
            tok = time.time()
            repair_time += tok - tik
            
            # print(f"Number tasks after repair: {len(temp_cost_lookup.keys())}: Iteration: {iteration}")

            cost_tik = time.time()
            new_cost = self._calculate_total_cost(temp_cost_lookup, new_solution)
            cost_computation_time += time.time() - cost_tik
            
            # print(f"New Cost: {new_cost} vs. Best Cost {self.best_cost}")

            # Debug: Print cost information every 50 iterations
            # if iteration % 50 == 0:
            #     print(f"Iteration {iteration}: Initial cost: {self.initial_cost}, Current cost: {current_cost}, New cost: {new_cost}, Best cost: {self.best_cost}")

            solution_costs.append(float(new_cost))
            repaired_solution_costs.append(new_cost)
            # Acceptance function
            if self.acceptance_function == "greedy":
                if new_cost < current_cost:
                    current_solution = self._copy_solution(new_solution)
                    current_allocations = temp_allocations.copy()
                    current_cost_lookup = temp_cost_lookup.copy()
                    current_agent_start_cost_tensor = temp_agent_start_cost_tensor.copy()
                    current_start_goal_dist = temp_start_goal_dist.copy()
                    current_task_start_mask = temp_task_start_mask.copy()
                    current_task_goal_mask = temp_task_goal_mask.copy()
                    current_task_deadline_costs = temp_task_deadline_costs.copy()
                    current_agent_task_sequence_time = temp_agent_task_sequence_time.copy()
                    current_cost = new_cost
            elif self.acceptance_function == "simulated_annealing":
                if new_cost < current_cost:
                    current_solution = self._copy_solution(new_solution)
                    current_allocations = temp_allocations.copy()
                    current_cost_lookup = temp_cost_lookup.copy()
                    current_agent_start_cost_tensor = temp_agent_start_cost_tensor.copy()
                    current_start_goal_dist = temp_start_goal_dist.copy()
                    current_task_start_mask = temp_task_start_mask.copy()
                    current_task_goal_mask = temp_task_goal_mask.copy()
                    current_task_deadline_costs = temp_task_deadline_costs.copy()
                    current_agent_task_sequence_time = temp_agent_task_sequence_time.copy()
                    current_cost = new_cost
                else:
                    prob = math.exp(-1*(new_cost - current_cost) / T) if T > 0 else 0
                    # print(f"prob: {prob}")
                    if random.random() < prob:
                        current_solution = self._copy_solution(new_solution)
                        current_allocations = temp_allocations.copy()
                        current_cost_lookup = temp_cost_lookup.copy()
                        current_agent_start_cost_tensor = temp_agent_start_cost_tensor.copy()
                        current_start_goal_dist = temp_start_goal_dist.copy()
                        current_task_start_mask = temp_task_start_mask.copy()
                        current_task_goal_mask = temp_task_goal_mask.copy()
                        current_task_deadline_costs = temp_task_deadline_costs.copy()
                        current_agent_task_sequence_time = temp_agent_task_sequence_time.copy()
                        current_cost = new_cost
                T = self.alpha * T

            # Update best if better
            if new_cost < self.best_cost:
                print(f"IMPROVEMENT FOUND! Iteration {iteration}: New best cost: {new_cost} (previous: {self.best_cost})")
                self.best_solution = self._copy_solution(new_solution)
                self.best_cost = new_cost
                best_allocations = temp_allocations.copy()
                best_cost_lookup = temp_cost_lookup.copy()
                lns_log["improvements"].append({
                    "iteration": int(iteration + 1),
                    "wall_time": float(np.abs(time.time() - start_time)),
                    "cost": float(new_cost)
                })
            
            iteration += 1
            
        print(f"LNS completed {iteration} iterations in {time.time() - start_time:.2f} seconds")
        print(f"Best cost found: {self.best_cost}")
        print(f"Initial cost: {self.initial_cost}")

        print(f"Initial task assignment time: {inital_task_assignment_time}")
        print(f"Total removal time: {removal_time}")
        print(f"Total repair time: {repair_time}")
        print(f"Total cost computation time: {cost_computation_time}")

        print(f"Total time: {time.time() - start_time}")

        # print(f"solution costs: {solution_costs}")
        # print(f"destroyed solution costs: {destroyed_solution_costs}")
        # print(f"repaired solution costs: {repaired_solution_costs}")
        # print(f"number of tasks removed: {number_of_tasks_removed}")
        # print(f"average number of removed tasks: {np.average(number_of_tasks_removed)}")
        print(f"best solution: {[agent.task_sequence for agent in self.best_solution.agents]}")

        for agent in self.best_solution.agents:
            if len(agent.task_sequence) == 0:
                agent.status = 0 # Set agent status to idle
        
        lns_log["final_best_cost"] = self.best_cost
        lns_log["total_iterations"] = iteration
        self.S.append_py_lns_log(lns_log)
        
        print(f"Final agent task sequences: {[agent.task_sequence for agent in self.best_solution.agents]}")

        return self.best_solution, best_allocations, self.best_cost
    
    def _copy_solution(self, solution: AgentLoader) -> AgentLoader:
        """Create a deep copy of the current solution."""
        new_solution = AgentLoader([])
        for agent in solution.agents:
            new_agent = agent.__class__(agent.id, agent.state, task_sequence=agent.task_sequence.copy(), home=agent.home)
            new_agent.status = agent.status
            new_agent.path_sequence = agent.path_sequence.copy()
            new_agent.sku_id_carrying = agent.sku_id_carrying
            new_solution.agents.append(new_agent)
        return new_solution
    
    def _calculate_total_cost(self, cost_lookup: Dict[Tuple[int, int, int, int], int], Rs : AgentLoader) -> float:
        """Calculate total cost of a solution."""
        # Sum up all the costs in the cost_lookup
        total_cost = sum(cost_lookup.values())

        # Penalize unallocated agents
        for agent in Rs.agents:
            if len(agent.task_sequence) == 0:
                total_cost += self.agent_unallocated_penalty
        return total_cost
    
    def _calculate_total_cost_V2(self, Rs: AgentLoader, G : Graph) -> float:
        total_cost = 0.0
        for agent in Rs.agents:
            if len(agent.task_sequence) == 0:
                # Add penalty for unallocated agent
                total_cost -= 10
                continue
            total_cost -= G.get_distance(agent.state, agent.task_sequence[0][1])
            for i in range(len(agent.task_sequence) - 1):
                total_cost -= G.get_distance(agent.task_sequence[i][2], agent.task_sequence[i+1][1])
        return total_cost
    
def py_lns_call(S: Stats, G: Graph, Rs: AgentLoader, J: Dict[int, Tuple], 
                initial_task_assignment_strategy: str, time_limit: float = 1.0,
                removal_size: int = 3, cost_calculation_method: str = "manhattan",
                removal_operator: str = "worst", repair_operator: str = "greedy", t: int = None,
                acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
                base_cost_weight: float = 1.0,
                deadline_weight: float = 0.0,
                sku_distribution_weight: float = 0.0,
                agent_unallocated_penalty: float = 0.0,
                crm2m_lambda: float = CRM2M_DEFAULT_LAMBDA,
                crm2m_detour_cutoff: float = CRM2M_DEFAULT_DETOUR_CUTOFF) -> Tuple[AgentLoader, List[Tuple[int, int, int, int]], float]:
    """
    Call the LNS algorithm with given parameters.
    
    Args:
        S: Statistics object
        G: Graph object
        Rs: AgentLoader object
        J: Dict of tasks
        initial_task_assignment_strategy: Strategy for initial task assignment
        time_limit: Maximum time to run LNS in seconds
        removal_size: Number of allocations to remove in each iteration
        cost_calculation_method: Method to use for cost calculation
        removal_operator: Removal operator to use ("random" or "worst" or "shaw")
        repair_operator: Repair operator to use ("greedy" or "fast_SCF")
        t: Current timestep for deadline calculations
        
    Returns:
        Tuple containing:
        - Updated AgentLoader with the best solution found
        - List of allocations (agent_idx, task_idx, start_idx, goal_idx)
        - Best cost found
    """
    num_allocated_tasks = 0
    for agent in Rs.agents:
        num_allocated_tasks += len(agent.task_sequence)
    if len(J.keys()) == num_allocated_tasks:  # No tasks to assign
        return Rs, [], 0.0

    # Remove all but the first task in each agent's task sequence
    for agent in Rs.agents:
        if agent.task_sequence == []:
            continue
        while len(agent.task_sequence) > 1:
            agent.task_sequence.pop(-1)
    
    lns = LNS(S, G, Rs, J, initial_task_assignment_strategy, time_limit, removal_size, 
              cost_calculation_method, removal_operator, repair_operator, acceptance_function=acceptance_function, T_0=T_0, alpha=alpha, current_time=t if t is not None else 0,
              base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_unallocated_penalty=agent_unallocated_penalty, agent_task_sequence_limit=3,
              crm2m_lambda=crm2m_lambda, crm2m_detour_cutoff=crm2m_detour_cutoff)
    return lns.run(t=t)
