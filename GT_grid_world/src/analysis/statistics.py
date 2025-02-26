import numpy as np
import json

from ..utils import *
from .graphing import *
from ..agent import *

class Stats: 
    def __init__(self, num_robots : int, simulation_time : int, output_file : str) -> None:
        self.__output_file = output_file
        self.__num_of_robots = num_robots
        self.__T = simulation_time
        
        # Temp SOC
        self.__soc = 0

        # Admissibility
        self.__admisibility = []
        self.__admisibility_differences = []

        # Aisle and Driveway Occupancy
        self.__aisle_occupancy = []
        self.__driveway_occupancy = []
        
        # Number of collisions at each timestep
        self.__collisions = []
        
        self.__throughputs = []
        
        self.__task_assignments = []
        
        self.__unallocated_agents = []
        
        # Estimated Distance for Task (task_start -> task_goal) in the form (task_id, estimated_distance)
        self.__estimated_distance = {}
        
        # Actual Distance for Task (task_start -> task_goal) in the form (task_id, actual_distance)
        self.__actual_distance = {}
        
        # Estimated duration for Task (interpolate time for (task_start->task_goal)) in the form (task_id, estimated_duration)
        self.__estimated_duration = {}
        
        # Actual duration for Task (time @ task_goal - time @ task_start) in the form (task_id, actual_duration)
        self.__actual_duration = {}
        
        # # Estimated distance for (robot_start_location -> task_start_location) in the form (task_id, estimated_distance)
        # self.__estimated_inbound_pickup_distance = []
        
        # # Actual distance for (robot_start_location -> task_start_location) in the form (task_id, actual_distance)
        # self.__actual_inbound_pickup_distance = []
        
        # # Estimated distance for (robot_start_location -> task_start_location) in the form (task_id, estimated_distance)
        # self.__estimated_outbound_pickup_distance = []
        
        # # Actual distance for (robot_start_location -> task_start_location) in the form (task_id, actual_distance)
        # self.__actual_outbound_pickup_distance = []
        
        # Estimated distance for (robot_start_location -> task_start_location) in the form (task_id, estimated_distance)
        self.__estimated_pickup_distance = {}
        
        # Actual distance for (robot_start_location -> task_start_location) in the form (task_id, actual_distance)
        self.__actual_pickup_distance = {}
        
        self.__estimated_pickup_duration = {}
        
        self.__actual_pickup_duration = {}

        self.__opened_nodes = []
        self.__expanded_nodes = []
        
        # Sequence of paths for each robot, init with empty array
        self.__truncated_paths = [] 
        self.__paths = []
        for _ in range(num_robots):
            self.__truncated_paths.append([])
            self.__paths.append([])
            self.__task_assignments.append([])
            
            
        self.__T = simulation_time
        
        self.__completed_task_ids = []
        self.__completed_to_pickup_task_ids = []
        
        
        # Runtime Stastics: 
        self.__total_runtime = []
        self.__CRG_time = []
        self.__TA_time = []
        self.__PF_time = []
        self.__SIM_time = []
        
        self.__num_path_plan_fails = 0
        
        #TODO: SoC (Sum(self.__actual_duration))
        #TODO: Throughput ((len(actual_duration) / T)*60)
        
    def compute_unallocated_agents(self, Rs : AgentLoader):
        num = 0
        for agent in Rs.agents:
            if not agent.task_sequence:
                num += 1
                
        self.__unallocated_agents.append(num)

    def append_open_nodes(self, opened_nodes : int) -> None:
        self.__opened_nodes.append(opened_nodes)

    def append_expanded_nodes(self, expanded_nodes : int) -> None:
        self.__expanded_nodes.append(expanded_nodes)
        
    def append_number_of_collisions(self, number_of_collisions : int) -> None:
        self.__collisions.append(number_of_collisions)
        
    def add_estimated_duration(self, task_id : int, estimated_duration : float) -> None:
        self.__estimated_duration[task_id] = estimated_duration
        
    def add_estimated_pickup_duration(self, task_id : int, estimated_duration : float) -> None:
        self.__estimated_pickup_duration[task_id] = estimated_duration
        
    def add_actual_duration(self, task_id : int) -> None:
        self.__actual_duration[task_id] = 0    
    
    def update_actual_duration(self, task_id : int, actual_duration : float) -> None:
        self.__actual_duration[task_id] += actual_duration
        
    def update_num_path_plan_fail(self) -> None:
        self.__num_path_plan_fails += 1
        
    def remove_actual_duration(self, task_id : int) -> None:
        del self.__actual_duration[task_id]
        
    def add_actual_pickup_duration(self, task_id : int) -> None:
        self.__actual_pickup_duration[task_id] = 0    
    
    def update_actual_pickup_duration(self, task_id : int, actual_duration : float) -> None:
        self.__actual_pickup_duration[task_id] += actual_duration
        
    def remove_actual_pickup_duration(self, task_id : int) -> None:
        del self.__actual_pickup_duration[task_id]
        
    # def add_estimated_inbound_pickup_distance(self, estimated_distance : float) -> None:
    #     self.__estimated_inbound_pickup_distance.append(estimated_distance)
        
    # def add_actual_inbound_pickup_distance(self, actual_distance : float) -> None:
    #     self.__actual_inbound_pickup_distance.append(actual_distance)
        
    # def add_estimated_outbound_pickup_distance(self, estimated_distance : float) -> None:
    #     self.__estimated_outbound_pickup_distance.append(estimated_distance)
    
    # def add_actual_oubound_pickup_distance(self, actual_distance : float) -> None:
    #     self.__actual_outbound_pickup_distance.append(actual_distance)
    
    # ====================== Estimated Task Distance Function
    
    def add_estimated_distance(self, task_id : int, estimated_distance : float) -> None:
        self.__estimated_distance[task_id] = estimated_distance
 
    # ====================== Estimated Pickup Distance Function
    
    def add_estimated_pickup_distance(self, task_id : int, distance : float) -> None:
        self.__estimated_pickup_distance[task_id] = distance
    
    # ====================== Actual Task Distance Functions
        
    def add_actual_distance(self, task_id : int) -> None:
        self.__actual_distance[task_id] = 0
        
    def update_actual_distance(self, task_id : int, added_distance : float) -> None:
        self.__actual_distance[task_id] += added_distance
        
    def remove_actual_distance(self, task_id : int) -> None:
        del self.__actual_distance[task_id]
    
    # ====================== Actual Pickup Distance Functions
    def add_actual_pickup_distance(self, task_id : int) -> None:
        self.__actual_pickup_distance[task_id] = 0
        
    def update_actual_pickup_distance(self, task_id : int, added_distance : float) -> None:
        self.__actual_pickup_distance[task_id] += added_distance
        
    def remove_actual_pickup_distance(self, task_id : int) -> None:
        del self.__actual_pickup_distance[task_id]
    
    # ====================== Paths Functions
    
    def add_paths(self, steps : list) -> None:
        for i, step in enumerate(steps):
            if self.__truncated_paths[i] == []:
                self.__truncated_paths[i].append(step)
                self.__paths[i].append(step)
                continue
            self.__paths[i].append(step)
            if self.__truncated_paths[i][-1] != step:
                self.__truncated_paths[i].append(step)
                
    def append_task_allocation(self, Rs : AgentLoader, J) -> None:
        for agent in Rs.agents:
            for task in agent.task_sequence:
                # assignment = (get_task_start_location(J, task), get_task_goal_location(J, task), agent.state)
                self.__task_assignments[agent.id].append(task)
                
    def return_full_paths(self) -> list:
        return self.__paths
    
    def set_soc(self, soc) -> None:
        self.__soc += soc
    
    # ====================== Aisle and Driveway Occupancy Functions
    def add_aisle_occupancy(self, aisle_occupancy : list) -> None:
        """Add occupancy of the aisles to the statistics object per timestep.

        Args:
            aisle_occupancy (list): Occupancy of the aisles per timestep.
        """
        self.__aisle_occupancy.append(aisle_occupancy)
    
    def add_driveway_occupancy(self, driveway_occupancy : list) -> None:
        """Add occupancy of the driveways to the statistics object per timestep.

        Args:
            driveway_occupancy (list): Occupancy of the driveways per timestep.
        """
        self.__driveway_occupancy.append(driveway_occupancy)
    
    # ====================== Completed Task Id Functions
    
    def add_completed_task_id(self, task_id : int) -> None:
        self.__completed_task_ids.append(task_id)
        
    def get_completed_task_ids(self) -> list:
        return self.__completed_task_ids
            
    # ====================== Completed To-Pickup Task Id Functions
    
    def add_completed_to_pickup_task_id(self, task_id : int) -> None:
        self.__completed_to_pickup_task_ids.append(task_id)
        
    # ====================== Runtime Functions
    
    def set_total_runtime(self, time : float) -> None:
        self.__total_runtime = time
        
    def add_total_CRG_time(self, time : float) -> None:
        self.__CRG_time.append(time)
        
    def add_total_TA_time(self, time : float) -> None:
        self.__TA_time.append(float(time))
        
    def add_total_PF_time(self, time : float) -> None:
        self.__PF_time.append(float(time))
        
    def add_total_SIM_time(self, time : float) -> None:
        self.__SIM_time.append(time)
            

    def add_admisibility(self, admisibility : list) -> None:
        for b in admisibility:
            self.__admisibility.append(b)
    
    def append_admisibility_differences(self, admisibility : list) -> None:
        for b in admisibility:
            self.__admisibility_differences.append(b)
    # ====================== Utils
    def trim_data(self):
        # Trim Actual Distance 
        for task_id in list(self.__actual_distance.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__actual_distance[task_id]
        
        # Trim Estimated Distance 
        for task_id in list(self.__estimated_distance.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__estimated_distance[task_id]
                
        # Trim Actual Pickup Distance
        for task_id in list(self.__actual_pickup_distance.copy().keys()):
            if task_id not in self.__completed_to_pickup_task_ids:
                del self.__actual_pickup_distance[task_id]
                
        # Trim Estimated Pickup Distance
        for task_id in list(self.__estimated_pickup_distance.copy().keys()):
            if task_id not in self.__completed_to_pickup_task_ids:
                del self.__estimated_pickup_distance[task_id]
                
        # Trim Actual Duration
        for task_id in list(self.__actual_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__actual_duration[task_id]    
            
        # Trim Estimated Duration
        for task_id in list(self.__estimated_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__estimated_duration[task_id]
                
        # Trim Actual Pickup Duration
        for task_id in list(self.__actual_pickup_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__actual_pickup_duration[task_id]
                
        # Trim Estimated Pickup Duration
        for task_id in list(self.__estimated_pickup_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__estimated_pickup_duration[task_id]
            
    def return_throughput(self):
        return (len(self.__completed_task_ids)/self.__T)*60
            
    def return_sum_of_costs(self):
        return np.sum(list(self.__actual_distance.values()))/60

    def compute_stationary_robots(self) -> list:
        num_idle_robots = [len(self.__paths)]
        
        # Create numpy array of tuples
        paths = np.asarray(self.__paths, dtype="f,f")
        
        # Save first state
        prev_state = paths.T[0]
        # Iterate over the state of the system at each timestep, checking how many robots are in the same position as the previous state
        for state in paths.T[1:]:
            num_idle_robots.append(np.count_nonzero(prev_state == state))
            prev_state = state
        return num_idle_robots
    
    def compute_velocity_timesteps(self) -> list:
        velocity_timesteps = []
        
        paths = np.asarray(self.__paths, dtype="f,f")
        # Save first state
        prev_state = paths.T[0]
        # Iterate over the state of the system at each timestep, checking how many robots are in the same position as the previous state
        for state in paths.T[1:]:
            velocity = []
            for robot_id in range(len(state)):
                if prev_state[robot_id] == state[robot_id]:
                    velocity.append(0)
                else:
                    velocity.append(1)
            velocity_timesteps.append(velocity)
            prev_state = state  
        return velocity_timesteps
        
    # ====================== Statistical Analysis
    
    def task_length(self) -> list:
        average_task_length = np.average(list(self.__actual_distance.values()))
        average_distance_to_task = np.average(list(self.__actual_pickup_distance.values()))
        average_estimated_task_length = np.average(list(self.__estimated_distance.values()))
        average_estimated_distance_to_task = np.average(list(self.__estimated_pickup_distance.values()))
        
        print("Average Actual Task Distance: ", average_task_length)
        print("Average Actual Distance to Task: ", average_distance_to_task)
        print("Average Estimated Task Distance: ", average_estimated_task_length)
        print("Average Estimated Distance To Task: ", average_estimated_distance_to_task)
            
            
    def print_statistics(self) -> None:
        print("===Runtime Statistics===")
        print("Total Runtime: " + str(self.__total_runtime))
        print("Total Task Generation Time: " + str(np.sum(self.__CRG_time)))
        print("Total Task Allocation Time: " + str(np.sum(self.__TA_time)))
        print("Total Path Planning Time: " + str(np.sum(self.__PF_time)))
        print("Total Simulation Time: " + str(np.sum(self.__SIM_time)))
        
        
        print("===Number of Completed Tasks===")
        print(self.__completed_task_ids)
        
        
        print("===Robot Paths===")
        # TODO: Remove locations in path where robot stays in the same spot for more than one timestep
        for i, robot_path in enumerate(self.__truncated_paths):
            print("Robot " + str(i) + " moved " + str(len(robot_path)) + " spaces during the simulation.")
            
        total = 0
        for robot_path in self.__truncated_paths:
            prev_state = robot_path[0]
            for state in robot_path[1:]:
                if prev_state != state:
                    total += 1
                else:
                    pass
                prev_state = state
        print("Total: ", total)
            
        print("===Estimated Task Distance===")
        
        total = 0
        for distance in list(self.__estimated_distance.values()):
            total += distance
            print(distance)
        print("Estimated Task Distance Total: ", total)
           
           
        print("===Estimated To-Pickup Distance===")
            
        total = 0
        for distance in list(self.__estimated_pickup_distance.values()):
            total += distance
            print(distance)
        print("Estimated Distance To-Pickup Total: ", total)
        
            
        print("===Actual Task Distance===")
        
        total = 0
        for distance in list(self.__actual_distance.values()):
            total += distance
            print(distance)
        print("Actual Task Distance Total: ", total)
            
        total = 0
        print("===Actual To-Pickup Distance===")
            
        for distance in list(self.__actual_pickup_distance.values()):
            total += distance
            print(distance)
        print("Actual Distance To-Pickup Total: ", total)
        
        
        print("===Throughout===")
        print("Throughput: " + str((len(self.__completed_task_ids)/self.__T)*60) + " tasks/min")
        
        print("===Sum of Costs===")
        print("Sum of Costs: " + str(np.sum(list(self.__actual_distance.values()))) + "s or " + str(np.sum(list(self.__actual_distance.values()))/60) + "min.")
        
    def output_graphs(self, folder : str = "") -> None:
        # actual_estimated_distance(list(self.__actual_distance.values()), list(self.__estimated_distance.values()))
        # actual_estimated_to_pickup_distance(list(self.__actual_pickup_distance.values()), list(self.__estimated_pickup_distance.values()))
        
        # total_actual_distance = []
        # total_estimate_distance = []
        # actual_task_distance = list(self.__actual_distance.values())
        # actual_to_picktup_distance = list(self.__actual_pickup_distance.values())
        # estimate_task_distance = list(self.__estimated_distance.values())
        # estimate_to_pickup_distance = list(self.__estimated_pickup_distance.values())
        # for i in range(len(list(self.__actual_distance))):
        #     total_actual_distance.append(actual_task_distance[i]+actual_to_picktup_distance[i])
        #     total_estimate_distance.append(estimate_task_distance[i]+estimate_to_pickup_distance[i])
            
        # actual_estimated_total_distance(total_actual_distance, total_estimate_distance)
        
        # Duration Graphs
        print(len(list(self.__actual_duration.values())))
        print(len(list(self.__estimated_duration.values())))
        actual_estimated_duration(list(self.__actual_duration.values()), list(self.__estimated_duration.values()), subfolder=folder)
        actual_estimated_to_pickup_duration(list(self.__actual_pickup_duration.values()), list(self.__estimated_pickup_duration.values()), subfolder=folder)
        
        total_actual_duration = []
        total_estimate_duration = []
        actual_task_duration = list(self.__actual_duration.values())
        actual_to_pickup_duration = list(self.__actual_pickup_duration.values())
        estimate_task_duration = list(self.__estimated_duration.values())
        estimate_to_pickup_duration = list(self.__estimated_pickup_duration.values())
        for i in range(len(list(self.__actual_duration))):
            total_actual_duration.append(actual_task_duration[i]+actual_to_pickup_duration[i])
            total_estimate_duration.append(estimate_task_duration[i]+estimate_to_pickup_duration[i])
        actual_estimated_total_duration(total_actual_duration, total_estimate_duration, subfolder=folder)
        
        # Runtime Graphs
        runtime_over_time(self.__CRG_time, self.__TA_time, self.__PF_time, self.__SIM_time)
        runtime_pie_chart(self.__CRG_time, self.__TA_time, self.__PF_time, self.__total_runtime)
        
        # Idle Time Graph
        stationary_robots_over_timesteps(self.__paths, subfolder=folder)
        
        # Unallocated Agents Graph
        unallocated_agents_over_timesteps(self.__unallocated_agents, subfolder=folder)
        
        #Aisle and Driveway Occupancy Graph
        aisle_occupancy_over_timesteps(self.__aisle_occupancy, subfolder=folder)
        driveway_occupancy_over_timesteps(self.__driveway_occupancy, subfolder=folder)
        
        
    def save_data(self):
        velocity_timesteps = self.compute_velocity_timesteps()
        # print(self.__task_assignments)
        
        # timestep_data = {}
        # for i in range(len(self.__paths)):
        #     timestep_data[f'timestep_{i}'] = {"paths" : self.__paths[i], "allocation" : self.__task_assignments[i], "velocity_timesteps" : velocity_timesteps[i]}
        
        
        
        data = {
            "timesteps_completed" : self.__T,
            "number_of_robots" : self.__num_of_robots,
            "total_completed_tasks" : int(len(self.__completed_task_ids)),
            "completed_tasks": self.__completed_task_ids,
            "actual_duration_of_task_from_pick_to_place" : list(self.__actual_duration.values()),
            "estimated_duration_of_task_from_pick_to_place" : list(self.__estimated_duration.values()),
            "actual_duration_of_task_from_start_to_pick" : list(self.__actual_pickup_duration.values()),
            "estimated_duration_of_task_from_start_to_pick" : list(self.__estimated_pickup_duration.values()),
            "collisions" : int(np.sum(self.__collisions)),
            "stationary_robots" : self.compute_stationary_robots(),
            "unallocated agents" : self.__unallocated_agents,
            "throughput (tasks/min)": self.return_throughput(),
            "SoC(min)" : self.__soc,
            "Total Runtime" : self.__total_runtime,
            "Total Path Planning Runtime" : int(np.sum(self.__PF_time)),
            "Opened Nodes" : self.__opened_nodes,
            "Expanded Nodes" : self.__expanded_nodes,
            "Path Planning Runtimes" : self.__PF_time,
            "Total Task Allocaiton Runtime" : int(np.sum(self.__TA_time)),
            "Number of Path Plan Fails" : int(self.__num_path_plan_fails),
            "Task Allocation Runtime" : self.__TA_time,
            "admissibility" : self.__admisibility,
            "admissibility_differences" : self.__admisibility_differences,
            "paths" : self.__paths,
            "aisle_occupancy" : self.__aisle_occupancy,
            "driveway_occupancy" : self.__driveway_occupancy,
            "allocation" : self.__task_assignments,
            "velocity_timesteps" : velocity_timesteps
        }
        
        
        with open(self.__output_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)