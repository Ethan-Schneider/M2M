import numpy as np
from .graphing import *

class Stats: 
    def __init__(self, num_robots : int, simulation_time : int) -> None:
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
        
        # Sequence of paths for each robot, init with empty array
        self.__truncated_paths = [] 
        self.__paths = []
        for _ in range(num_robots):
            self.__truncated_paths.append([])
            self.__paths.append([])
            
            
        self.__T = simulation_time
        
        self.__completed_task_ids = []
        self.__completed_to_pickup_task_ids = []
        
        
        # Runtime Stastics: 
        self.__total_runtime = []
        self.__CRG_time = []
        self.__TA_time = []
        self.__PF_time = []
        self.__SIM_time = []
        
        #TODO: SoC (Sum(self.__actual_duration))
        #TODO: Throughput ((len(actual_duration) / T)*60)
        
        
    def add_estimated_duration(self, task_id : int, estimated_duration : float) -> None:
        self.__estimated_duration[task_id] = estimated_duration
        
    def add_estimated_pickup_duration(self, task_id : int, estimated_duration : float) -> None:
        self.__estimated_pickup_duration[task_id] = estimated_duration
        
    def add_actual_duration(self, task_id : int) -> None:
        self.__actual_duration[task_id] = 0    
    
    def update_actual_duration(self, task_id : int, actual_duration : float) -> None:
        self.__actual_duration[task_id] += actual_duration
        
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
                
    def return_full_paths(self) -> list:
        return self.__paths
    
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
        self.__TA_time.append(time)
        
    def add_total_PF_time(self, time : float) -> None:
        self.__PF_time.append(time)
        
    def add_total_SIM_time(self, time : float) -> None:
        self.__SIM_time.append(time)
            
    
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
        
    def output_graphs(self) -> None:
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
        
        actual_estimated_duration(list(self.__actual_duration.values()), list(self.__estimated_duration.values()))
        actual_estimated_to_pickup_duration(list(self.__actual_pickup_duration.values()), list(self.__estimated_pickup_duration.values()))
        
        runtime_over_time(self.__CRG_time, self.__TA_time, self.__PF_time, self.__SIM_time)
        runtime_pie_chart(self.__CRG_time, self.__TA_time, self.__PF_time, self.__total_runtime)
        idle_robots_over_timesteps(self.__paths)
        