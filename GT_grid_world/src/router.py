from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs
# from src.path_finding_algorithms.external_algorithms.MAPF_PC import mgpbs

from .agent import *
from .utils import *

def pathPlan(G, map : str, Rs : AgentLoader, J : set, path_planning_strategy : str) -> list:
    if path_planning_strategy == "mgpbs":
        states = [robot[1] for robot in Rs]

        goal_locations = []
        for robot in Rs:
            robot_id = robot[0]
            
            # If robot is going to pickup, set goal location to the task's start location
            if robot_id in to_pickup:
                # Get assigned task_id
                task_id = -1
                for assignment in Ra:
                    if assignment[1] == robot_id:
                        task_id = assignment[0]
                # Get assigned task's start location
                start_loc = (-1, -1)
                for task in J:
                    if task[0] == task_id:
                        start_loc = task[1]
                goal_locations.append([(robot_id, start_loc)])
                
            # If robot is going to delivery, set goal location to the task's goal location
            elif robot_id in to_delivery:
                # Get assigned task_id
                task_id = -1
                for assignment in Ra:
                    if assignment[1] == robot_id:
                        task_id = assignment[0]
                        
                # Get assigned task's goal location
                goal_loc = (-1, -1)
                for task in J:
                    if task[0] == task_id:
                        goal_loc = task[2]
                goal_locations.append([(robot_id, goal_loc)])
                
            # If robot is a free_agent, set goal location to current state
            else:
                goal_locations.append([(robot_id, robot[1])])
                
        goal_locations.sort()
        goal_locations = [x[0][1] for x in goal_locations]
        sequences = mgpbs.test_cpp_func(map, len(Rs), 60, 1.2, states, goal_locations)
        print("+++++++++++++++++++Robot Sequence is: ", sequences)
        if len(sequences) < 2:
            print("++++++++++++++++++++++++++++++++Skipping Routing++++++++++++++++++++++++++++++++")
            return previous_sequence
        # Remove first item in sequences, as they are the robot's current location
        for sequence in sequences:
            sequence.pop(0)
        
        return sequences
    
    else: 
        states = [agent.state for agent in Rs.agents]

        goal_locations = []
        for agent in Rs.agents:
            # If robot is going to pickup, set goal location to the task's start location
            if agent.status == 1:
                # Get current assigned task's start location
                goal_locations.append(get_task_start_location(J, agent.task_sequence[0]))
                
            # If robot is going to delivery, set goal location to the task's goal location
            elif agent.status == 2:
                goal_locations.append(get_task_goal_location(J, agent.task_sequence[0]))
                
            # If robot is a free_agent, set goal location to current state
            else:
                goal_locations.append(agent.state)
        # print("States: ", states)
        # print("Goal Locations: ", goal_locations)
        # for agent in Rs.agents:
        #     print("Agent Status: ", agent.status)
        #     print("Agent Id: ", agent.id)
        #     if agent.task_sequence:
        #         print("Agent's Task Start Location: ", get_task_start_location(J, agent.task_sequence[0]))
        #         print("Agent's Task Goal Location: ", get_task_goal_location(J, agent.task_sequence[0]))

        sequences = []
        w = 1.2
        while not sequences:
            # Execute the path planning algorithm
            if path_planning_strategy == "pbs":
                sequences = pbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
            elif path_planning_strategy == "ecbs":
                sequences = eecbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
            if sequences == []:
                print("+++++++++++++++++++Execution Failed with w = ", w)
                print(sequences)
            # else:
            #     print("Execution Succeeded with Sequences:", sequences)
                
            # If a solution cannot be found with a higher suboptimality bound, break
            if w >= 12.0:
                break
            w += 5.0
        
        # If no sequences are returned, return the previous sequence
        # TODO: If no solution is found, recall the path planning algorithm with a higher suboptimality bound
        if not sequences:
            return Rs
        
        # Remove first item in sequences, as they are the robot's current location
        for i, agent in enumerate(Rs.agents):
            temp_sequence = sequences[i][1:]
            agent.path_sequence = temp_sequence
        
        return Rs