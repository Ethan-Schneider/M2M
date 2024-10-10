def simulate(Rs : list, robot_sequences : list, Ra : list, J : set, to_delivery : set, to_pickup : set, free_agents : set, robot_path_sequences : list):

    # Update state of robots
    for robot in Rs:
        # If robot sequence is stationary, leave the robot in place (wait action)
        if len(robot_sequences[robot[0]]) == 0:
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            Rs[robot[0]] = (robot[0], robot_sequences[robot[0]].pop(0))

    # Update free_agents, to_pickup, and to_delivery

    robot_ids = []
    # Update to_pickup -> to_delivery
    for robot in to_pickup:
        task_id = -1
        # Find task_id assignment
        for assignment in Ra:
            if assignment[1] == robot:
                task_id = assignment[0]
                break
        start_loc = (-1, -1)
        # Find start location
        for task in J:
            if task_id == task[0]:
                start_loc = task[1]
                break
        # Find robot.state
        robot_state = Rs[robot][1]
        # Check if robot.state == start_location, if so remove robot from to_pickup and add to_delivery
        if robot_state == start_loc:
            robot_ids.append(robot)
            
    for id in robot_ids:
        to_pickup.remove(id)
        to_delivery.add(id)  
            
            
    robot_ids = []
    # Update to_delivery -> free_agents
    for robot in to_delivery:
        task_id = -1
        #Find task_id assignment
        current_allocation = -1
        for assignment in Ra:
            if assignment[1] == robot:
                current_allocation = assignment
                task_id = assignment[0]
                break
        goal_loc = (-1, -1)
        curent_task = (-1, -1)
        # Find goal location
        for task in J:
            if task_id == task[0]:
                current_task = task
                goal_loc = task[2]
                break
        # Find robot.state
        robot_state = Rs[robot][1]
        # Check if robot.state == goal_location, if so remove robot from to_delivery and add to free_agents, then remove task from J
        if robot_state == goal_loc:
            robot_ids.append(robot)
            J.remove(current_task)
            Ra.remove(current_allocation)
            
    for id in robot_ids:
        to_delivery.remove(id)
        free_agents.add(id)
            
    # Update robot_path_sequences
    for i, robot in enumerate(Rs):
        robot_path_sequences[i].append(robot[1])
            
    return Rs, Ra, J, to_pickup, to_delivery, free_agents, robot_path_sequences