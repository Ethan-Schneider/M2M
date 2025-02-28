import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import os

#region Distance Graphs
def actual_estimated_distance(actual_distance : list, estimate_distance : list) -> None:
    actual_distance = np.asarray(actual_distance).reshape(-1, 1)
    estimate_distance = np.asarray(estimate_distance)
    
    N = len(actual_distance)
    p = actual_distance.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    X_with_intercept = np.empty(shape=(N, p))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = actual_distance

    ols = sm.OLS(estimate_distance, X_with_intercept)
    ols_result = ols.fit()
    results_summary = ols_result.summary()

    results_as_html = results_summary.tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    linear_regression = df['coef'].to_list()
    lower_std = df['[0.025'].to_list()
    upper_std = df['0.975]'].to_list()

    plt.scatter(actual_distance, estimate_distance)
    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Estimated Distance (m)")
    plt.title("Actual Distance(m) vs. Estimated Distance(m) with 2 X Standard Deviation")
    plt.xlim((0, 360))
    plt.ylim((0, 360))


    x = np.linspace(0, 350, 100)
    y = linear_regression[1]*x + linear_regression[0]
    plt.plot(x, y, 'k')


    y = lower_std[1]*x + lower_std[0]
    plt.plot(x, y, '--r')

    y = upper_std[1]*x + upper_std[0]
    plt.plot(x, y, '--r')

    ols_result.summary()
    
    plt.savefig("task_output_graph.png", bbox_inches='tight', dpi=300)
    plt.clf()
    
    
    
def actual_estimated_to_pickup_distance(actual_distance : list, estimate_distance : list) -> None:
    actual_distance = np.asarray(actual_distance).reshape(-1, 1)
    estimate_distance = np.asarray(estimate_distance)
    
    N = len(actual_distance)
    p = actual_distance.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    X_with_intercept = np.empty(shape=(N, p))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = actual_distance

    ols = sm.OLS(estimate_distance, X_with_intercept)
    ols_result = ols.fit()
    results_summary = ols_result.summary()

    results_as_html = results_summary.tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    linear_regression = df['coef'].to_list()
    lower_std = df['[0.025'].to_list()
    upper_std = df['0.975]'].to_list()

    plt.scatter(actual_distance, estimate_distance)
    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Estimated Distance (m)")
    plt.title("Actual Distance(m) vs. Estimated Distance(m) To-Pickup with 2 X Standard Deviation")
    plt.xlim((0, 360))
    plt.ylim((0, 360))


    x = np.linspace(0, 350, 100)
    y = linear_regression[1]*x + linear_regression[0]
    plt.plot(x, y, 'k')


    y = lower_std[1]*x + lower_std[0]
    plt.plot(x, y, '--r')

    y = upper_std[1]*x + upper_std[0]
    plt.plot(x, y, '--r')

    ols_result.summary()
    
    plt.savefig("to_pickup_output_graph.png", bbox_inches='tight', dpi=300)
    plt.clf()
    
    
def actual_estimated_total_distance(actual_distance : list, estimate_distance : list) -> None:
    actual_distance = np.asarray(actual_distance).reshape(-1, 1)
    estimate_distance = np.asarray(estimate_distance)
    
    N = len(actual_distance)
    p = actual_distance.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    X_with_intercept = np.empty(shape=(N, p))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = actual_distance

    ols = sm.OLS(estimate_distance, X_with_intercept)
    ols_result = ols.fit()
    results_summary = ols_result.summary()

    results_as_html = results_summary.tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    linear_regression = df['coef'].to_list()
    lower_std = df['[0.025'].to_list()
    upper_std = df['0.975]'].to_list()

    plt.scatter(actual_distance, estimate_distance)
    plt.xlabel("Actual Distance (m)")
    plt.ylabel("Estimated Distance (m)")
    plt.title("Actual Distance(m) vs. Estimated Distance(m) Total with 2 X Standard Deviation")
    plt.xlim((0, 360))
    plt.ylim((0, 360))


    x = np.linspace(0, 350, 100)
    y = linear_regression[1]*x + linear_regression[0]
    plt.plot(x, y, 'k')


    y = lower_std[1]*x + lower_std[0]
    plt.plot(x, y, '--r')

    y = upper_std[1]*x + upper_std[0]
    plt.plot(x, y, '--r')

    ols_result.summary()
    
    plt.savefig("total_task_distance_output_graph.png", bbox_inches='tight', dpi=300)
    plt.clf()
    
#endregion
    
#region Duration Graphs

def actual_estimated_duration(actual_duration : list, estimate_duration : list, subfolder : str = "") -> None:
    
    print("Actual Duration")
    print(len(actual_duration))
    for duration in actual_duration:
        print(duration)
        
    print("Estimated Duration")
    print(len(estimate_duration))
    for duration in estimate_duration:
        print(duration)
    
    actual_duration = np.asarray(actual_duration).reshape(-1, 1)
    estimate_duration = np.asarray(estimate_duration)
    
    # N = len(actual_duration)
    # p = actual_duration.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    # X_with_intercept = np.empty(shape=(N, p))
    # X_with_intercept[:, 0] = 1
    # X_with_intercept[:, 1:p] = actual_duration


    # ols = sm.OLS(estimate_duration, X_with_intercept)
    # ols_result = ols.fit()
    # results_summary = ols_result.summary()

    # results_as_html = results_summary.tables[1].as_html()
    # df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    # linear_regression = df['coef'].to_list()
    # lower_std = df['[0.025'].to_list()
    # upper_std = df['0.975]'].to_list()
    
    x = np.linspace(0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50, 100)
    y = x
    plt.plot(x, y, 'k', label="Perfectly Predicted Tasks")
    
    # y = x - 5
    # plt.plot(x, y, 'r--')
    
    # y = x + 5
    # plt.plot(x, y, 'r--')
    
    plt.scatter(actual_duration, estimate_duration, c='red', label='Completed Tasks')
    plt.xlabel("Actual Duration (s)")
    plt.ylabel("Estimated Duration (s)")
    plt.xlim((0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50))
    plt.ylim((0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50))
    plt.title("Estimated vs. Actual Task Duration from Pickup to Dropoff Location")
    plt.grid(True)

    # x = estimate_duration
    # y = actual_duration
    # plt.plot(x, y, 'k')


    # y = linear_regression[1]*x + linear_regression[0]
    # plt.plot(x, y, 'b', label="Linear Regression")

    # y = lower_std[1]*x + lower_std[0]
    # plt.plot(x, y, '--b')

    # y = upper_std[1]*x + upper_std[0]
    # plt.plot(x, y, '--b')

    # ols_result.summary()
    plt.legend(loc="upper left")
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "task_duration_graph.png"), bbox_inches='tight', dpi=300)
    plt.clf()
    
def actual_estimated_to_pickup_duration(actual_duration : list, estimate_duration : list, subfolder : str = "") -> None:
    actual_duration = np.asarray(actual_duration).reshape(-1, 1)
    estimate_duration = np.asarray(estimate_duration)
    
    # N = len(actual_duration)
    # p = actual_duration.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    # X_with_intercept = np.empty(shape=(N, p))
    # X_with_intercept[:, 0] = 1
    # X_with_intercept[:, 1:p] = actual_duration

    # ols = sm.OLS(estimate_duration, X_with_intercept)
    # ols_result = ols.fit()
    # results_summary = ols_result.summary()

    # results_as_html = results_summary.tables[1].as_html()
    # df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    # linear_regression = df['coef'].to_list()
    # lower_std = df['[0.025'].to_list()
    # upper_std = df['0.975]'].to_list()
    
    print(actual_duration)
    print(estimate_duration)
    
    x = np.linspace(0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50, 100)
    y = x
    plt.plot(x, y, 'k', label="Perfectly Predicted Tasks")
    
    # y = x - 5
    # plt.plot(x, y, 'r--')
    
    # y = x + 5
    # plt.plot(x, y, 'r--')

    plt.scatter(actual_duration, estimate_duration, c='red', label='Completed Tasks')
    plt.xlabel("Actual Duration (s)")
    plt.ylabel("Estimated Duration (s)")
    plt.xlim((0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50))
    plt.ylim((0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50))
    plt.grid(True)

    plt.title("Estimated vs. Actual Task Duration from Agent Start Location to Pickup Location")
    # x = estimate_duration
    # y = actual_duration
    # plt.plot(x, y, 'k')


    # y = linear_regression[1]*x + linear_regression[0]
    # plt.plot(x, y, 'b', label="Linear Regression")

    # y = lower_std[1]*x + lower_std[0]
    # plt.plot(x, y, '--b')

    # y = upper_std[1]*x + upper_std[0]
    # plt.plot(x, y, '--b')

    # ols_result.summary()
    plt.legend(loc="upper left")
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    plt.savefig(os.path.join(folder_path, "to_pickup_duration_graph.png"), bbox_inches='tight', dpi=300)
    plt.clf()
    
def actual_estimated_total_duration(actual_duration : list, estimate_duration : list, subfolder : str = "") -> None:
    actual_duration = np.asarray(actual_duration).reshape(-1, 1)
    estimate_duration = np.asarray(estimate_duration)
    
    # N = len(actual_duration)
    # p = actual_duration.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    # X_with_intercept = np.empty(shape=(N, p))
    # X_with_intercept[:, 0] = 1
    # X_with_intercept[:, 1:p] = actual_duration

    # ols = sm.OLS(estimate_duration, X_with_intercept)
    # ols_result = ols.fit()
    # results_summary = ols_result.summary()

    # results_as_html = results_summary.tables[1].as_html()
    # df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    # linear_regression = df['coef'].to_list()
    # lower_std = df['[0.025'].to_list()
    # upper_std = df['0.975]'].to_list()
    
    x = np.linspace(0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50, 100)
    y = x
    plt.plot(x, y, 'k', label="Perfectly Predicted Tasks")
    
    # y = x - 5
    # plt.plot(x, y, 'r--')
    
    # y = x + 5
    # plt.plot(x, y, 'r--')

    plt.scatter(actual_duration, estimate_duration, c='red', label='Completed Tasks')
    plt.xlabel("Actual Duration (s)")
    plt.ylabel("Estimated Duration (s)")
    plt.xlim((0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50))
    plt.ylim((0, np.max([np.max(actual_duration), np.max(estimate_duration)]) + 50))
    plt.grid(True)
    
    plt.title("Estimated vs. Actual Task Duration")


    # x = estimate_duration
    # y = actual_duration
    # plt.plot(x, y, 'k')

    # y = linear_regression[1]*x + linear_regression[0]
    # plt.plot(x, y, 'b', label="Linear Regression")

    # y = lower_std[1]*x + lower_std[0]
    # plt.plot(x, y, '--b')

    # y = upper_std[1]*x + upper_std[0]
    # plt.plot(x, y, '--b')

    # ols_result.summary()
    plt.legend(loc="upper left")
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    plt.savefig(os.path.join(folder_path, "total_task_duration_graph.png"), bbox_inches='tight', dpi=300)
    plt.clf()
    
#endregion

#region Runtime Graphs
def runtime_over_time(CRG_runtime : list, TA_runtime : list, PF_runtime : list, SIM_runtime : list, subfolder : str = "") -> None:
    plt.plot(np.arange(len(CRG_runtime)), CRG_runtime, '-', label="Task Generation")
    plt.plot(np.arange(len(TA_runtime)), TA_runtime, '-', label="Task Allocation")
    plt.plot(np.arange(len(PF_runtime)), PF_runtime, '-', label="Path Planning")
    
    # Removed due to being a flat line at the bottom
    # plt.plot(np.arange(len(SIM_runtime)), SIM_runtime, '-.', label="Simulation")
    
    plt.legend()
    plt.xlabel("Simulation Step")
    plt.ylabel("Computation Time (s)")
    plt.title("Computation Time over Simulation Steps")
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "computation_time_during_simulation_graph"), bbox_inches='tight', dpi=300)
    plt.clf()
    

def runtime_pie_chart(CRG_runtime : list, TA_runtime : list, PF_runtime : list, total_runtime : float, subfolder : str = "") -> None:
    other_runtime = total_runtime - np.sum(CRG_runtime) - np.sum(TA_runtime) - np.sum(PF_runtime)
    
    data = np.asarray([np.sum(CRG_runtime), np.sum(TA_runtime), np.sum(PF_runtime), other_runtime])
    labels = ["Task Generation: " + str(np.round((np.sum(CRG_runtime)/total_runtime)*100, 1)) + '%', \
        "Task Assignment: " + str(np.round((np.sum(TA_runtime)/total_runtime)*100, 1)) + '%', 
        "Path Planning: " + str(np.round((np.sum(PF_runtime)/total_runtime)*100, 1)) + '%', 
        "Other: " + str(np.round((np.sum(other_runtime)/total_runtime)*100, 1)) + '%']
    
    plt.pie(data, labels=labels, startangle=90)
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "total_runtime_pie_chart"), bbox_inches='tight', dpi=300)
    plt.clf()
    
#endregion
    
#region Idle Graph

def stationary_robots_over_timesteps(paths : list, subfolder : str = "") -> None:
    num_idle_robots = [len(paths)]
    
    # Create numpy array of tuples
    paths = np.asarray(paths, dtype="f,f")
    
    # Save first state
    prev_state = paths.T[0]
    # Iterate over the state of the system at each timestep, checking how many robots are in the same position as the previous state
    for state in paths.T[1:]:
        num_idle_robots.append(np.count_nonzero(prev_state == state))
        prev_state = state
        
    plt.plot(np.arange(len(num_idle_robots)), num_idle_robots, '-')
    
    plt.xlabel("Time Step")
    plt.ylabel("Number of Stationary Robots")
    plt.title("Number of Stationary Robots Over Time")
    plt.yticks(np.arange(0, max(num_idle_robots)+1, 1))
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "number_of_stationary_robots_over_time"), bbox_inches='tight', dpi=300)
    plt.clf()
    
#endregion

#region Unallocated Agents Graph
def unallocated_agents_over_timesteps(unallocated_agents : list, subfolder : str = "") -> None:
    plt.plot(np.arange(len(unallocated_agents)), unallocated_agents, '-')

    plt.xlabel("Time Step")
    plt.ylabel("Number of Unallocated Robots")
    plt.title("Number of Unallocated Robots Over Time")
    plt.yticks(np.arange(0, max(unallocated_agents)+1, 1))
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "number_of_unallocated_robots_over_time"), bbox_inches='tight', dpi=300)
    plt.clf()
#endregion

#region Aisle and Driveway Occupancy Graphs
def aisle_occupancy_over_timesteps(aisle_occupancy : list, subfolder : str = "") -> None:
    plt.plot(np.arange(len(aisle_occupancy)), aisle_occupancy, '-')

    plt.xlabel("Time Step")
    plt.ylabel("Aisle Occupancy")
    plt.title("Aisle Occupancy Over Time")
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "aisle_occupancy_over_time"), bbox_inches='tight', dpi=300)
    plt.clf()
    
def driveway_occupancy_over_timesteps(driveway_occupancy : list, subfolder : str = "") -> None:
    plt.plot(np.arange(len(driveway_occupancy)), driveway_occupancy, '-')

    plt.xlabel("Time Step")
    plt.ylabel("Driveway Occupancy")
    plt.title("Driveway Occupancy Over Time")
    
    folder_path = "data/figures/" + subfolder + "/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(os.path.join(folder_path, "driveway_occupancy_over_time"), bbox_inches='tight', dpi=300)
    plt.clf()
#endregion
# TODO: Generate graphs (same as the distance ones) for the duration of a task