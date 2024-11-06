import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

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

def actual_estimated_duration(actual_duration : list, estimate_duration : list) -> None:
    
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
    
    N = len(actual_duration)
    p = actual_duration.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    X_with_intercept = np.empty(shape=(N, p))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = actual_duration


    ols = sm.OLS(estimate_duration, X_with_intercept)
    ols_result = ols.fit()
    results_summary = ols_result.summary()

    results_as_html = results_summary.tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    linear_regression = df['coef'].to_list()
    lower_std = df['[0.025'].to_list()
    upper_std = df['0.975]'].to_list()

    plt.scatter(actual_duration, estimate_duration)
    plt.xlabel("Actual Duration (s)")
    plt.ylabel("Estimated Duration (s)")
    plt.title("Actual Duration(s) vs. Estimated Duration(s) with 2 X Standard Deviation")
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
    
    plt.savefig("data/figures/task_duration_graph.png", bbox_inches='tight', dpi=300)
    plt.clf()
    
def actual_estimated_to_pickup_duration(actual_duration : list, estimate_duration : list) -> None:
    actual_duration = np.asarray(actual_duration).reshape(-1, 1)
    estimate_duration = np.asarray(estimate_duration)
    
    N = len(actual_duration)
    p = actual_duration.shape[1] + 1  # plus one because LinearRegression adds an intercept term

    X_with_intercept = np.empty(shape=(N, p))
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = actual_duration

    ols = sm.OLS(estimate_duration, X_with_intercept)
    ols_result = ols.fit()
    results_summary = ols_result.summary()

    results_as_html = results_summary.tables[1].as_html()
    df = pd.read_html(results_as_html, header=0, index_col=0)[0]

    linear_regression = df['coef'].to_list()
    lower_std = df['[0.025'].to_list()
    upper_std = df['0.975]'].to_list()

    plt.scatter(actual_duration, estimate_duration)
    plt.xlabel("Actual Duration (s)")
    plt.ylabel("Estimated Duration (s)")
    plt.title("Actual Duration (s) vs. Estimated Duration (s) To-Pickup with 2 X Standard Deviation")
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
    
    plt.savefig("data/figures/to_pickup_duration_graph.png", bbox_inches='tight', dpi=300)
    plt.clf()
    
#endregion

def runtime_over_time(CRG_runtime : list, TA_runtime : list, PF_runtime : list, SIM_runtime : list) -> None:
    plt.plot(np.arange(len(CRG_runtime)), CRG_runtime, '-', label="Task Generation")
    plt.plot(np.arange(len(TA_runtime)), TA_runtime, '-', label="Task Allocation")
    plt.plot(np.arange(len(PF_runtime)), PF_runtime, '-', label="Path Planning")
    
    # Removed due to being a flat line at the bottom
    # plt.plot(np.arange(len(SIM_runtime)), SIM_runtime, '-.', label="Simulation")
    
    plt.legend()
    plt.xlabel("Simulation Step")
    plt.ylabel("Computation Time (s)")
    plt.title("Computation Time over Simulation Steps")
    
    plt.savefig("data/figures/computation_time_during_simulation_graph", bbox_inches='tight', dpi=300)
    plt.clf()
    

def runtime_pie_chart(CRG_runtime : list, TA_runtime : list, PF_runtime : list, total_runtime : float) -> None:
    other_runtime = total_runtime - np.sum(CRG_runtime) - np.sum(TA_runtime) - np.sum(PF_runtime)
    
    data = np.asarray([np.sum(CRG_runtime), np.sum(TA_runtime), np.sum(PF_runtime), other_runtime])
    labels = ["Task Generation", "Task Assignment", "Path Planning", "Other"]
    
    plt.pie(data, labels=labels, startangle=90)
    plt.savefig("data/figures/total_runtime_pie_chart", bbox_inches='tight', dpi=300)
    plt.clf()
    
    
def idle_robots_over_timesteps(paths : list) -> None:
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
    plt.ylabel("Number of Idle Robots")
    plt.title("Number of Idle Robots Over Time")
    
    plt.savefig("data/figures/number_of_idle_robots_over_time", bbox_inches='tight', dpi=300)
    plt.clf()
    
    
# TODO: Generate graphs (same as the distance ones) for the duration of a task