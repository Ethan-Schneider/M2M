from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs

from .agent import *
from .utils import *
from .analysis.statistics import *
from .simulate import STATUS_PICKING, STATUS_PLACING
from typing import Dict, Tuple

# Consecutive ticks an agent may sit blocked (its next cell held by another
# agent) before we escalate from "just wait" to a full global replan that can
# route it around the obstruction. Mirrors Ethan's task_queue default.
BLOCKED_REPLAN_THRESHOLD = 5


def needs_path_plan(Rs: AgentLoader, blocked_threshold: int = BLOCKED_REPLAN_THRESHOLD) -> bool:
    """Return True when the PBS router should replan this tick.

    Replan when some agent has no remaining plan (it just finished and needs its
    next route) or has been blocked in place for ``blocked_threshold`` ticks (a
    sustained stand-off the reactive move-guard in ``simulate`` cannot clear on
    its own -- e.g. two agents wanting to swap cells).
    """
    for agent in Rs.agents:
        if agent.path_sequence == []:
            return True
        if agent.blocked_ticks >= blocked_threshold:
            return True
    return False


def clear_all_paths(Rs: AgentLoader) -> None:
    """Drop every agent's plan and reset its block counter.

    Called when PBS cannot find any solution (common under heavy congestion).
    Agents then hold position (empty plan -> wait) instead of continuing to
    execute a stale, now-uncoordinated plan -- that stale-plan execution is what
    made box-carrying agents reverse in and out of aisles.
    """
    for agent in Rs.agents:
        agent.path_sequence = []
        agent.blocked_ticks = 0


def pathPlan(map : str, Rs : AgentLoader, path_planning_strategy : str, S : Stats, seed : int) -> AgentLoader:
    states = [agent.state for agent in Rs.agents]

    goal_locations = []
    for agent in Rs.agents:
        # If agent is blocked, set goal location to its current location
        if agent.blocked_ticks >= BLOCKED_REPLAN_THRESHOLD:
            goal_locations.append(agent.state)

        # If robot is going to pickup, set goal location to the task's start location
        elif agent.status == 1:
            # Get current assigned task's start location
            goal_locations.append(agent.task_sequence[0][1])
            
        # If robot is going to delivery, set goal location to the task's goal location
        elif agent.status == 2:
            goal_locations.append(agent.task_sequence[0][2])

        # Pick/place service: hold position while waiting at pickup or delivery.
        elif agent.status in (STATUS_PICKING, STATUS_PLACING):
            goal_locations.append(agent.state)

        # If robot is a free_agent, set goal location to current state
        else:
            goal_locations.append(agent.home)

    sequences = []
    w = 1.2

    # Guarantee unique goal locations before path planning. PBS asserts a
    # single permanent goal per cell (ConstraintTable::insert2CAT); two agents
    # sharing a goal cell abort the planner. The baseline allocators never
    # emit duplicate goals, but crM2M shuffle tasks can place a pickup/drop-off
    # cell that coincides with a concurrent real task's cell, so any residual
    # collision must be resolved here. Keep the first occurrence of each cell
    # and defer the rest: idle agents (status 0) have no real goal so they go
    # to their home; task-bearing agents wait in place at their current state
    # this tick (a one-tick deferral -- they re-plan toward the real goal next
    # tick, the task is not dropped) and fall back to home if the state is also
    # taken. This loop is a no-op when goals are already unique, so baseline
    # and LNS-PBS routing behaviour is unchanged.
    if len(goal_locations) != len(set(goal_locations)):
        used = set()
        for i, agent in enumerate(Rs.agents):
            goal = goal_locations[i]
            if goal not in used:
                used.add(goal)
                continue
            fallbacks = (
                [agent.home, agent.state]
                if agent.status == 0
                else [agent.state, agent.home]
            )
            new_goal = next((c for c in fallbacks if c not in used), goal)
            goal_locations[i] = new_goal
            used.add(new_goal)
        if len(goal_locations) != len(set(goal_locations)):
            print(f"[WARN] router could not fully de-duplicate goal locations: {goal_locations}")

    print(f"Goal locations: {goal_locations}")
    # print(f"Number of goal locations: {len(goal_locations)}")
    # print(f"Number of unique goal locations: {len(set(goal_locations))}")
    
    latch = False
    while not sequences:
        # Execute the path planning algorithm
        if path_planning_strategy == "pbs":
            if len(goal_locations) != len(set(goal_locations)):
                print(f"Duplicate goal locations: {goal_locations}")
            else:
                print(f"No duplicate goal locations.")
            if len(Rs.get_agent_states()) != len(set(Rs.get_agent_states())):
                print(f"Duplicate agent states: {Rs.get_agent_states()}")
            else:
                print(f"No duplicate agent states.")
            sequences = pbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations, seed)
        elif path_planning_strategy == "ecbs":
            sequences = eecbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        if sequences == []:
            print("+++++++++++++++++++Execution Failed with w = ", w)
            S.update_num_path_plan_fail()
            
        # If a solution cannot be found with a higher suboptimality bound, break
        if w >= 1.2:
            if latch:
                break
            for i, agent in enumerate(Rs.agents):
                if agent.path_sequence == []:
                    # Pick/place agents must hold position, not return home.
                    if agent.status in (STATUS_PICKING, STATUS_PLACING):
                        goal_locations[i] = agent.state
                    else:
                        goal_locations[i] = agent.home
            latch = True
        w += 5.0

    if not sequences:
        # No feasible joint plan: hold everyone in place rather than run stale
        # paths that ignore each other (the aisle in/out oscillation source).
        clear_all_paths(Rs)
        return Rs
    
    # Remove first item in sequences, as they are the robot's current location
    for i, agent in enumerate(Rs.agents):
        temp_sequence = sequences[i][1:]
        agent.path_sequence = temp_sequence
        agent.blocked_ticks = 0
    
    return Rs