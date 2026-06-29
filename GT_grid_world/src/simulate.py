from typing import Tuple, Dict

from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .logging_config import get_logger
from .utils import *

_log = get_logger("simulate")

# Task type codes mirror those in case_request_generator.py:
#   0 = outbound (warehouse -> driveway)
#   1 = inbound  (driveway  -> warehouse)
#   2 = shuffle  (warehouse -> warehouse, shelf-to-shelf)
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2

# Task types whose pickup happens in the warehouse (so SKU removal there
# affects their start_locs).
WAREHOUSE_PICKUP_TASK_TYPES = frozenset({TASK_TYPE_OUTBOUND, TASK_TYPE_SHUFFLE})

# Task types whose dropoff happens in the warehouse (so warehouse-empty
# changes affect their goal_locs).
WAREHOUSE_DROPOFF_TASK_TYPES = frozenset({TASK_TYPE_INBOUND, TASK_TYPE_SHUFFLE})


def _refresh_tasks_after_warehouse_change(J : set, G : Graph, changed_task_id : int, sku_id : int) -> None:
    """
    Ensure all tasks referencing warehouse locations remain consistent with the
    current inventory layout after a SKU is removed or added.

    Refresh policy:
      - For tasks whose pickup happens in the warehouse and whose SKU matches
        the changed SKU, re-derive start_locs from current SKU instances.
      - For tasks whose dropoff happens in the warehouse, re-derive goal_locs
        from current warehouse-empty cells (regardless of SKU, because adding
        or removing any SKU shifts the empty set).
      - For outbound tasks (dropoff = driveway), re-derive goal_locs from
        current driveway-empty cells. The current execution flow only modifies
        warehouse occupancy here, so this branch is mostly defensive but kept
        consistent so the same helper is reusable when driveway-side
        transitions are added.
    """
    if not J:
        return

    warehouse_empty = frozenset(G.warehouse.get_empty_locations())
    driveway_empty = frozenset(G.driveway.get_empty_locations())
    same_sku_warehouse_instances = frozenset(G.warehouse.get_sku_instances(sku_id)) if sku_id is not None else frozenset()

    for other_task_id in list(J.keys()):
        if other_task_id == changed_task_id:
            continue

        start_locations, goal_locations, deadline, task_sku_id, task_type = J[other_task_id]

        new_start_locs = start_locations
        new_goal_locs = goal_locations

        if task_type in WAREHOUSE_PICKUP_TASK_TYPES and task_sku_id == sku_id:
            new_start_locs = same_sku_warehouse_instances
        if task_type in WAREHOUSE_DROPOFF_TASK_TYPES:
            new_goal_locs = warehouse_empty
        elif task_type == TASK_TYPE_OUTBOUND:
            new_goal_locs = driveway_empty

        if new_start_locs is not start_locations or new_goal_locs is not goal_locations:
            J[other_task_id] = (new_start_locs, new_goal_locs, deadline, task_sku_id, task_type)


def simulate(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], J_a : Dict[int, Tuple], map_name : str, t : int) -> Tuple[AgentLoader, Dict[int, Tuple], Dict[int, Tuple]]:
    """
    Simulate the system for one timestep.

    Dual cycling (IB->OB at aisles, OB->IB at the driveway) is *not* handled
    here any more. It is now a pipeline stage inside ``TaskAllocation`` (see
    ``task_allocation_algorithms.dual_cycle_allocation``), so chained tails
    appear in ``agent.task_sequence`` *before* the path planner runs. The
    simulator is a pure executor of the plan it receives.

    Args:
        S (Stats): Statistics object
        G (Graph): Graph object
        Rs (AgentLoader): AgentLoader object
        J (Dict[int, Tuple]): Dictionary of tasks
        map_name (str): Name of the map
        t (int): Current timestep

    Returns:
        Tuple[AgentLoader, Dict[int, Tuple]]: Updated AgentLoader object and updated dictionary of tasks
    """

    # Update state of robots
    for agent in Rs.agents:
        # If robot sequence is stationary, leave the robot in place (wait action)
        if len(agent.path_sequence) == 0:
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            # Update Agent State and Graph Occupied States
            G.set_occupied(agent.state, False)
            old_state = agent.state
            agent.state = agent.path_sequence.pop(0)
            G.set_occupied(agent.state, True)
            
            if agent.status == 1:
                if old_state != agent.state:
                    S.update_actual_pickup_distance(agent.task_sequence[0][0], 1)
                S.update_actual_pickup_duration(agent.task_sequence[0][0], S.get_actual_pickup_duration(agent.task_sequence[0][0]) + 1)
                
            elif agent.status == 2:
                if agent.get_sku_id_carrying() is None:
                    raise ValueError(f"Agent {agent.id} is not carrying any item when it is in the delivery phase for task {agent.task_sequence[0][0]}: agent status {agent.status}")
                if old_state != agent.state:
                    S.update_actual_distance(agent.task_sequence[0][0], 1)
                S.update_actual_duration(agent.task_sequence[0][0], S.get_actual_duration(agent.task_sequence[0][0]) + 1)
            else:
                pass
    
    S.append_number_of_collisions(Rs.detect_collisions())
    
    # Update Statistics for the total paths taken
    S.add_paths(Rs.get_agent_states())
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    
    S.append_carrying_skus(Rs.get_all_agent_carrying_skus())
    
    for agent in Rs.agents:
        if agent.status == 1:
            if agent.state == agent.task_sequence[0][1]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                deadline = task[3]

                # TA-Hybrid pre-allocates future tasks to agents at t=0, so an
                # agent can reach its pre-committed pickup cell *before* the
                # task is released into ``J``. Two failure modes can follow:
                #   (1) The cell is empty -- pickup yields no SKU; we must
                #       stay in status=1 until the task releases (handled
                #       by the existing ``pickup_succeeded`` gate below).
                #   (2) Another task's SKU happens to be at the pre-committed
                #       cell (e.g. a different inbound was placed there by
                #       ``add_tasks_from_schedule`` after materialization);
                #       the agent would otherwise pick up the wrong SKU and
                #       crash later at delivery with ``J[task_id]`` raising
                #       ``KeyError``. The pre-check below blocks that path
                #       by refusing to pick up anything when the head task
                #       isn't in J yet -- the SKU sitting at the cell
                #       belongs to a different, already-released task whose
                #       own assignee will collect it.
                pickup_succeeded = False

                if task_id not in J and task_id not in J_a:
                    # Task hasn't been released yet (TA-Hybrid pre-allocation
                    # case). Don't touch any SKU at this cell. Shuffles live in
                    # ``J_a`` rather than ``J``, so they must be admitted here
                    # too -- otherwise the pickup is silently skipped and the
                    # shuffle can never execute.
                    pass
                # Stale shuffle: the agent reached its committed source cell but
                # that cell no longer holds the SKU this shuffle was generated
                # for. Warehouse churn (an outbound emptied the cell, possibly an
                # inbound then refilled it with a different SKU) invalidated the
                # move between commitment and arrival. Moving the wrong item is
                # pointless and would crash at the delivery-side SKU-match check,
                # so abort the shuffle and free the agent. Shuffles are optional,
                # so dropping one is harmless (unlike a real task).
                elif task_id in J_a and (
                    start_location not in G.warehouse.get_full_locations()
                    or G.warehouse.get_sku_at_location(start_location).sku_id != J_a[task_id][3]
                ):
                    J_a.pop(task_id, None)
                    agent.task_sequence.pop(0)
                    agent.path_sequence = []
                    agent.status = 1 if agent.task_sequence else 0
                    continue
                # Outbound or warehouse-based pickup
                elif start_location in G.warehouse.get_full_locations():
                    try:
                        sku_id = G.warehouse.get_sku_at_location(start_location).sku_id
                        if sku_id is None:
                            raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
                        agent.set_sku_id_carrying(G.warehouse.get_sku_at_location(start_location).sku_id)
                        G.warehouse.remove_sku_instance(start_location)
                        G.update_sku_KD_trees(agent.get_sku_id_carrying())

                        _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)
                        _refresh_tasks_after_warehouse_change(J_a, G, task_id, sku_id)

                        if agent.get_sku_id_carrying() is None:
                            raise ValueError(f"Agent should be holding item after pickup ... Exiting")
                        pickup_succeeded = True

                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from warehouse at {start_location}: {e}")
                        exit()
                # Inbound: picking up from driveway (now empty)
                elif start_location in G.driveway.get_full_locations():
                    try:
                        # Only save sku id
                        print(f"Agent {agent.id} picking up task {task_id} with sku {G.driveway.get_sku_at_location(start_location)}")
                        sku_id = G.driveway.get_sku_at_location(start_location).sku_id
                        if sku_id is None:
                            raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
                        agent.set_sku_id_carrying(sku_id)
                        G.driveway.remove_sku_instance(start_location)

                        _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)

                        if agent.get_sku_id_carrying() is None:
                            raise ValueError(f"Agent should be holding item after pickup ... Exiting")
                        pickup_succeeded = True

                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from driveway at {start_location}: {e}")

                if pickup_succeeded:
                    S.add_completed_to_pickup_task_id(task_id)
                    agent.status = 2
                # else: task not yet released OR SKU not yet at start_location
                # (e.g. inbound task whose release time hasn't elapsed). Keep
                # status=1 and let the agent retry the pickup on the next tick.
        elif agent.status == 2:
            if agent.state == agent.task_sequence[0][2]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                deadline = task[3]
                
                # Shuffles live in the separate ``J_a`` pool; real tasks in ``J``.
                if task_id in J_a:
                    deadline = J_a[task_id][2]
                    sku_id = J_a[task_id][3]
                    inbound_task = J_a[task_id][4]
                else:
                    deadline = J[task_id][2]
                    sku_id = J[task_id][3]
                    inbound_task = J[task_id][4]

                if sku_id != agent.get_sku_id_carrying():
                    raise ValueError(f"Agent {agent.id} carrying sku {agent.get_sku_id_carrying()} but task {task_id} requires sku {sku_id} ... Exiting")

                # Inbound task: dropping off to warehouse
                if goal_location in G.warehouse.get_empty_locations():
                    print(f"Agent {agent.id} dropping off task {task_id} with sku {agent.get_sku_id_carrying()}")
                    carried_sku = agent.get_sku_id_carrying()
                    G.warehouse.add_sku_instance(carried_sku, goal_location)
                    G.update_sku_KD_trees(agent.get_sku_id_carrying())
                    _refresh_tasks_after_warehouse_change(J, G, task_id, carried_sku)
                    _refresh_tasks_after_warehouse_change(J_a, G, task_id, carried_sku)

                                
                elif goal_location in G.driveway.get_empty_locations():
                    pass
                agent.set_sku_id_carrying(None)

                if inbound_task == TASK_TYPE_SHUFFLE:
                    # crM2M (concatenated rearrangement): a completed shuffle is
                    # reported on the separate rearrangement track. Service time /
                    # tardiness are deliberately skipped -- those are deadline-graded
                    # metrics for real (inbound/outbound) tasks, whereas a shuffle's
                    # "deadline" is just its look-ahead window edge.
                    S.add_completed_rearrangement_task_id(task_id, t, start_location, goal_location, int(deadline), int(sku_id), int(inbound_task))
                else:
                    S.add_completed_task_id(task_id, t, start_location, goal_location, int(deadline), int(sku_id), int(inbound_task))
                    S.update_service_time(task_id, t)

                if task_id in J_a:
                    J_a.pop(task_id)
                else:
                    J.pop(task_id)

                agent.task_sequence.pop(0)

                if agent.task_sequence == []:
                    agent.status = 0
                else:
                    agent.status = 1
                    new_task_id = agent.task_sequence[0][0]
                    S.add_actual_distance(new_task_id)
                    S.add_actual_pickup_distance(new_task_id)
                    
                    # Initialize durations for new task 
                    S.add_actual_duration(new_task_id)
                    S.add_actual_pickup_duration(new_task_id)

                    estimated_to_pickup_path = G.get_distance(agent.state, agent.task_sequence[0][1])
                    estimated_task_path = G.get_distance(agent.task_sequence[0][1], agent.task_sequence[0][2])

                    S.add_estimated_pickup_duration(new_task_id, estimated_to_pickup_path)
                    S.add_estimated_pickup_distance(new_task_id, estimated_to_pickup_path)

                    S.add_estimated_distance(new_task_id, estimated_task_path)
                    S.add_estimated_duration(new_task_id, estimated_task_path)
                    
                    if t <= 100:
                        S.append_early_task_ids(new_task_id)
                        
    return Rs, J, J_a