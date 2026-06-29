from typing import Optional, Tuple, Dict, TYPE_CHECKING

from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .logging_config import get_logger
from .output_buffer import consumption_tick, outbound_delivery_blocked
from .utils import *

if TYPE_CHECKING:
    from .output_buffer import OutputBuffer

_log = get_logger("simulate")

# Task type codes mirror those in case_request_generator.py:
#   0 = outbound (warehouse -> driveway)
#   1 = inbound  (driveway  -> warehouse)
#   2 = shuffle  (warehouse -> warehouse, shelf-to-shelf)
TASK_TYPE_OUTBOUND = 0
TASK_TYPE_INBOUND = 1
TASK_TYPE_SHUFFLE = 2

STATUS_FREE = 0
STATUS_TO_PICKUP = 1
STATUS_TO_DELIVERY = 2
STATUS_PICKING = 3
STATUS_PLACING = 4

DEFAULT_PICK_PLACE_DURATION = 4

# Task types whose pickup happens in the warehouse (so SKU removal there
# affects their start_locs).
WAREHOUSE_PICKUP_TASK_TYPES = frozenset({TASK_TYPE_OUTBOUND, TASK_TYPE_SHUFFLE})

# Task types whose dropoff happens in the warehouse (so warehouse-empty
# changes affect their goal_locs).
WAREHOUSE_DROPOFF_TASK_TYPES = frozenset({TASK_TYPE_INBOUND, TASK_TYPE_SHUFFLE})


def _is_cell_occupied_by_other_agent(
    agents,
    cell: Tuple[int, int],
    moving_agent,
) -> bool:
    """Return True when another agent is currently at ``cell``."""
    return any(
        other.state == cell
        for other in agents
        if other.id != moving_agent.id
    )


def _collect_global_allocation_snapshot(Rs):
    """Return ``(allocated_task_ids, allocated_locs)`` across every agent's
    task_sequence. Used by the dual-cycle helpers to avoid stealing a task
    that the regular allocator already gave to another agent.
    """
    allocated_task_ids = set()
    allocated_locs = set()
    for ag in Rs.agents:
        for task_tuple in ag.task_sequence:
            allocated_task_ids.add(task_tuple[0])
            allocated_locs.add(task_tuple[1])
            allocated_locs.add(task_tuple[2])
    return allocated_task_ids, allocated_locs


def _find_aisle_dual_cycle_chain(
    agent, completed_goal: Tuple[int, int], J: Dict[int, Tuple], Rs, G: Graph
) -> Optional[Tuple]:
    """Aisle dual cycling (IB -> OB).

    The agent just dropped an inbound box at ``completed_goal`` (a warehouse
    aisle cell). Look for an unallocated outbound (type=0) task whose pickup
    is in the SAME aisle (same column) as ``completed_goal``. If one exists
    and has a valid driveway dropoff cell, return a concrete
    ``(task_id, chosen_start, chosen_goal, deadline)`` tuple ready to append
    to ``agent.task_sequence``. Otherwise return ``None``.

    The chosen ``(start, goal)`` minimises agent.state -> start + start ->
    goal travel as a tiebreaker among eligible OB tasks. The proper
    rearrangement objective for chaining decisions lives in roadmap 1.6 / 3.4;
    this is the 1.5-skeleton heuristic.
    """
    if not G.is_warehouse_aisle_location(completed_goal):
        return None
    same_aisle_cells = set(G.get_same_aisle_locations(completed_goal))

    allocated_task_ids, allocated_locs = _collect_global_allocation_snapshot(Rs)
    driveway_empty = set(G.driveway.get_empty_locations())
    warehouse_full = set(G.warehouse.get_full_locations())

    best = None
    best_cost = float("inf")
    for task_id, (start_locs, goal_locs, deadline, _sku, type_) in J.items():
        if task_id in allocated_task_ids:
            continue
        if type_ != TASK_TYPE_OUTBOUND:
            continue
        eligible_starts = [
            s for s in start_locs
            if s in same_aisle_cells and s in warehouse_full and s not in allocated_locs
        ]
        if not eligible_starts:
            continue
        eligible_goals = [
            g for g in goal_locs if g in driveway_empty and g not in allocated_locs
        ]
        if not eligible_goals:
            continue
        chosen_start = min(eligible_starts, key=lambda s: G.get_distance(agent.state, s))
        chosen_goal = min(eligible_goals, key=lambda g: G.get_distance(chosen_start, g))
        cost = G.get_distance(agent.state, chosen_start) + G.get_distance(chosen_start, chosen_goal)
        if cost < best_cost:
            best_cost = cost
            best = (task_id, chosen_start, chosen_goal, deadline)
    return best


def _find_driveway_dual_cycle_chain(
    agent, completed_goal: Tuple[int, int], J: Dict[int, Tuple], Rs, G: Graph
) -> Optional[Tuple]:
    """Driveway dual cycling (OB -> IB at driveways).

    The agent just dropped an outbound box at ``completed_goal`` (a driveway
    cell). Look for an unallocated inbound (type=1) task whose pickup is at
    any driveway cell that currently holds a pre-placed SKU. If one exists
    and has a valid warehouse-empty dropoff cell, return a concrete
    ``(task_id, chosen_start, chosen_goal, deadline)`` tuple. Otherwise return
    ``None``.

    On the current ``symbotic_2026`` branch the driveway is a single physical
    region, so "same driveway" is trivially "any driveway cell". When the
    per-cell I/O direction typing from ``local_task_reallocation`` lands
    (DPS-style mixed layout), this helper will tighten the eligibility
    filter to the matching subregion.
    """
    if not G.is_driveway_location(completed_goal):
        return None

    allocated_task_ids, allocated_locs = _collect_global_allocation_snapshot(Rs)
    driveway_full = set(G.driveway.get_full_locations())
    warehouse_empty = set(G.warehouse.get_empty_locations())

    best = None
    best_cost = float("inf")
    for task_id, (start_locs, goal_locs, deadline, _sku, type_) in J.items():
        if task_id in allocated_task_ids:
            continue
        if type_ != TASK_TYPE_INBOUND:
            continue
        eligible_starts = [
            s for s in start_locs if s in driveway_full and s not in allocated_locs
        ]
        if not eligible_starts:
            continue
        eligible_goals = [
            g for g in goal_locs if g in warehouse_empty and g not in allocated_locs
        ]
        if not eligible_goals:
            continue
        chosen_start = min(eligible_starts, key=lambda s: G.get_distance(agent.state, s))
        chosen_goal = min(eligible_goals, key=lambda g: G.get_distance(chosen_start, g))
        cost = G.get_distance(agent.state, chosen_start) + G.get_distance(chosen_start, chosen_goal)
        if cost < best_cost:
            best_cost = cost
            best = (task_id, chosen_start, chosen_goal, deadline)
    return best


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


def _execute_pickup(
    agent,
    task_id: int,
    start_location: Tuple[int, int],
    G: Graph,
    J: Dict[int, Tuple],
    S: Stats = None,
) -> None:
    if start_location in G.warehouse.get_full_locations():
        sku_id = G.warehouse.get_sku_at_location(start_location).sku_id
        if sku_id is None:
            raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
        agent.set_sku_id_carrying(sku_id)
        G.warehouse.remove_sku_instance(start_location)
        G.update_sku_KD_trees(agent.get_sku_id_carrying())
        _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)
        if agent.get_sku_id_carrying() is None:
            raise ValueError("Agent should be holding item after pickup ... Exiting")
    elif start_location in G.driveway.get_full_locations():
        print(f"Agent {agent.id} picking up task {task_id} with sku {G.driveway.get_sku_at_location(start_location)}")
        sku_id = G.driveway.get_sku_at_location(start_location).sku_id
        if sku_id is None:
            raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
        agent.set_sku_id_carrying(sku_id)
        G.driveway.remove_sku_instance(start_location)
        _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)
        if agent.get_sku_id_carrying() is None:
            raise ValueError("Agent should be holding item after pickup ... Exiting")

    if S is not None:
        S.record_aisle_pick_place_activity(start_location, G)


def _complete_delivery(
    agent,
    task_id: int,
    start_location: Tuple[int, int],
    goal_location: Tuple[int, int],
    deadline: int,
    sku_id: int,
    inbound_task: int,
    J: Dict[int, Tuple],
    J_a: Dict[int, Tuple],
    J_a_objectives: Dict[int, Dict[str, float]],
    G: Graph,
    S: Stats,
    Rs: AgentLoader,
    t: int,
    aisle_dual_cycle: bool,
    driveway_dual_cycle: bool,
    output_buffer: Optional["OutputBuffer"] = None,
) -> bool:
    if inbound_task == TASK_TYPE_OUTBOUND and outbound_delivery_blocked(output_buffer):
        return False

    if sku_id != agent.get_sku_id_carrying():
        raise ValueError(
            f"Agent {agent.id} carrying sku {agent.get_sku_id_carrying()} "
            f"but task {task_id} requires sku {sku_id} ... Exiting"
        )

    if goal_location in G.warehouse.get_empty_locations():
        print(f"Agent {agent.id} dropping off task {task_id} with sku {agent.get_sku_id_carrying()}")
        carried_sku = agent.get_sku_id_carrying()
        G.warehouse.add_sku_instance(carried_sku, goal_location)
        G.update_sku_KD_trees(agent.get_sku_id_carrying())
        _refresh_tasks_after_warehouse_change(J, G, task_id, carried_sku)
    elif goal_location in G.driveway.get_empty_locations():
        if inbound_task == TASK_TYPE_OUTBOUND and output_buffer is not None:
            output_buffer.record_outbound_delivery(1.0)

    S.record_aisle_pick_place_activity(goal_location, G)

    agent.set_sku_id_carrying(None)

    if inbound_task == TASK_TYPE_SHUFFLE:
        objectives = J_a_objectives.pop(task_id, None)
        S.add_completed_rearrangement_task_id(
            task_id,
            t,
            start_location,
            goal_location,
            int(deadline),
            int(sku_id),
            int(inbound_task),
            benefit=objectives.get("benefit") if objectives else None,
            utility=objectives.get("utility") if objectives else None,
            detour_cost=objectives.get("detour_cost") if objectives else None,
        )
    else:
        S.add_completed_task_id(
            task_id,
            t,
            start_location,
            goal_location,
            int(deadline),
            int(sku_id),
            int(inbound_task),
        )
        S.update_service_time(task_id, t)

    if task_id in J_a:
        J_a.pop(task_id)
    else:
        J.pop(task_id)

    agent.task_sequence.pop(0)

    if agent.task_sequence == []:
        chained_tuple: Optional[Tuple] = None
        if aisle_dual_cycle and inbound_task == TASK_TYPE_INBOUND:
            chained_tuple = _find_aisle_dual_cycle_chain(agent, goal_location, J, Rs, G)
            if chained_tuple is not None:
                _log.debug(
                    "Aisle dual cycle (IB->OB) at t=%s: agent %s chained task %s after task %s",
                    t, agent.id, chained_tuple[0], task_id,
                )
        elif driveway_dual_cycle and inbound_task == TASK_TYPE_OUTBOUND:
            chained_tuple = _find_driveway_dual_cycle_chain(agent, goal_location, J, Rs, G)
            if chained_tuple is not None:
                _log.debug(
                    "Driveway dual cycle (OB->IB) at t=%s: agent %s chained task %s after task %s",
                    t, agent.id, chained_tuple[0], task_id,
                )
        if chained_tuple is not None:
            agent.task_sequence.append(chained_tuple)

    if agent.task_sequence == []:
        agent.status = STATUS_FREE
    else:
        agent.status = STATUS_TO_PICKUP
        new_task_id = agent.task_sequence[0][0]
        S.add_actual_distance(new_task_id)
        S.add_actual_pickup_distance(new_task_id)
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

    return True


def simulate(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], J_a : Dict[int, Tuple], map_name : str, t : int,
             aisle_dual_cycle: bool = False, driveway_dual_cycle: bool = False,
             J_a_objectives: Dict[int, Dict[str, float]] = None,
             pick_place_time: bool = False,
             pick_place_duration: int = DEFAULT_PICK_PLACE_DURATION,
             output_buffer: Optional["OutputBuffer"] = None) -> Tuple[AgentLoader, Dict[int, Tuple]]:
    """
    Simulate the system for one timestep.

    Args:
        S (Stats): Statistics object
        G (Graph): Graph object
        Rs (AgentLoader): AgentLoader object
        J (Dict[int, Tuple]): Dictionary of tasks
        map_name (str): Name of the map
        t (int): Current timestep
        aisle_dual_cycle: When True, after an agent completes an inbound task
            in the warehouse and would otherwise idle, search J for an
            unallocated same-aisle outbound task and chain it onto
            ``agent.task_sequence`` (1.5-skeleton dual cycling, IB -> OB).
        driveway_dual_cycle: When True, after an agent completes an outbound
            task at a driveway cell and would otherwise idle, search J for an
            unallocated inbound task whose pickup is at any driveway cell and
            chain it (1.5-skeleton dual cycling, OB -> IB).
        pick_place_time: When True, agents wait ``pick_place_duration`` timesteps
            at pickup (status 3) and delivery (status 4) before updating inventory.
        pick_place_duration: Timesteps to wait at each pick/place (default 5).
        output_buffer: Optional shared outbound output buffer; when set, one
            consumption tick is applied each call.

    Returns:
        Tuple[AgentLoader, Dict[int, Tuple]]: Updated AgentLoader object and updated dictionary of tasks and rearrangement tasks
    """

    if J_a_objectives is None:
        J_a_objectives = {}

    # Update state of robots
    for agent in Rs.agents:
        if agent.status in (STATUS_PICKING, STATUS_PLACING):
            continue
        # If robot sequence is stationary, leave the robot in place (wait action)
        if len(agent.path_sequence) == 0:
            continue

        next_state = agent.path_sequence[0]
        if _is_cell_occupied_by_other_agent(Rs.agents, next_state, agent):
            continue

        # If robot does have a sequence of actions, pop next state and update
        G.set_occupied(agent.state, False)
        old_state = agent.state
        agent.state = agent.path_sequence.pop(0)
        G.set_occupied(agent.state, True)
            
        if agent.status == STATUS_TO_PICKUP:
            if old_state != agent.state:
                S.update_actual_pickup_distance(agent.task_sequence[0][0], 1)
            S.update_actual_pickup_duration(agent.task_sequence[0][0], S.get_actual_pickup_duration(agent.task_sequence[0][0]) + 1)
            
        elif agent.status == STATUS_TO_DELIVERY:
            if agent.get_sku_id_carrying() is None:
                raise ValueError(f"Agent {agent.id} is not carrying any item when it is in the delivery phase for task {agent.task_sequence[0][0]}: agent status {agent.status}")
            if old_state != agent.state:
                S.update_actual_distance(agent.task_sequence[0][0], 1)
            S.update_actual_duration(agent.task_sequence[0][0], S.get_actual_duration(agent.task_sequence[0][0]) + 1)
    
    S.append_number_of_collisions(Rs.detect_collisions())
    
    # Update Statistics for the total paths taken
    S.add_paths(Rs.get_agent_states())
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    
    S.append_carrying_skus(Rs.get_all_agent_carrying_skus())
    
    for agent in Rs.agents:
        if pick_place_time and agent.status == STATUS_PICKING:
            task_id = agent.task_sequence[0][0]
            S.update_actual_pickup_duration(
                task_id,
                S.get_actual_pickup_duration(task_id) + 1,
            )
            agent.pick_place_counter -= 1
            if agent.pick_place_counter > 0:
                continue
            task = agent.task_sequence[0]
            start_location = task[1]
            try:
                _execute_pickup(agent, task_id, start_location, G, J, S)
            except Exception as e:
                print(f"[WARN] Could not complete pickup for agent {agent.id} at {start_location}: {e}")
                exit()
            S.add_completed_to_pickup_task_id(task_id)
            agent.status = STATUS_TO_DELIVERY
            continue

        if pick_place_time and agent.status == STATUS_PLACING:
            task = agent.task_sequence[0]
            task_id = task[0]
            S.update_actual_duration(
                task_id,
                S.get_actual_duration(task_id) + 1,
            )
            agent.pick_place_counter -= 1
            if agent.pick_place_counter > 0:
                continue
            start_location = task[1]
            goal_location = task[2]
            if task_id in J_a:
                deadline = J_a[task_id][2]
                sku_id = J_a[task_id][3]
                inbound_task = J_a[task_id][4]
            else:
                deadline = J[task_id][2]
                sku_id = J[task_id][3]
                inbound_task = J[task_id][4]
            if inbound_task == TASK_TYPE_OUTBOUND and outbound_delivery_blocked(output_buffer):
                print(f"============Output Delivery Blocked due to buffer level: {output_buffer.level}")
                agent.status = STATUS_TO_DELIVERY
                continue
            if not _complete_delivery(
                agent,
                task_id,
                start_location,
                goal_location,
                deadline,
                sku_id,
                inbound_task,
                J,
                J_a,
                J_a_objectives,
                G,
                S,
                Rs,
                t,
                aisle_dual_cycle,
                driveway_dual_cycle,
                output_buffer=output_buffer,
            ):
                agent.status = STATUS_TO_DELIVERY
            continue

        if agent.status == STATUS_TO_PICKUP:
            if agent.state == agent.task_sequence[0][1]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                if pick_place_time:
                    agent.status = STATUS_PICKING
                    agent.pick_place_counter = pick_place_duration
                    continue
                try:
                    _execute_pickup(agent, task_id, start_location, G, J, S)
                except Exception as e:
                    print(f"[WARN] Could not remove SKU at {start_location}: {e}")
                    exit()
                S.add_completed_to_pickup_task_id(task_id)
                agent.status = STATUS_TO_DELIVERY
        elif agent.status == STATUS_TO_DELIVERY:
            if agent.state == agent.task_sequence[0][2]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                if task_id in J_a:
                    deadline = J_a[task_id][2]
                    sku_id = J_a[task_id][3]
                    inbound_task = J_a[task_id][4]
                else:
                    deadline = J[task_id][2]
                    sku_id = J[task_id][3]
                    inbound_task = J[task_id][4]
                if inbound_task == TASK_TYPE_OUTBOUND and outbound_delivery_blocked(output_buffer):
                    continue
                if pick_place_time:
                    agent.status = STATUS_PLACING
                    agent.pick_place_counter = pick_place_duration
                    continue
                if not _complete_delivery(
                    agent,
                    task_id,
                    start_location,
                    goal_location,
                    deadline,
                    sku_id,
                    inbound_task,
                    J,
                    J_a,
                    J_a_objectives,
                    G,
                    S,
                    Rs,
                    t,
                    aisle_dual_cycle,
                    driveway_dual_cycle,
                    output_buffer=output_buffer,
                ):
                    continue

    consumption_tick(output_buffer)
    if output_buffer is not None:
        S.log_output_buffer_level(t, output_buffer.level)
                        
    return Rs, J, J_a