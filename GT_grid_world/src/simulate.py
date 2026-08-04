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

# Agent status codes (mirror agent.py).
STATUS_FREE = 0
STATUS_TO_PICKUP = 1
STATUS_TO_DELIVERY = 2
STATUS_PICKING = 3   # waiting at pickup cell during pick/place delay
STATUS_PLACING = 4   # waiting at delivery cell during pick/place delay

# Default ticks an agent spends picking and (separately) placing when
# pick/place time is enabled. Matches Ethan's task_queue default.
DEFAULT_PICK_PLACE_DURATION = 4

# Task types whose pickup happens in the warehouse (so SKU removal there
# affects their start_locs).
WAREHOUSE_PICKUP_TASK_TYPES = frozenset({TASK_TYPE_OUTBOUND, TASK_TYPE_SHUFFLE})

# Task types whose dropoff happens in the warehouse (so warehouse-empty
# changes affect their goal_locs).
WAREHOUSE_DROPOFF_TASK_TYPES = frozenset({TASK_TYPE_INBOUND, TASK_TYPE_SHUFFLE})


def _is_cell_occupied_by_other_agent(agents, cell: Tuple[int, int], moving_agent) -> bool:
    """Return True when another agent currently stands on ``cell``.

    Matches Ethan's ``task_queue`` reactive collision guard: movement is only
    permitted into a cell no other agent occupies *right now*. Agents processed
    earlier in the tick have already updated ``state`` in place, so a follower
    correctly sees a leader's cell free once the leader has advanced.
    """
    return any(other.state == cell for other in agents if other.id != moving_agent.id)


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
        currently-empty cells within the task's already-committed aisle
        (the driveway column its goal_locs were originally restricted to),
        never expanding back out to every empty driveway cell. The current
        execution flow only modifies warehouse occupancy here, so this
        branch is mostly defensive but kept consistent so the same helper
        is reusable when driveway-side transitions are added.
    """
    if not J:
        return

    warehouse_empty = frozenset(G.warehouse.get_empty_locations())
    driveway_empty = frozenset(G.driveway.get_empty_locations())
    same_sku_warehouse_instances = frozenset(G.warehouse.get_sku_instances(sku_id)) if sku_id is not None else frozenset()

    for other_task_id in list(J.keys()):
        if other_task_id == changed_task_id:
            continue

        entry = J[other_task_id]
        start_locations, goal_locations, deadline, task_sku_id, task_type = entry[:5]
        # Optional 6th field: driveway reference_location on crM2M J_a shuffles.
        extra = entry[5:]

        new_start_locs = start_locations
        new_goal_locs = goal_locations

        if task_type in WAREHOUSE_PICKUP_TASK_TYPES and task_sku_id == sku_id:
            new_start_locs = same_sku_warehouse_instances
        if task_type in WAREHOUSE_DROPOFF_TASK_TYPES:
            new_goal_locs = warehouse_empty
        elif task_type == TASK_TYPE_OUTBOUND and goal_locations:
            aisle_column = next(iter(goal_locations))[1]
            new_goal_locs = frozenset(
                loc for loc in driveway_empty if loc[1] == aisle_column
            )

        if new_start_locs is not start_locations or new_goal_locs is not goal_locations:
            J[other_task_id] = (
                new_start_locs,
                new_goal_locs,
                deadline,
                task_sku_id,
                task_type,
                *extra,
            )


def _attempt_pickup(agent, task, G: Graph, J: Dict[int, Tuple], J_a: Dict[int, Tuple], S: Stats) -> str:
    """Try to execute the pickup for ``agent``'s head ``task`` at its current cell.

    Returns one of:
      ``"not_released"``  task not yet in ``J``/``J_a`` (TA-Hybrid pre-allocation)
                          -- leave the agent waiting in status 1.
      ``"stale_shuffle"`` a committed shuffle's source cell no longer holds its
                          SKU (warehouse churn) -- caller aborts and frees agent.
      ``"succeeded"``     SKU picked up; caller should advance to delivery.
      ``"failed"``        SKU not present yet (e.g. inbound not arrived) -- retry.

    This shares the (unchanged) status-1 pickup logic between the immediate and
    the pick/place-delayed (status 3) execution paths, preserving our J_a admit /
    stale-shuffle abort / inventory-refresh behaviour.
    """
    task_id = task[0]
    start_location = task[1]

    # Task hasn't been released yet (TA-Hybrid pre-allocation). Don't touch any
    # SKU at this cell. Shuffles live in J_a rather than J, so admit them too.
    if task_id not in J and task_id not in J_a:
        return "not_released"

    # Stale shuffle: the committed source cell no longer holds the SKU this
    # shuffle was generated for (an outbound emptied it, maybe an inbound
    # refilled it with a different SKU). Moving the wrong item would crash at
    # the delivery-side SKU-match check, so abort -- shuffles are optional.
    if task_id in J_a and (
        start_location not in G.warehouse.get_full_locations()
        or G.warehouse.get_sku_at_location(start_location).sku_id != J_a[task_id][3]
    ):
        return "stale_shuffle"

    # Outbound or shuffle: pickup from the warehouse shelf.
    if start_location in G.warehouse.get_full_locations():
        try:
            sku_id = G.warehouse.get_sku_at_location(start_location).sku_id
            if sku_id is None:
                raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
            agent.set_sku_id_carrying(sku_id)
            G.warehouse.remove_sku_instance(start_location)
            G.update_sku_KD_trees(agent.get_sku_id_carrying())

            _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)
            _refresh_tasks_after_warehouse_change(J_a, G, task_id, sku_id)

            if agent.get_sku_id_carrying() is None:
                raise ValueError(f"Agent should be holding item after pickup ... Exiting")
            S.record_aisle_pick_place_activity(start_location, G)
            return "succeeded"
        except Exception as e:
            print(f"[WARN] Could not remove SKU from warehouse at {start_location}: {e}")
            exit()

    # Inbound: pickup from the driveway (now empty).
    if start_location in G.driveway.get_full_locations():
        try:
            print(f"Agent {agent.id} picking up task {task_id} with sku {G.driveway.get_sku_at_location(start_location)}")
            sku_id = G.driveway.get_sku_at_location(start_location).sku_id
            if sku_id is None:
                raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
            agent.set_sku_id_carrying(sku_id)
            G.driveway.remove_sku_instance(start_location)

            _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)

            if agent.get_sku_id_carrying() is None:
                raise ValueError(f"Agent should be holding item after pickup ... Exiting")
            S.record_aisle_pick_place_activity(start_location, G)
            return "succeeded"
        except Exception as e:
            print(f"[WARN] Could not remove SKU from driveway at {start_location}: {e}")

    # SKU not at the start cell yet (e.g. inbound release time not elapsed).
    return "failed"


def _complete_delivery(agent, task, G: Graph, J: Dict[int, Tuple], J_a: Dict[int, Tuple],
                       S: Stats, Rs: AgentLoader, t: int,
                       J_a_objectives: Dict[int, Dict[str, float]] = None,
                       output_buffer: Optional["OutputBuffer"] = None) -> bool:
    """Execute the delivery for ``agent``'s head ``task`` at its goal cell.

    Returns ``True`` on completion. Returns ``False`` *without* mutating state
    when the task is outbound and the shared output buffer is full -- the caller
    keeps the agent waiting (backpressure). Shared between the immediate and the
    pick/place-delayed (status 4) execution paths.

    ``J_a_objectives`` (irM2M insertion) carries the per-shuffle benefit/utility/
    detour terms recorded on completion; ``None`` for crM2M / plain shuffles.
    """
    task_id = task[0]
    start_location = task[1]
    goal_location = task[2]

    # Shuffles live in the separate J_a pool; real tasks in J.
    if task_id in J_a:
        deadline = J_a[task_id][2]
        sku_id = J_a[task_id][3]
        inbound_task = J_a[task_id][4]
    else:
        deadline = J[task_id][2]
        sku_id = J[task_id][3]
        inbound_task = J[task_id][4]

    # Outbound deliveries enter the shared output buffer; if it is full the
    # delivery is blocked and the agent must keep waiting at the driveway cell.
    if inbound_task == TASK_TYPE_OUTBOUND and outbound_delivery_blocked(output_buffer):
        S.record_outbound_buffer_placement_blocked(agent.id, t)
        return False

    if sku_id != agent.get_sku_id_carrying():
        raise ValueError(f"Agent {agent.id} carrying sku {agent.get_sku_id_carrying()} but task {task_id} requires sku {sku_id} ... Exiting")

    # Inbound / shuffle: dropping off to a warehouse cell.
    if goal_location in G.warehouse.get_empty_locations():
        print(f"Agent {agent.id} dropping off task {task_id} with sku {agent.get_sku_id_carrying()}")
        carried_sku = agent.get_sku_id_carrying()
        G.warehouse.add_sku_instance(carried_sku, goal_location)
        G.update_sku_KD_trees(agent.get_sku_id_carrying())
        _refresh_tasks_after_warehouse_change(J, G, task_id, carried_sku)
        _refresh_tasks_after_warehouse_change(J_a, G, task_id, carried_sku)
    # Outbound: dropping off to a driveway cell -> record into the output buffer.
    elif goal_location in G.driveway.get_empty_locations():
        if inbound_task == TASK_TYPE_OUTBOUND and output_buffer is not None:
            output_buffer.record_outbound_delivery(1.0)
            # A successful place closes any open blocking event for this agent.
            S.record_outbound_buffer_unblocked(agent.id, t)

    S.record_aisle_pick_place_activity(goal_location, G)

    agent.set_sku_id_carrying(None)

    if inbound_task == TASK_TYPE_SHUFFLE:
        # crM2M (concatenated rearrangement): a completed shuffle is reported on
        # the separate rearrangement track. Service time / tardiness are skipped
        # -- those are deadline-graded metrics for real tasks, whereas a
        # shuffle's "deadline" is just its look-ahead window edge.
        objectives = J_a_objectives.pop(task_id, None) if J_a_objectives else None
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
        S.add_completed_task_id(task_id, t, start_location, goal_location, int(deadline), int(sku_id), int(inbound_task))
        S.update_service_time(task_id, t)

    if task_id in J_a:
        J_a.pop(task_id)
    else:
        J.pop(task_id)

    agent.task_sequence.pop(0)

    if agent.task_sequence == []:
        agent.status = STATUS_FREE
    else:
        agent.status = STATUS_TO_PICKUP
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

    return True


def simulate(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], J_a : Dict[int, Tuple], map_name : str, t : int,
             J_a_objectives: Dict[int, Dict[str, float]] = None,
             pick_place_time: bool = False,
             pick_place_duration: int = DEFAULT_PICK_PLACE_DURATION,
             output_buffer: Optional["OutputBuffer"] = None) -> Tuple[AgentLoader, Dict[int, Tuple], Dict[int, Tuple]]:
    """
    Simulate the system for one timestep.

    The simulator is a pure executor of the plan it receives: it steps agents
    along their planned paths and executes pickups/deliveries for whatever is
    in ``agent.task_sequence`` (real tasks in ``J``, shuffles in ``J_a``).

    Args:
        S (Stats): Statistics object
        G (Graph): Graph object
        Rs (AgentLoader): AgentLoader object
        J (Dict[int, Tuple]): Dictionary of tasks
        map_name (str): Name of the map
        t (int): Current timestep
        J_a_objectives: irM2M insertion per-shuffle benefit/utility/detour terms,
            popped and recorded on shuffle completion. Empty/``None`` for crM2M.
        pick_place_time: When True, agents wait ``pick_place_duration`` ticks at
            pickup (status 3 = Picking) and at delivery (status 4 = Placing)
            before inventory is mutated -- a per-task service-time penalty
            applied uniformly to every method (no allocator logic changes).
        pick_place_duration: Ticks to wait at each pick and each place.
        output_buffer: Optional shared outbound output buffer. When set, one
            consumption tick is applied each call, outbound deliveries are
            recorded into it, and an outbound delivery is blocked (the agent
            waits) whenever the buffer is at capacity. Inbound/shuffle tasks
            are never affected.

    Returns:
        Tuple[AgentLoader, Dict[int, Tuple], Dict[int, Tuple]]: Updated
        AgentLoader and the (possibly mutated) J and J_a dictionaries.
    """

    if J_a_objectives is None:
        J_a_objectives = {}

    # Snapshot per-SKU Gini at t=0 (start) and every 1800 timesteps.
    num_skus = S.get_num_skus()
    if num_skus and (t == 0 or t % 1800 == 0):
        label = "start" if t == 0 else None
        S.record_sku_gini_snapshot(G.warehouse, num_skus, G.get_aisle_locations(), t, label)

    # Update state of robots
    for agent in Rs.agents:
        # Agents mid pick/place service wait in place (no movement) until their
        # service counter expires.
        if agent.status in (STATUS_PICKING, STATUS_PLACING):
            agent.blocked_ticks = 0
            continue
        # If robot sequence is stationary, leave the robot in place (wait action)
        if len(agent.path_sequence) == 0:
            agent.blocked_ticks = 0
            continue

        # Reactive collision avoidance (Ethan's task_queue approach): never step
        # into a cell another agent currently occupies. The collision-free
        # planner does not model the multi-tick pick/place stall, and on a PBS
        # failure agents fall back to a stale, uncoordinated plan -- either way a
        # follower would otherwise walk straight into a neighbour. Blocked agents
        # wait and count ticks; a sustained block escalates to a replan via
        # ``router.needs_path_plan``.
        next_state = agent.path_sequence[0]
        if _is_cell_occupied_by_other_agent(Rs.agents, next_state, agent):
            agent.blocked_ticks += 1
            continue
        agent.blocked_ticks = 0

        # Update Agent State and Graph Occupied States
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
        # --- Pick service in progress (pick/place delay enabled) ---
        if pick_place_time and agent.status == STATUS_PICKING:
            task = agent.task_sequence[0]
            task_id = task[0]
            S.update_actual_pickup_duration(task_id, S.get_actual_pickup_duration(task_id) + 1)
            agent.pick_place_counter -= 1
            if agent.pick_place_counter > 0:
                continue
            outcome = _attempt_pickup(agent, task, G, J, J_a, S)
            if outcome == "stale_shuffle":
                J_a.pop(task_id, None)
                agent.task_sequence.pop(0)
                agent.path_sequence = []
                agent.status = STATUS_TO_PICKUP if agent.task_sequence else STATUS_FREE
            elif outcome == "succeeded":
                S.add_completed_to_pickup_task_id(task_id)
                agent.status = STATUS_TO_DELIVERY
            else:
                # not_released / failed: revert to pickup and retry next tick.
                agent.status = STATUS_TO_PICKUP
            continue

        # --- Place service in progress (pick/place delay enabled) ---
        if pick_place_time and agent.status == STATUS_PLACING:
            task = agent.task_sequence[0]
            task_id = task[0]
            S.update_actual_duration(task_id, S.get_actual_duration(task_id) + 1)
            agent.pick_place_counter -= 1
            if agent.pick_place_counter > 0:
                continue
            if not _complete_delivery(agent, task, G, J, J_a, S, Rs, t, J_a_objectives=J_a_objectives, output_buffer=output_buffer):
                # Output buffer full (or transient race): revert to delivery and
                # retry later (re-entering the place wait next time). Flag the
                # agent as buffer-waiting for the waiting-at-buffer diagnostic.
                agent.waiting_at_buffer = True
                agent.status = STATUS_TO_DELIVERY
            else:
                agent.waiting_at_buffer = False
            continue

        if agent.status == STATUS_TO_PICKUP:
            if agent.state == agent.task_sequence[0][1]:
                task = agent.task_sequence[0]
                task_id = task[0]
                if pick_place_time:
                    # Begin the pick service wait; SKU is removed when it ends.
                    agent.status = STATUS_PICKING
                    agent.pick_place_counter = pick_place_duration
                    continue
                outcome = _attempt_pickup(agent, task, G, J, J_a, S)
                if outcome == "stale_shuffle":
                    J_a.pop(task_id, None)
                    agent.task_sequence.pop(0)
                    agent.path_sequence = []
                    agent.status = STATUS_TO_PICKUP if agent.task_sequence else STATUS_FREE
                    continue
                if outcome == "succeeded":
                    S.add_completed_to_pickup_task_id(task_id)
                    agent.status = STATUS_TO_DELIVERY
                # else: task not yet released OR SKU not yet at start_location;
                # keep status=1 and retry the pickup on the next tick.
        elif agent.status == STATUS_TO_DELIVERY:
            if agent.state == agent.task_sequence[0][2]:
                task = agent.task_sequence[0]
                task_id = task[0]
                inbound_task = J_a[task_id][4] if task_id in J_a else J[task_id][4]

                # Outbound + full buffer: wait at the driveway cell (backpressure)
                # rather than starting a place that can't complete. Flag the agent
                # as buffer-waiting for the waiting-at-buffer diagnostic.
                if inbound_task == TASK_TYPE_OUTBOUND and outbound_delivery_blocked(output_buffer):
                    S.record_outbound_buffer_placement_blocked(agent.id, t)
                    agent.waiting_at_buffer = True
                    continue
                if inbound_task == TASK_TYPE_OUTBOUND:
                    agent.waiting_at_buffer = False
                if pick_place_time:
                    # Begin the place service wait; delivery completes when it ends.
                    agent.status = STATUS_PLACING
                    agent.pick_place_counter = pick_place_duration
                    continue
                if not _complete_delivery(agent, task, G, J, J_a, S, Rs, t, J_a_objectives=J_a_objectives, output_buffer=output_buffer):
                    continue

    S.append_agents_waiting_at_buffer(sum(1 for agent in Rs.agents if agent.waiting_at_buffer))

    # Drain the shared output buffer by one tick and log its level.
    consumption_tick(output_buffer)
    if output_buffer is not None:
        S.log_output_buffer_level(t, output_buffer.level)

    return Rs, J, J_a
