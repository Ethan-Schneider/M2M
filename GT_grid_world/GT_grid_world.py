import time
import os
import numpy as np
import argparse
from typing import List, Optional

from src import graph, simulate, task_allocation, case_request_generator, import_schedule, router, agent
from src.task_allocation_algorithms.local_repair.branch_and_bound_V2 import BnB
from src.task_allocation_algorithms.local_repair.exact_repair import HA_exact_repair
from src.task_allocation_algorithms.repair_detection.backtracking import detect_backtracking
from src.task_allocation_algorithms.repair_detection.duration_difference import duration_difference
from src.task_allocation_algorithms.repair_detection.sliding_window_progress import sliding_window_progress
from src.analysis import visualize, statistics
from src.reallocation_tasks.generate_reallocation_tasks import (
    generate_reallocation_tasks,
    generate_crm2m_reallocation_tasks,
    merge_reallocation_tasks_into_J,
)
from src.reallocation_tasks.jr_consumer import (
    add_reallocation_tasks_to_J_a,
    prune_uncommitted_rearrangements,
    REARRANGEMENT_TASK_ID_BASE,
)
from src.reallocation_tasks.optimal_insertion_gurobi import solve_insertion
from src.reallocation_tasks.fast_optimal_insertion import solve_insertion_fast
from src.task_allocation_algorithms.initial_solutions.construct_cost_elements import (
    CRM2M_DEFAULT_LAMBDA,
    CRM2M_DEFAULT_DETOUR_CUTOFF,
    CRM2M_DEFAULT_RETURN_MARGIN,
    compute_ambient_slack,
    committed_shuffle_terms,
)
from src.output_buffer import OutputBuffer, QUEUE_DEADLINE, tasks_per_min_to_per_tick

# irM2M post-allocation insertion methods (Ethan's task_queue). crM2M runs
# instead via ``enable_rearrangement`` (concatenated, buffer-aware) below.
REALLOCATION_TASK_METHODS = ("none", "simultaneous", "insertion", "fast_insertion")

def execute(S : statistics.Stats, map : str, Rs : agent.AgentLoader, G : graph.Graph, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            max_task_number : int = 20,
            initial_task_assignment_strategy : str = "lns",
            improvement_task_assignment_strategy : str = "py_lns",
            path_planning_strategy : str = "ecbs", time_limit : int = 999999,
            cost_calculation_method : str = "manhattan",
            removal_operator : str = "worst", repair_operator : str = "greedy",
            acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
            deadline_generation_method: str = "constant",
            deadline_offset: float = 30,
            output_intermediate_data: bool = False,
            intermediate_data_interval: int = 1800,
            base_cost_weight: float = 1.0,
            deadline_weight: float = 0.0,
            sku_distribution_weight: float = 0.0,
            agent_unallocated_penalty: float = 0.0,
            solution_repair_detection_function: str = "none",
            solution_repair_function: str = "none",
            initial_inventory : float = 25.0,
            use_precomputed_schedule: bool = False,
            schedule: np.ndarray = None,
            run_until_schedule_complete: bool = False,
            use_precomputed_queue: bool = False,
            queue: np.ndarray = None,
            run_until_queue_complete: bool = False,
            W: int = 300,
            B: int = 60,
            enable_rearrangement: bool = False,
            crm2m_lambda: float = CRM2M_DEFAULT_LAMBDA,
            crm2m_detour_cutoff: float = CRM2M_DEFAULT_DETOUR_CUTOFF,
            crm2m_return_margin: float = CRM2M_DEFAULT_RETURN_MARGIN,
            lambda_: float = 1.5,
            reallocation_task_method: str = "none",
            pick_place_time: bool = False,
            pick_place_duration: int = simulate.DEFAULT_PICK_PLACE_DURATION,
            buffer_capacity_k: int = 0,
            buffer_consumption_rate: float = 70.0,
            queue_release_window: int = 60,
            log_buffer_predictions: bool = False) -> int:
    if reallocation_task_method not in REALLOCATION_TASK_METHODS:
        raise ValueError(
            f"Unknown reallocation_task_method {reallocation_task_method!r}; "
            f"expected one of {REALLOCATION_TASK_METHODS}"
        )
    # Initilize empty dict of tasks, task is defined as (id: (start_loc, goal_loc, deadline, sku_id, type))
    J = {}
    # Shared outbound output buffer (capacity model). Disabled when K <= 0 so
    # default behaviour is unchanged. When enabled it throttles *outbound*
    # throughput at the consumption rate; inbound/shuffle are unaffected.
    output_buffer = None
    if buffer_capacity_k > 0:
        output_buffer = OutputBuffer(
            capacity=buffer_capacity_k,
            consumption_rate_per_min=buffer_consumption_rate,
        )
    # crM2M: rearrangement (shuffle) tasks live in their own pool, kept separate
    # from the persistent real-task pool ``J``. They are scored alongside real
    # tasks via a transient union ``{**J, **J_a}`` handed to the allocator, but
    # have their own lifecycle (expiry / pruning) and completion track.
    J_a = {}

    # Initilize empty dict of rearrangement tasks defined as (id: (start_loc, goal_loc, deadline, sku_id, type))
    # Rearrangement tasks only added when committed to an agent's task sequence
    J_a = {}
    J_a_objectives = {}

    last_task_id = 0
    # crM2M: monotonically increasing id allocator for rearrangement tasks, kept
    # above REARRANGEMENT_TASK_ID_BASE so shuffle ids never collide with real ones.
    next_rearrangement_task_id = REARRANGEMENT_TASK_ID_BASE
    # irM2M (insertion): separate id counter for MILP-inserted reallocation tasks.
    last_rearrangement_task_id = 100000
    # crM2M reads the upcoming queue to flag future demand; warn once if
    # rearrangement was requested without a queue to look ahead into.
    if enable_rearrangement and not use_precomputed_queue:
        print("[WARN] enable_rearrangement requires --use-precomputed-queue; "
              "no rearrangement tasks will be generated.")
    total_schedule_tasks = schedule.shape[0] if use_precomputed_schedule and schedule is not None else 0
    total_queue_tasks = queue.shape[0] if use_precomputed_queue and queue is not None else 0
    deferred_queue: List[List[int]] = []
    # TA-Hybrid materialises the FULL task pool once at t=0 to run TSP
    # (paper Section 3 + materialize_schedule). ``schedule`` itself is
    # mutated by ``add_tasks_from_schedule`` -- released rows are removed
    # in place -- so we hold an immutable snapshot of the original
    # schedule and pass *that* to TaskAllocation for the ta_hybrid
    # branch. Other allocators keep using the live (shrinking) schedule.
    original_schedule = (
        schedule.copy() if use_precomputed_schedule and schedule is not None else None
    )
    
    global_tik = time.time()
    t = 0
    allow_reallocation_tasks = True

    max_task_number = 0
    while True:
        if not run_until_schedule_complete and not run_until_queue_complete and t >= T:
            break

        print(f"Number of tasks in system: {len(J)}")
        if use_precomputed_schedule:
            print(f"Number of tasks remaining in schedule: {schedule.shape[0]}")
        if use_precomputed_queue:
            print(f"Number of tasks remaining in queue: {queue.shape[0]}")
            print(f"Number of deferred tasks: {len(deferred_queue)}")

        if t == 300:
            allow_reallocation_tasks = False

        if t%500 == 0 and t > 0:
            if max_task_number == 0:
                max_task_number = 20
                allow_reallocation_tasks = False
            else:
                max_task_number = 0
                allow_reallocation_tasks = True

        print("============================= T : " + str(t) + "=============================")
        # Check if new tasks need to be generated
        # Update Buffer 
        # B.add(Rs.get_agent_states(), t)
        
        print("=============================" + "Task Generation"+ "=============================")
        if use_precomputed_queue:
            tik = time.time()
            # Demand-driven release (Ethan's task_queue): pull at most a
            # frequency-sized batch from the queue, gated by
            # ``len(J) < max_task_number``, parking un-placeable rows in
            # ``deferred_queue`` for retry. This is the backpressure that
            # makes arrival "on a needed basis" rather than a fixed burst.
            J, __, __, queue, deferred_queue, last_task_id = import_schedule.add_tasks_from_queue(
                t,
                queue,
                deferred_queue,
                J,
                S,
                G,
                last_task_id,
                max_task_number=max_task_number,
                frequency=frequency,
                deadline_generation_method=deadline_generation_method,
                deadline_offset=deadline_offset,
            )
            tok = time.time()
            S.add_total_CRG_time(tok - tik)
        elif use_precomputed_schedule:
            tik = time.time()
            # TA-Hybrid pre-commits to deterministic (lowest-(row, col))
            # inbound driveway pickup cells in its task-assignment stage
            # via ``materialize_schedule``. Force the schedule loader to
            # use the same rule so the cells in ``agent.task_sequence``
            # actually match the cells the simulator populates with the
            # SKU at release time. Other allocators keep the legacy
            # random selection (default ``deterministic=False``).
            _deterministic_release = (
                improvement_task_assignment_strategy == "ta_hybrid"
            )
            J, __, __, schedule, last_task_id = import_schedule.add_tasks_from_schedule(
                t,
                schedule,
                J,
                S,
                G,
                last_task_id,
                deterministic=_deterministic_release,
            )
            tok = time.time()
            S.add_total_CRG_time(tok - tik)
        elif t%frequency == 0:
            if len(J) < max_task_number:
                if frequency >= 1.: 
                    N = 1
                elif frequency < 1.: 
                    N = frequency**-1
                else:
                    N = 0
                # Generate new tasks
                tik = time.time()

                J, last_task_id, __, __ = case_request_generator.CRG(S, t, J, G, Rs, N, 
                                                                                                inbound_to_outbound_ratio, last_task_id, max_task_number,
                                                                                                  G.warehouse, case_request_strategy, 
                                                                                                  deadline_generation_method, deadline_offset, improvement_task_assignment_strategy,
                                                                                                  initial_inventory)
                tok = time.time()
                S.add_total_CRG_time(tok-tik)

        # crM2M (roadmap 4.2): look-ahead-driven proactive rearrangement.
        # Generate fresh candidates from the queue look-ahead and stage them in the
        # separate ``J_a`` pool. They are scored alongside real tasks by the union
        # dict passed to TaskAllocation below (the concatenated approach), not
        # merged into ``J``. Shuffles carry no time deadline in the queue model, so
        # there is no clock-based expiry -- uncommitted candidates are pruned and
        # regenerated each tick instead.
        if enable_rearrangement and use_precomputed_queue and queue is not None:
            print("=============================" + "Reallocation Tasks (crM2M)" + "=============================")
            tik = time.time()
            Ta = generate_crm2m_reallocation_tasks(queue, J, G, Rs, B, W, t)
            next_rearrangement_task_id = add_reallocation_tasks_to_J_a(
                Ta, J_a, G, next_rearrangement_task_id
            )
            S.log_reallocation_tasks_generated(t, len(Ta))
            tok = time.time()
            S.log_reallocation_generation_time(t, tok - tik)
            print(f"Number of reallocation tasks generated: {len(Ta)} ({tok - tik:.4f}s)")

        tik = time.time()

        print("=============================" + "Task Allocation"+ "=============================")
        if improvement_task_assignment_strategy == "hbh_mla_star" and J:
            resolve_open_task_locations(J, G, Rs)
        # if len(J) > 0:
        #     print(f"Printout a task in the system: {J[list(J.keys())[0]]}")
        # Check if all tasks are allocated, if so, skip
        total = 0
        for agent in Rs.agents:
            total += len(agent.task_sequence)

        # crM2M: hand the allocator a transient union of real tasks and staged
        # shuffles so the cost cube scores both in one concatenated pass. ``J``
        # and ``J_a`` stay the canonical stores (TaskAllocation never mutates the
        # dict it is given -- it only reads it and writes agent.task_sequence).
        J_for_alloc = {**J, **J_a} if enable_rearrangement else J

        if len(J_a) > 0 or total < max_task_number or improvement_task_assignment_strategy == "ta_hybrid":
            # TA-Hybrid must be called every tick (not gated on task
            # backlog) so the outer driver can detect group transitions
            # and replan even when no new tasks are unallocated. Other
            # allocators keep the legacy "skip when everything is
            # already assigned" optimisation.
            # For ta_hybrid we pass the ORIGINAL (un-consumed) schedule
            # so the driver's TSP sees the full task pool and produces
            # schedule_row_index values that line up with the task ids
            # ``add_tasks_from_schedule`` assigns over time.
            schedule_for_alloc = (
                original_schedule
                if improvement_task_assignment_strategy == "ta_hybrid"
                else schedule
            )
            print(f"Attempting to allocate tasks")

            # Buffer-aware crM2M: sample the ambient slack scalar from the live
            # output buffer and derive the shuffle pick+place service time. A
            # shuffle incurs one pick (at s_p) plus one place (at d_q), hence
            # 2 x pick_place_duration. Both default to 0.0 when the buffer /
            # pick-place timing are disabled, so the rearrangement utility
            # reduces exactly to the original ``b - lambda * Delta``.
            crm2m_slack = 0.0
            crm2m_cpp = 2.0 * pick_place_duration if pick_place_time else 0.0
            if output_buffer is not None:
                # alpha := outbound arrival rate into the buffer (items/tick).
                # Preferred: read it from the arrival/deadline timestamp column of
                # the look-ahead window (Ethan's task_queue format) via
                # ``estimate_arrival_rate``, which decouples alpha from the drain
                # rate mu. Fall back to the outbound-share x mu heuristic only for
                # the timeless 2-column queue format (no timestamp column).
                arrival_per_tick = 0.0
                if use_precomputed_queue and queue is not None and queue.size > 0:
                    if np.atleast_2d(queue).shape[1] > QUEUE_DEADLINE:
                        rate_per_min = OutputBuffer.estimate_arrival_rate(queue, W)
                        arrival_per_tick = tasks_per_min_to_per_tick(rate_per_min)
                    else:
                        outbound_fraction = OutputBuffer.estimate_outbound_fraction(queue, W)
                        arrival_per_tick = (
                            outbound_fraction * output_buffer.consumption_rate_per_tick
                        )
                crm2m_slack = compute_ambient_slack(
                    buffer_level=output_buffer.level,
                    capacity=output_buffer.capacity,
                    drain_per_tick=output_buffer.consumption_rate_per_tick,
                    arrival_per_tick=arrival_per_tick,
                    lambda_=crm2m_lambda,
                    detour_cutoff=crm2m_detour_cutoff,
                    c_pp=crm2m_cpp,
                )

            Rs, _, _ = task_allocation.TaskAllocation(S, G, Rs, J_for_alloc, initial_task_assignment_strategy, improvement_task_assignment_strategy, map, t, cost_calculation_method, removal_operator, repair_operator, acceptance_function, T_0, alpha, base_cost_weight, deadline_weight, sku_distribution_weight, agent_unallocated_penalty, schedule=schedule_for_alloc, crm2m_lambda=crm2m_lambda, crm2m_detour_cutoff=crm2m_detour_cutoff, crm2m_slack=crm2m_slack, crm2m_cpp=crm2m_cpp, crm2m_return_margin=crm2m_return_margin)

        tok = time.time()
        S.add_total_TA_time(tok-tik)

        # crM2M: drop staged shuffles no agent committed to this tick. M2M
        # re-decides trailing tasks every tick, so uncommitted shuffles are
        # regenerated fresh next tick from the look-ahead window rather than
        # lingering in J_a. Shuffles an agent is executing stay in a task
        # sequence and are kept so the simulator can complete them.
        if enable_rearrangement:
            prune_uncommitted_rearrangements(J_a, Rs)
            S.log_reallocation_tasks_committed(t, len(J_a))

            # Record the objective terms (benefit, actual detour, utility) for
            # every committed shuffle from the freshly-decided task sequences, so
            # each completed rearrangement can be charted with the exact numbers
            # the allocator scored it with. FIRST-write-wins: the utility that
            # justified the shuffle is the one at its *commit* tick (where it
            # cleared the U>0 gate at that tick's ambient slack). A shuffle stays
            # committed while in-flight, so overwriting each tick would instead
            # log a later tick's slack -- after the buffer has drained, slack -> 0
            # and the recomputed U goes spuriously negative. Recording only on the
            # first commit pins U to the decision that actually drove execution.
            if len(J_a) > 0:
                crm2m_home = Rs.agents[0].home
                already_scored = S.get_rearrangement_task_scores()
                for crm2m_agent in Rs.agents:
                    for seq_i, seq_task in enumerate(crm2m_agent.task_sequence):
                        shuffle_id = seq_task[0]
                        if shuffle_id not in J_a or shuffle_id in already_scored:
                            continue
                        prev_cell = (
                            crm2m_agent.task_sequence[seq_i - 1][2]
                            if seq_i > 0
                            else crm2m_agent.state
                        )
                        benefit, detour, utility = committed_shuffle_terms(
                            G,
                            prev_cell=prev_cell,
                            s_p=seq_task[1],
                            d_q=seq_task[2],
                            home=crm2m_home,
                            slack=crm2m_slack,
                            c_pp=crm2m_cpp,
                            lambda_=crm2m_lambda,
                            method=cost_calculation_method,
                            return_margin=crm2m_return_margin,
                        )
                        S.record_rearrangement_task_score(
                            shuffle_id, benefit, detour, utility
                        )

        # Check if any agent is allocated the same tasks
        for agent in Rs.agents:
            task_ids = set()
            for task_id in agent.task_sequence:
                if task_id in task_ids:
                    raise ValueError(f"Task {task_id} is allocated to multiple agents.")
                task_ids.add(task_id)

        # # Count the number of outbound and inbound tasks allocated to each agent
        # outbound_tasks_allocated = 0
        # inbound_tasks_allocated = 0
        # initial_outbound_tasks = 0
        # initial_inbound_tasks = 0
        # for agent in Rs.agents:
        #     for n, task in enumerate(agent.task_sequence):
        #         task_id = task[0]
        #         if task_id in outbound_tasks:
        #             outbound_tasks_allocated += 1
        #         elif task_id in inbound_tasks:
        #             inbound_tasks_allocated += 1
        #         if task_id in outbound_tasks and n == 0:
        #             initial_outbound_tasks += 1
        #         elif task_id in inbound_tasks and n == 0:
        #             initial_inbound_tasks += 1
        # print(f"Outbound tasks allocated: {outbound_tasks_allocated}")
        # print(f"Inbound tasks allocated: {inbound_tasks_allocated}")
        # print(f"Initial outbound tasks: {initial_outbound_tasks}")
        # print(f"Initial inbound tasks: {initial_inbound_tasks}")

        if (
            reallocation_task_method != "none"
            and use_precomputed_queue
            and queue is not None
        ):
            print("=============================" + "Reallocation Tasks" + "=============================")
            if allow_reallocation_tasks:
                tik = time.time()
                Ta = generate_reallocation_tasks(queue, J, G, Rs, B, W, t)
                generation_time = time.time() - tik
                S.log_reallocation_tasks_generated(t, len(Ta))
                S.log_reallocation_generation_time(t, generation_time)
                tok = time.time()
                print(f"Generate reallocation tasks time: {generation_time:.4f}s for {len(Ta)}")

                if reallocation_task_method in ("insertion", "fast_insertion"):
                    solve_insertion_fn = (
                        solve_insertion_fast if reallocation_task_method == "fast_insertion" else solve_insertion
                    )
                    tik = time.time()
                    Rs, J_a, num_chosen, num_binary_vars, construct_time, solve_time, new_objectives = solve_insertion_fn(
                        Ta,
                        Rs,
                        G,
                        J,
                        J_a,
                        output_buffer,
                        S=S,
                        lambda_=lambda_,
                        t=t,
                        next_rearrangement_task_id=last_rearrangement_task_id,
                        pick_place_time=pick_place_time,
                    )
                    J_a_objectives.update(new_objectives)
                    S.log_reallocation_tasks_chosen(t, num_chosen)
                    S.log_reallocation_milp_binary_vars(t, num_binary_vars)
                    S.log_reallocation_milp_construct_time(t, construct_time)
                    S.log_reallocation_milp_solve_time(t, solve_time)
                    if J_a:
                        last_rearrangement_task_id = max(J_a.keys())
                    tok = time.time()
                    print(
                        f"MILP insertion time: {tok - tik:.4f}s "
                        f"(construct: {construct_time:.4f}s, solve: {solve_time:.4f}s, "
                        f"{num_binary_vars} binary vars, {num_chosen} tasks chosen)"
                    )
                elif reallocation_task_method == "simultaneous":
                    tik = time.time()
                    last_task_id = merge_reallocation_tasks_into_J(Ta, J, G, last_task_id)
                    Rs, _, _ = task_allocation.TaskAllocation(
                        S,
                        G,
                        Rs,
                        J,
                        initial_task_assignment_strategy,
                        improvement_task_assignment_strategy,
                        map,
                        t,
                        cost_calculation_method,
                        removal_operator,
                        repair_operator,
                        acceptance_function,
                        T_0,
                        alpha,
                        base_cost_weight,
                        deadline_weight,
                        sku_distribution_weight,
                        agent_unallocated_penalty,
                    )
                    tok = time.time()
                    print(f"Simultaneous reallocation allocation time: {tok - tik}")

        S.record_buffer_prediction(t, G, Rs, J, J_a, output_buffer)

        print("=============================" +"Routing"+ "=============================")
        tik = time.time()
        
        # if t > 50:
        #     costs = []
        #     for i in range(100):
        #         for agent in Rs.agents:
        #             if agent.path_sequence == []:
        #                 Rs = router.pathPlan(map, Rs, path_planning_strategy, S)
        #                 break
        #         cost = 0
        #         for agent in Rs.agents:
        #             cost += len(agent.path_sequence)
        #         costs.append(cost)
        #     print(f"Routing costs over 100 iterations: {costs}")
        #     print(f"Number of different costs: {len(set(costs))}")
        #     exit()
        # else:
        #     for agent in Rs.agents:
        #         if agent.path_sequence == []:
        #             Rs = router.pathPlan(map, Rs, path_planning_strategy, S)
        #             break            
        
        # HBH+MLA* (Grenouilleau et al., ICAPS 2019) and TA-Hybrid
        # (Liu et al., AAMAS 2019) are both coupled allocator + path
        # planner methods: their TaskAllocation hooks already populate
        # ``agent.path_sequence`` for every agent they replanned, so
        # calling the external ECBS/PBS router here would clobber those
        # plans with a different (unreservation-aware) routing solution.
        # Skip the external router for these strategies and let the
        # native MAPF-aware paths drive the simulator.
        if improvement_task_assignment_strategy not in ("hbh_mla_star", "ta_hybrid"):
            # Replan when some agent needs its next route (empty plan) or has
            # been blocked in place too long (see router.needs_path_plan). On a
            # PBS failure the router now clears all plans so agents hold instead
            # of running stale, uncoordinated paths -- together with the reactive
            # move-guard in simulate this removes the aisle in/out oscillation
            # and drives collisions to ~0. pathPlan replans all agents in one
            # call, so a single trigger suffices.
            if router.needs_path_plan(Rs):
                Rs = router.pathPlan(map, Rs, path_planning_strategy, S)

        tok = time.time()
        S.add_total_PF_time(tok-tik)
        
        soc = 0
        for agent in Rs.agents:
            soc += len(agent.path_sequence)
        S.set_soc(soc)
        
        print("=============================" +"Task Reallocation"+ "=============================")
        # Identify which agents are backtracking, and need to reallocate their current task 

        if solution_repair_function != "none":
            
            detection_tik = time.time()
            
            if solution_repair_detection_function == "Backtracking":
                aisle_groups = detect_backtracking(Rs)
            elif solution_repair_detection_function == "Duration":
                aisle_groups = duration_difference(Rs, G)
            elif solution_repair_detection_function == "Progress":
                aisle_groups = sliding_window_progress(Rs, G)
            elif solution_repair_detection_function == "Ensamble":
                aisle_groups_bt = detect_backtracking(Rs)
                aisle_groups_du = duration_difference(Rs, G)
                aisle_groups_sw = sliding_window_progress(Rs, G)

                # Combine the aisle groups, remove duplicates
                aisle_groups = {}
                for group in [aisle_groups_bt, aisle_groups_du, aisle_groups_sw]:
                    for aisle, agents in group.items():
                        if aisle not in aisle_groups:
                            aisle_groups[aisle] = agents
                        else:
                            for agent in agents:
                                if agent not in aisle_groups[aisle]:
                                    aisle_groups[aisle].append(agent)
            else:
                raise ValueError(f"Unknown solution repair detection function: {solution_repair_detection_function}: Exiting ...")
                
            detection_tok = time.time()
                
            if aisle_groups:
                print(f"Aisle groups for reallocation: { {aisle: [a.id for a in group] for aisle, group in aisle_groups.items()} }")

                agents = [[a.id for a in group] for aisle, group in aisle_groups.items()]
                t_key = S.create_new_realloc_data(t)
                S.reallocation_data[t_key]["agents"] = []
                S.reallocation_data[t_key]["tasks"] = []
                S.reallocation_data[t_key]["prior_path_cost"] = []
                S.reallocation_data[t_key]["post_path_cost"] = [] 
                S.reallocation_data[t_key]["change_in_path_cost"] = []     
                S.reallocation_data[t_key]["path_planning_compute_time"] = []
                S.reallocation_data[t_key]["lower_bound_and_checks"] = []
                S.reallocation_data[t_key]["rejected_solution"] = []
                S.reallocation_data[t_key]["bnb_init_compute_time"] = []
                S.reallocation_data[t_key]["bnb_solve_time"] = []
                S.reallocation_data[t_key]["bnb_routing_time"] = []
                S.reallocation_data[t_key]["detection_computation_time"] = np.abs(detection_tok - detection_tik)

                S.reallocation_data[t_key]["prior_agent_path_cost"] = []
                S.reallocation_data[t_key]["post_agent_path_cost"] = []
                S.reallocation_data[t_key]["prior_group_path_cost"] = []
                S.reallocation_data[t_key]["post_group_path_cost"] = []
                repair_tik = time.time()
                for agent_group in agents:
                    temp_Rs = Rs.copy()
                    if np.abs(time.time() - repair_tik) >= 1.0:
                        break
                    
                    group_cost = 0
                    for agent_id in agent_group:
                        group_cost += len(temp_Rs.get_agent(agent_id).path_sequence)

                    # Skip if only one agent in group
                    if len(agent_group) <= 1:
                        continue

                    task_ids = []
                    for agent_id in agent_group:
                        task = temp_Rs.get_agent(agent_id).task_sequence[0][0]
                        task_ids.append(task)
                    S.reallocation_data[t_key]["tasks"].append(task_ids)

                    prior_cost = 0
                    for agent in temp_Rs.agents:
                        prior_cost += len(agent.path_sequence)
                    S.reallocation_data[t_key]["prior_path_cost"].append(prior_cost)
                    S.reallocation_data[t_key]["prior_agent_path_cost"].append(temp_Rs.get_agent(agent_group[0]).path_sequence.__len__())

                    prior_group_cost  = 0
                    for agent_id in agent_group:
                        prior_group_cost += len(temp_Rs.get_agent(agent_id).path_sequence)
                    S.reallocation_data[t_key]["prior_group_path_cost"].append(prior_group_cost)

                    if solution_repair_function == "HA":
                        if len(agent_group) >= 5:
                            continue

                        HA_agent_group = agent_group.copy()
                        S.reallocation_data[t_key]["agents"].append(HA_agent_group)
                        # temp_Rs = Rs.copy()
                        temp_Rs = HA_exact_repair(temp_Rs, G, S, J, HA_agent_group)
                    elif solution_repair_function == "BnB":
                        if len(agent_group) >= 4:
                            continue
                            
                        S.reallocation_data[t_key]["agents"].append(agent_group)
                        
                        init_tik = time.time()
                        bnb = BnB(temp_Rs, G, J, S, agent_group, map, t_key, prior_cost)
                        S.reallocation_data[t_key]["bnb_init_compute_time"].append(np.abs(time.time() - init_tik))
                        
                        solve_tik = time.time()
                        temp_Rs = bnb.solve()
                        S.reallocation_data[t_key]["bnb_solve_time"].append(np.abs(time.time() - solve_tik))
                    else:
                        print(f"Unknown solution repair function: {solution_repair_function}: Exiting ...")
                        exit()
                        

                    routing_tik = time.time()
                    temp_Rs = router.pathPlan(map, temp_Rs, path_planning_strategy, S)
                    S.reallocation_data[t_key]["bnb_routing_time"].append(np.abs(time.time() - routing_tik))

                    post_cost = 0

                    for agent in temp_Rs.agents:
                        post_cost += len(agent.path_sequence)
                        
                    S.reallocation_data[t_key]["post_agent_path_cost"].append(temp_Rs.get_agent(agent_group[0]).path_sequence.__len__())
                    
                    post_group_cost = 0
                    for agent_id in agent_group:
                        post_group_cost += len(temp_Rs.get_agent(agent_id).path_sequence)
                    S.reallocation_data[t_key]["post_group_path_cost"].append(post_group_cost)

                    post_goals = []
                    for agent_id in agent_group:
                        if temp_Rs.get_agent(agent_id).status == 1:
                            post_goals.append(temp_Rs.get_agent(agent_id).task_sequence[0][1])
                        elif temp_Rs.get_agent(agent_id).status == 2:
                            post_goals.append(temp_Rs.get_agent(agent_id).task_sequence[0][2])

                    S.reallocation_data[t_key]["post_path_cost"].append(post_cost)
                    S.reallocation_data[t_key]["change_in_path_cost"].append(post_cost - prior_cost)
                    
                    if post_cost < prior_cost:
                        Rs = temp_Rs.copy()
                        S.reallocation_data[t_key]["rejected_solution"].append(False)
                    else:
                        S.reallocation_data[t_key]["rejected_solution"].append(True)

                    # print(f"final assignment : {assignment} with cost: {cost}")
                    
                print(f"Repaired Solution")
                S.reallocation_data[t_key]["computation_time"] = np.abs(time.time() - repair_tik)
                print(S.reallocation_data[t_key]["tasks"])

        print("=============================" +"Taking Step"+ "=============================")
        tik = time.time()
        Rs, J, J_a = simulate.simulate(
            S, G, Rs, J, J_a, map, t,
            J_a_objectives=J_a_objectives,
            pick_place_time=pick_place_time,
            pick_place_duration=pick_place_duration,
            output_buffer=output_buffer,
        )
        tok = time.time()
        S.add_total_SIM_time(tok-tik)
        
        # Log the number of tasks in the system
        S.append_tasks_in_system(len(J))
        
        # Log warehouse inventory state
        S.append_warehouse_inventory_state(G.warehouse, G.get_aisle_locations())
        
        # Log driveway inventory state
        S.append_driveway_inventory_state(G.driveway)
        
        # Log SKU inventory state
        S.append_sku_inventory_state(G.warehouse, G.driveway, G.warehouse.get_all_skus().__len__())
        
        # Log SKU centroids per timestep
        S.append_sku_centroids(G.warehouse, G.warehouse.get_all_skus().__len__())
        
        # Log SKU locations per timestep
        S.append_sku_locations(G.warehouse, G.driveway, G.warehouse.get_all_skus().__len__())

        # Roadmap 1.7: per-timestep SKU Spread metric (hierarchical entropy
        # weighted by per-SKU count, clustered by aisle column).
        S.append_sku_spread(G.warehouse, G.warehouse.get_all_skus().__len__(), G.get_aisle_locations())
        
        # Record agent statuses and goal locations for this timestep
        S.add_agent_statuses_and_goals(Rs)
        
        S.compute_unallocated_agents(Rs)

        global_tok = time.time()

        # Note: the historical wall-clock guard that early-returned from
        # this loop when ``global_tok - global_tik >= time_limit`` has
        # been removed. Per-condition runtime is now controlled solely by
        # ``--time-horizon`` (set by ``run_baselines.sh``); each run
        # always executes exactly T simulated ticks regardless of how
        # long that takes in wall-clock seconds. Mixing a wall-clock cap
        # with ``--time-horizon`` produced runs of varying simulated
        # lengths across (method, density, robots) cells, which made the
        # baseline visuals impossible to compare apples-to-apples. The
        # ``time_limit`` parameter and ``--time-limit`` CLI flag are
        # preserved (defaulting to a sentinel ``999999``) so external
        # scripts that still pass them are not broken; it just no longer
        # does anything.

        # Output intermediate data if enabled
        if output_intermediate_data and t % intermediate_data_interval == 0:
            intermediate_output_file = S.get_output_file().replace(".json", f"_{t}.json")
            S.save_data(intermediate_output_file)

        if (
            run_until_queue_complete
            and use_precomputed_queue
            and import_schedule.queue_tasks_finished(
                queue, deferred_queue, J, S, total_queue_tasks
            )
        ):
            print(f"All {total_queue_tasks} queue tasks completed at t={t}")
            print(f"Current tasks in system: {len(J)}")
            print(f"Current tasks in queue: {queue.shape[0]}")
            print(f"Current deferred tasks: {len(deferred_queue)}")
            t += 1
            break

        if (
            run_until_queue_complete
            and use_precomputed_queue
            and import_schedule.queue_tasks_finished(
                queue, deferred_queue, J, S, total_queue_tasks
            )
        ):
            print(f"All {total_queue_tasks} queue tasks completed at t={t}")
            t += 1
            break

        t += 1

    return t

def main(seed: int, num_robots: int, T: int, max_number_tasks: int, 
         task_generation_strategy: str, initial_task_assignment_strategy: str, improvement_task_assignment_strategy: str,
         path_planning_strategy: str, map_name: str, time_limit: int = 999999,
         visualize_output: bool = False, initial_inventory: float = 25.0, 
         frequency: float = 1.0, inbound_outbound_ratio: float = 1.0,
         output_graphs: bool = False, num_skus: int = 10,
         weight_init_method: str = "random",
         cost_calculation_method: str = "manhattan",
         removal_operator: str = "worst", repair_operator: str = "greedy",
         acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
         deadline_generation_method: str = "constant",
         deadline_offset: float = 30,
         output_intermediate_data: bool = False,
         intermediate_data_interval: int = 1800,
         base_cost_weight: float = 1.0,
         deadline_weight: float = 0.0,
         sku_distribution_weight: float = 0.0,
         agent_unallocated_penalty: float = 0.0,
         solution_repair_detection_function: str = "none",
         solution_repair_function: str = "none",
         use_precomputed_schedule: bool = False,
         schedule_file: str = None,
         initial_inventory_file: str = None,
         run_until_schedule_complete: bool = False,
         use_precomputed_queue: bool = False,
         queue_file: str = None,
         run_until_queue_complete: bool = False,
         W: int = 300,
         B: int = 60,
         enable_rearrangement: bool = False,
         crm2m_lambda: float = CRM2M_DEFAULT_LAMBDA,
         crm2m_detour_cutoff: float = CRM2M_DEFAULT_DETOUR_CUTOFF,
         crm2m_return_margin: float = CRM2M_DEFAULT_RETURN_MARGIN,
         lambda_: float = 1.5,
         reallocation_task_method: str = "none",
         pick_place_time: bool = False,
         pick_place_duration: int = simulate.DEFAULT_PICK_PLACE_DURATION,
         buffer_capacity_k: int = 0,
         buffer_consumption_rate: float = 70.0,
         queue_release_window: int = 60,
         log_buffer_predictions: bool = False,
         output_file_override: str = None) -> None:
    """
    Run a single instance of the simulation with specified parameters.
    
    Args:
        seed: Random seed for reproducibility
        num_robots: Number of robots in the simulation
        T: Time horizon
        max_number_tasks: Maximum number of tasks
        task_generation_strategy: Strategy for generating tasks
        initial_task_assignment_strategy: Strategy for assigning tasks
        improvement_task_assignment_strategy: Strategy for improving tasks
        path_planning_strategy: Strategy for path planning
        map_name: Name of the map to use
        time_limit: Maximum runtime in seconds
        visualize_output: Whether to generate visualization
        initial_inventory: Initial inventory fill percentage (0.0 to 100.0)
        frequency: Frequency of task generation
        inbound_outbound_ratio: Ratio of inbound to outbound tasks
        output_graphs: Whether to output analysis graphs
        num_skus: Number of unique SKUs in the warehouse
        weight_init_method: Method for initializing SKU weights ("random" or "uniform")
        cost_calculation_method: Method for calculating cost ("manhattan" or "shortest_path")
        removal_operator: Removal operator for task allocation
        repair_operator: Repair operator for task allocation
        acceptance_function: Acceptance function for LNS (greedy or simulated_annealing)
        T_0: Initial temperature for simulated annealing
        alpha: Temperature decay rate for simulated annealing
        deadline_generation_method: Method for generating task deadlines (e.g., constant, normal, bimodal, etc.)
        deadline_offset: Offset for task deadlines
        output_intermediate_data: Whether to output intermediate data
        intermediate_data_interval: Interval for outputting intermediate data
        base_cost_weight: Weight for base cost
        deadline_weight: Weight for deadline
        sku_distribution_weight: Weight for sku distribution
        agent_unallocated_penalty: Weight for agent unallocated penalty
        solution_repair_detection_function: Function to detect solution repair needs
        solution_repair_function: Function to repair solution post allocation and motion planning
    """
    np.random.seed(seed)

    if reallocation_task_method not in REALLOCATION_TASK_METHODS:
        raise ValueError(
            f"Unknown reallocation_task_method {reallocation_task_method!r}; "
            f"expected one of {REALLOCATION_TASK_METHODS}"
        )

    if use_precomputed_queue:
        if not queue_file or not initial_inventory_file:
            raise ValueError(
                "--use-precomputed-queue requires both --queue-file and --initial-inventory-file"
            )

    if run_until_queue_complete and not use_precomputed_queue:
        raise ValueError(
            "--run-until-queue-complete requires --use-precomputed-queue"
        )

    if use_precomputed_schedule and use_precomputed_queue:
        raise ValueError(
            "--use-precomputed-schedule and --use-precomputed-queue are mutually "
            "exclusive (a run is driven by either a schedule or a queue)."
        )

    if use_precomputed_queue:
        if not queue_file or not initial_inventory_file:
            raise ValueError(
                "--use-precomputed-queue requires both --queue-file and --initial-inventory-file"
            )

    if run_until_queue_complete and not use_precomputed_queue:
        raise ValueError(
            "--run-until-queue-complete requires --use-precomputed-queue"
        )

    if enable_rearrangement and not use_precomputed_queue:
        raise ValueError(
            "--enable-rearrangement requires --use-precomputed-queue (the queue "
            "look-ahead is the source of future demand for shuffle candidates)."
        )

    if improvement_task_assignment_strategy == "ta_hybrid":
        if not use_precomputed_schedule:
            raise ValueError(
                "--improvement-task-assign-strategy ta_hybrid is an "
                "offline algorithm and requires --use-precomputed-schedule."
            )
    
    stripped_map_name = map_name.split("/")[-1].replace(".json", "")
    effective_task_generation_strategy = (
        "precomputed_queue" if use_precomputed_queue else task_generation_strategy
    )
    queue_stem = (
        os.path.splitext(os.path.basename(queue_file))[0]
        if queue_file
        else "none"
    )

    output_file = (
        f"data/raw_data/{T}_{effective_task_generation_strategy}_"
        f"{reallocation_task_method}_{lambda_}_{queue_stem}_"
        f"{initial_inventory}_{initial_task_assignment_strategy}_"
        f"{improvement_task_assignment_strategy}_{path_planning_strategy}_"
        f"{stripped_map_name}_{num_robots}_{max_number_tasks}_"
        f"{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_"
        f"{solution_repair_detection_function}_{solution_repair_function}_{seed}.json"
    )
    # An explicit override lets otherwise-identical configs (e.g. M2M vs crM2M,
    # which share the py_lns strategy and thus the auto-name) run concurrently
    # without clobbering each other's output / intermediate files.
    if output_file_override:
        output_file = output_file_override
    buffer_file = f"data/buffer_data/{T}_{effective_task_generation_strategy}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}_{path_planning_strategy}_{stripped_map_name}_{num_robots}_{max_number_tasks}_{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_{seed}"
    
    # B = buffer.Buffer(80, buffer_file)
    S = statistics.Stats(
        num_robots=num_robots,
        simulation_time=T,
        output_file=output_file,
        map_name=map_name,
        cost_calculation_method=cost_calculation_method,
        seed=seed,
        max_tasks=max_number_tasks,
        task_generation_strategy=effective_task_generation_strategy,
        initial_task_assignment_strategy=initial_task_assignment_strategy,
        improvement_task_assignment_strategy=improvement_task_assignment_strategy,
        path_planning_strategy=path_planning_strategy,
        time_limit=time_limit,
        visualize_output=visualize_output,
        initial_inventory=initial_inventory,
        frequency=frequency,
        inbound_outbound_ratio=inbound_outbound_ratio,
        output_graphs=output_graphs,
        num_skus=num_skus,
        weight_init_method=weight_init_method,
        removal_operator=removal_operator,
        repair_operator=repair_operator,
        acceptance_function=acceptance_function,
        T_0=T_0,
        alpha=alpha,
        deadline_generation_method=deadline_generation_method,
        deadline_offset=deadline_offset,
        output_intermediate_data=output_intermediate_data,
        intermediate_data_interval=intermediate_data_interval,
        base_cost_weight=base_cost_weight,
        deadline_weight=deadline_weight,
        sku_distribution_weight=sku_distribution_weight,
        agent_unallocated_penalty=agent_unallocated_penalty,
        solution_repair_detection_function=solution_repair_detection_function,
        solution_repair_function=solution_repair_function,
        schedule_name=queue_stem,
        W=W,
        B=B,
        lambda_=lambda_,
        reallocation_task_method=reallocation_task_method,
        queue_release_window=queue_release_window,
        pick_place_time=pick_place_time,
        buffer_capacity_k=buffer_capacity_k,
        buffer_consumption_rate=buffer_consumption_rate,
        use_precomputed_queue=use_precomputed_queue,
        queue_file=queue_file,
        initial_inventory_file=initial_inventory_file,
        run_until_queue_complete=run_until_queue_complete,
        log_buffer_predictions=log_buffer_predictions,
    )
    schedule = None
    queue = None
    if use_precomputed_schedule:
        schedule = import_schedule.import_schedule(schedule_file)
        G = graph.Graph(num_robots, map_name, 0.0, num_skus, weight_init_method)
        import_schedule.load_initial_inventory(initial_inventory_file, G)
        print(f"Loaded precomputed schedule ({schedule.shape[0]} tasks) and initial inventory from file")
    elif use_precomputed_queue:
        queue = import_schedule.import_queue(queue_file)
        G = graph.Graph(num_robots, map_name, 0.0, num_skus, weight_init_method)
        import_schedule.load_initial_inventory(initial_inventory_file, G)
        print(f"Loaded precomputed queue ({queue.shape[0]} tasks) and initial inventory from file")
    else:
        G = graph.Graph(num_robots, map_name, initial_inventory, num_skus, weight_init_method)

    # print(f"Number of full locations in warehouse: {len(G.warehouse.get_full_locations())}")
    # exit()

    # Initialize state of robots (robot_id, state)
    robots = []
    for robot_id, location in enumerate(G.get_all_occupied()):
        robots.append(agent.Agent(robot_id, location))
    
    Rs = agent.AgentLoader(robots)
    
    init_locations = [agent.state for agent in Rs.agents]
    S.add_paths(init_locations)
    
    tik = time.time()
    # Execute online algorithm
    simulated_timesteps = execute(S, map_name, Rs, G, frequency, inbound_outbound_ratio, T, 
            case_request_strategy=task_generation_strategy, 
            max_task_number=max_number_tasks, 
            initial_task_assignment_strategy=initial_task_assignment_strategy, 
            improvement_task_assignment_strategy=improvement_task_assignment_strategy, 
            path_planning_strategy=path_planning_strategy, 
            time_limit=time_limit,
            cost_calculation_method=cost_calculation_method,
            removal_operator=removal_operator,
            repair_operator=repair_operator,
            acceptance_function=acceptance_function,
            T_0=T_0,
            alpha=alpha,
            deadline_generation_method=deadline_generation_method,
            deadline_offset=deadline_offset,
            output_intermediate_data=output_intermediate_data,
            intermediate_data_interval=intermediate_data_interval,
            base_cost_weight=base_cost_weight,
            deadline_weight=deadline_weight,
            sku_distribution_weight=sku_distribution_weight,
            agent_unallocated_penalty=agent_unallocated_penalty,
            solution_repair_detection_function=solution_repair_detection_function,
            solution_repair_function=solution_repair_function,
            initial_inventory=initial_inventory,
            use_precomputed_schedule=use_precomputed_schedule,
            schedule=schedule,
            run_until_schedule_complete=run_until_schedule_complete,
            use_precomputed_queue=use_precomputed_queue,
            queue=queue,
            run_until_queue_complete=run_until_queue_complete,
            W=W,
            B=B,
            enable_rearrangement=enable_rearrangement,
            crm2m_lambda=crm2m_lambda,
            crm2m_detour_cutoff=crm2m_detour_cutoff,
            crm2m_return_margin=crm2m_return_margin,
            lambda_=lambda_,
            reallocation_task_method=reallocation_task_method,
            pick_place_time=pick_place_time,
            pick_place_duration=pick_place_duration,
            buffer_capacity_k=buffer_capacity_k,
            buffer_consumption_rate=buffer_consumption_rate,
            queue_release_window=queue_release_window,
    )
    if run_until_schedule_complete or run_until_queue_complete:
        S.set_simulation_time(simulated_timesteps)
        print(f"Simulated {simulated_timesteps} timesteps (run until task source complete)")
    tok = time.time()
    S.set_total_runtime(tok-tik)

    S.record_sku_gini_snapshot(
        G.warehouse,
        G.warehouse.get_all_skus().__len__(),
        G.get_aisle_locations(),
        simulated_timesteps,
        label="end",
    )

    S.save_data()
    
    folder_name = f"{T}_{effective_task_generation_strategy}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}_{path_planning_strategy}_{num_robots}_{max_number_tasks}_{deadline_generation_method}_{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_{seed}"
    if output_graphs:
        S.output_graphs(folder_name)
    
    if visualize_output:
        print("============================Visualizing Output============================")
        visualize.main((G.width, G.height), G.obstacles, S.return_full_paths(), 
                      f'data/videos/{path_planning_strategy}_{T}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}.mp4', 
                      speed=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run grid world simulation with specified parameters')
    
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--num-robots', type=int, required=True, help='Number of robots')
    parser.add_argument('--time-horizon', type=int, required=True, help='Time horizon T')
    parser.add_argument('--max-tasks', type=int, required=True, help='Maximum number of tasks')
    parser.add_argument('--task-gen-strategy', type=str, required=True, 
                       choices=['informed_uniform', 'uninformed_uniform', 'feedback_control'],
                       help='Task generation strategy')
    parser.add_argument('--initial-task-assign-strategy', type=str, required=True,
                       choices=['cost_matrix', 'random', 'fast_greedy', 'fast_FCF', 'fast_SCF'],
                       help='Task assignment strategy')
    parser.add_argument('--improvement-task-assign-strategy', type=str, required=True,
                       choices=['M2M', 'py_lns', 'c_lns', 'c_p_lns', 'c_rmca', 'hbh_mla_star', 'ta_hybrid', 'none'],
                       help='Task assignment strategy for improvement. '
                            '"hbh_mla_star" implements Grenouilleau et al. (ICAPS 2019) '
                            'HBH+MLA*: a coupled allocator + path planner that bypasses '
                            'the external ECBS/PBS routing call. '
                            '"ta_hybrid" implements Liu, Ma, Li, Koenig (AAMAS 2019) '
                            'TA-Hybrid: offline TSP-based task assignment paired with '
                            'ICBS (Group 1) + time-extended min-cost max-flow (Group 2) '
                            'path planning. Requires --use-precomputed-schedule.')
    parser.add_argument('--path-planning-strategy', type=str, required=True,
                       choices=['ecbs', 'pbs'],
                       help='Path planning strategy')
    parser.add_argument('--map', type=str, default='data/maps/symbotic_small',
                       help='Map file path')
    parser.add_argument('--time-limit', type=int, default=999999,
                       help='Maximum runtime in seconds')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization output')
    parser.add_argument('--initial-inventory', type=float, default=25.0,
                       help='Initial inventory fill percentage (0.0 to 100.0)')
    parser.add_argument('--frequency', type=float, default=1.0,
                       help='Frequency of task generation')
    parser.add_argument('--inbound-outbound-ratio', type=float, default=1.0,
                       help='Ratio of inbound to outbound tasks')
    parser.add_argument('--output-graphs', action='store_true',
                       help='Output analysis graphs')
    parser.add_argument('--num-skus', type=int, default=10,
                       help='Number of unique SKUs in the warehouse')
    parser.add_argument('--weight-init-method', type=str, default='random',
                       choices=['random', 'uniform'],
                       help='Method for initializing SKU weights')
    parser.add_argument('--cost-calculation-method', type=str, default='manhattan',
                       choices=['manhattan', 'shortest_path'],
                       help='Method for calculating cost')
    parser.add_argument('--removal-operator', type=str, default='worst',
                       choices=['worst', 'random', 'greedy', 'shaw'],
                       help='Removal operator for task allocation')
    parser.add_argument('--repair-operator', type=str, default='greedy',
                       choices=['greedy', 'random', 'worst', 'fast_SCF'],
                       help='Repair operator for task allocation')
    parser.add_argument('--acceptance-function', type=str, default='greedy', choices=['greedy', 'simulated_annealing'], help='Acceptance function for LNS (greedy or simulated_annealing)')
    parser.add_argument('--T-0', type=float, default=1.0, help='Initial temperature for simulated annealing')
    parser.add_argument('--alpha', type=float, default=0.99, help='Temperature decay rate for simulated annealing')
    parser.add_argument('--deadline-generation-method', type=str, default='constant',
                       help='Method for generating task deadlines (e.g., constant, normal, bimodal, etc.)', choices=['constant', 'normal', 'bimodal', 'none'])
    parser.add_argument('--deadline-offset', type=float, default=30, help='Offset for task deadlines')
    parser.add_argument('--output-intermediate-data', action='store_true',
                       help='Output intermediate data')
    parser.add_argument('--intermediate-data-interval', type=int, default=1800, help='Interval for outputting intermediate data')
    parser.add_argument('--base-cost-weight', type=float, default=1.0, help='Weight for base cost')
    parser.add_argument('--deadline-weight', type=float, default=0.0, help='Weight for deadline')
    parser.add_argument('--sku-distribution-weight', type=float, default=0.0, help='Weight for sku distribution')
    parser.add_argument('--agent-unallocated-penalty', type=float, default=0.0, help='Penalty for unallocated agents')
    parser.add_argument('--solution-repair-detection-function', type=str, default='none', help='Solution repair detection function')
    parser.add_argument('--solution-repair-function', type=str, default='none', help='Solution repair function')
    parser.add_argument('--pick-place-time', action='store_true',
                       help='When set, agents spend 5 seconds picking up and 5 seconds '
                            'placing items at pickup/delivery locations before inventory '
                            'is updated (status 3=picking, 4=placing).')
    parser.add_argument('--buffer-capacity-k', type=int, default=0,
                       help='Shared outbound output buffer capacity K (0 disables).')
    parser.add_argument('--buffer-consumption-rate', type=float, default=70.0,
                       help='Output buffer consumption rate in tasks/min (drains '
                            'rate/60 items per simulation tick).')
    parser.add_argument('--log-buffer-predictions', action='store_true',
                       help='Every 100 timesteps, forecast the shared output buffer '
                            'level for the next 300 timesteps (same prediction logic '
                            'used by the fast_insertion reallocation path) and record '
                            'it alongside the real observed buffer levels for that '
                            'window. Requires --buffer-capacity-k > 0.')
    parser.add_argument('--use-precomputed-queue', action='store_true',
                       help='Load tasks from a precomputed queue file and skip CRG. '
                            'Requires --queue-file and --initial-inventory-file.')
    parser.add_argument('--queue-file', type=str, default=None,
                       help='Path to precomputed queue text file (sku_id, task_type)')
    parser.add_argument('--initial-inventory-file', type=str, default=None,
                       help='Path to precomputed initial inventory text file (sku_id, row, col)')
    parser.add_argument('--use-precomputed-schedule', action='store_true',
                       help='Load tasks from a precomputed schedule file and skip CRG. '
                            'Requires --schedule-file and --initial-inventory-file.')
    parser.add_argument('--schedule-file', type=str, default=None,
                       help='Path to precomputed schedule text file (task_id, time, sku_id, type)')
    parser.add_argument('--run-until-schedule-complete', action='store_true',
                       help='Run until all precomputed schedule tasks are completed instead '
                            'of stopping at --time-horizon. Requires --use-precomputed-schedule.')
    parser.add_argument('--run-until-queue-complete', action='store_true',
                       help='Run until all precomputed queue tasks are completed instead '
                            'of stopping at --time-horizon. Requires --use-precomputed-queue.')
    parser.add_argument('--W', type=int, default=300,
                       help='Lookahead window end (queue index, exclusive). Reallocation '
                            'candidates are generated from outbound tasks in the remaining '
                            'queue slice queue[B:W]. Only used with --use-precomputed-queue.')
    parser.add_argument('--B', type=int, default=60,
                       help='Lookahead window start (queue index, inclusive). Skips the first '
                            'B upcoming queue tasks when scanning for reallocation candidates, '
                            'to avoid proactive moves for demand that is already imminent.')
    parser.add_argument('--queue-release-window', type=int, default=60,
                       help='Release precomputed queue tasks only when their '
                            'deadline is within this many timesteps of the '
                            'current time (deadline <= t + window).')
    parser.add_argument('--enable-rearrangement', action='store_true',
                       help='Enable crM2M concatenated proactive rearrangement: generate '
                            'look-ahead shuffle candidates into J_a and assign them alongside '
                            'real tasks. Requires --use-precomputed-queue.')
    parser.add_argument('--crm2m-lambda', type=float, default=CRM2M_DEFAULT_LAMBDA,
                       help='crM2M utility detour weight lambda in U = b - lambda*Delta '
                            '(lambda >= 1; larger penalizes detours more).')
    parser.add_argument('--crm2m-detour-cutoff', type=float, default=CRM2M_DEFAULT_DETOUR_CUTOFF,
                       help='crM2M weighted-detour cutoff: shuffles with lambda*Delta >= this '
                            'value are rejected (gated to +inf).')
    parser.add_argument('--crm2m-return-margin', type=float, default=CRM2M_DEFAULT_RETURN_MARGIN,
                       help='crM2M return margin (ticks) added to the effective delay in the '
                            'utility penalty only: a shuffle is credited as free only when '
                            'slack >= Delta + c_pp + margin, forcing it to finish with that much '
                            'spare buffer-time before a freed slot is re-consumed by other '
                            'agents\' outbound arrivals. 0 reproduces the pre-margin behaviour.')
    parser.add_argument(
        '--lambda_',
        type=float,
        default=1.5,
        help='Detour penalty weight for rearrangement insertion (irM2M)',
    )
    parser.add_argument(
        '--reallocation-task-method',
        type=str,
        default='none',
        choices=list(REALLOCATION_TASK_METHODS),
        help='irM2M task integration method: none, simultaneous, '
             'insertion (irM2M), or fast_insertion (numpy-vectorized MILP construction, '
             'same result as insertion but faster to build).',
    )
    parser.add_argument('--pick-place-duration', type=int, default=4,
                       help='Ticks spent at each pick and each place when --pick-place-time '
                            'is set (default 4).')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Explicit output JSON path, overriding the auto-generated name. '
                            'Use to run configs that share an auto-name (e.g. M2M vs crM2M) '
                            'concurrently without clobbering each other.')
    args = parser.parse_args()
    
    main(
        seed=args.seed,
        num_robots=args.num_robots,
        T=args.time_horizon,
        max_number_tasks=args.max_tasks,
        task_generation_strategy=args.task_gen_strategy,
        initial_task_assignment_strategy=args.initial_task_assign_strategy,
        improvement_task_assignment_strategy=args.improvement_task_assign_strategy,
        path_planning_strategy=args.path_planning_strategy,
        map_name=args.map,
        time_limit=args.time_limit,
        visualize_output=args.visualize,
        initial_inventory=args.initial_inventory,
        frequency=args.frequency,
        inbound_outbound_ratio=args.inbound_outbound_ratio,
        output_graphs=args.output_graphs,
        num_skus=args.num_skus,
        weight_init_method=args.weight_init_method,
        cost_calculation_method=args.cost_calculation_method,
        removal_operator=args.removal_operator,
        repair_operator=args.repair_operator,
        acceptance_function=args.acceptance_function,
        T_0=args.T_0,
        alpha=args.alpha,
        deadline_generation_method=args.deadline_generation_method,
        deadline_offset=args.deadline_offset,
        output_intermediate_data=args.output_intermediate_data,
        intermediate_data_interval=args.intermediate_data_interval,
        base_cost_weight=args.base_cost_weight,
        deadline_weight=args.deadline_weight,
        sku_distribution_weight=args.sku_distribution_weight,
        agent_unallocated_penalty=args.agent_unallocated_penalty,
        solution_repair_detection_function=args.solution_repair_detection_function,
        solution_repair_function=args.solution_repair_function,
        use_precomputed_schedule=args.use_precomputed_schedule,
        schedule_file=args.schedule_file,
        initial_inventory_file=args.initial_inventory_file,
        run_until_schedule_complete=args.run_until_schedule_complete,
        use_precomputed_queue=args.use_precomputed_queue,
        queue_file=args.queue_file,
        run_until_queue_complete=args.run_until_queue_complete,
        W=args.W,
        B=args.B,
        enable_rearrangement=args.enable_rearrangement,
        crm2m_lambda=args.crm2m_lambda,
        crm2m_detour_cutoff=args.crm2m_detour_cutoff,
        crm2m_return_margin=args.crm2m_return_margin,
        lambda_=args.lambda_,
        reallocation_task_method=args.reallocation_task_method,
        pick_place_time=args.pick_place_time,
        pick_place_duration=args.pick_place_duration,
        buffer_capacity_k=args.buffer_capacity_k,
        buffer_consumption_rate=args.buffer_consumption_rate,
        queue_release_window=args.queue_release_window,
        log_buffer_predictions=args.log_buffer_predictions,
        output_file_override=args.output_file,
    )