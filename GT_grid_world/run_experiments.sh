#!/bin/bash

#Init absolute path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
repo_root="$parent_path/.."

# Create directories if they don't exist
mkdir -p "$repo_root/data/raw_data"
mkdir -p "$repo_root/data/buffer_data"
mkdir -p "$repo_root/data/videos"

# Define arrays of parameters to test
seeds=(1)
num_robots=(40)
time_horizons=(3000)
max_tasks=(120)
frequencies=(0.25)
inbound_outbound_ratio=(1.0)
num_skus=(30)
initial_inventory=(30.0)
weight_init_method="uniform"
task_gen_strategy="feedback_control"
initial_task_assign_strategy="fast_greedy"
improvement_task_assign_strategy="M2M"
cost_calculation_method="shortest_path"
path_planning_strategy="pbs"
map="$repo_root/data/maps/study_small_restricted"
removal_operator="shaw"
repair_operator="greedy"
acceptance_function="simulated_annealing"
T_0=1.0
alpha=0.99
deadline_generation_method="normal"
deadline_offset=180
output_intermediate_data=False
intermediate_data_interval=4000
base_cost_weight=1.0
deadline_weight=0.0
sku_distribution_weight=0.0
agent_unallocated_penalty=5.0
solution_repair_detection_function="none"
solution_repair_function="none"
W=600
B=30
lambda_=1.0
reallocation_task_method="insertion"

# Precomputed schedule/inventory (set use_precomputed_schedule=true to enable)
use_precomputed_schedule=true
schedule_file="$repo_root/data/schedules/schedule_1_hour.txt"
initial_inventory_file="$repo_root/data/initial_inventories/schedule_1_hour_init_inventory.txt"
run_until_schedule_complete=false

precomputed_args=()
if [ "$use_precomputed_schedule" = true ]; then
    precomputed_args+=(--use-precomputed-schedule)
    precomputed_args+=(--schedule-file "$schedule_file")
    precomputed_args+=(--initial-inventory-file "$initial_inventory_file")
    if [ "$run_until_schedule_complete" = true ]; then
        precomputed_args+=(--run-until-schedule-complete)
    fi
fi

# Loop through all combinations
for seed in "${seeds[@]}"; do
    for robots in "${num_robots[@]}"; do
        for T in "${time_horizons[@]}"; do
            for max_task in "${max_tasks[@]}"; do
                for frequency in "${frequencies[@]}"; do
                    for num_sku in "${num_skus[@]}"; do
                        echo "Running experiment with:"
                        echo "  Seed: $seed"
                        echo "  Robots: $robots"
                        echo "  Time Horizon: $T"
                        echo "  Max Tasks: $max_task"
                        echo "  Frequency: $frequency"
                        echo "  Inbound Outbound Ratio: $inbound_outbound_ratio"
                        echo "  Number of SKUs: $num_sku"
                        echo "  Weight Init Method: $weight_init_method"
                        echo "  Task Gen Strategy: $task_gen_strategy"
                        echo "  Initial Task Assign Strategy: $initial_task_assign_strategy"
                        echo "  Improvement Task Assign Strategy: $improvement_task_assign_strategy"
                        echo "  Path Planning Strategy: $path_planning_strategy"
                        echo "  Map: $map"
                        echo "  Removal Operator: $removal_operator"
                        echo "  Repair Operator: $repair_operator"
                        echo "  Acceptance Function: $acceptance_function"
                        echo "  T_0: $T_0"
                        echo "  Alpha: $alpha"
                        echo "  Base Cost Weight: $base_cost_weight"
                        echo "  Deadline Weight: $deadline_weight"
                        echo "  Sku Distribution Weight: $sku_distribution_weight"
                        echo "  Agent Unallocated Penalty: $agent_unallocated_penalty"
                        echo "  Solution Repair Detection Function: $solution_repair_detection_function"
                        echo "  Solution Repair Function: $solution_repair_function"
                        echo "  Use Precomputed Schedule: $use_precomputed_schedule"
                        echo "  Lookahead Window (W): $W"
                        echo "  Lookahead Beginning (B): $B"
                        echo "  Detour Penalty (lambda_): $lambda_"
                        echo "  Reallocation Task Method: $reallocation_task_method"
                        if [ "$use_precomputed_schedule" = true ]; then
                            echo "  Schedule File: $schedule_file"
                            echo "  Initial Inventory File: $initial_inventory_file"
                            echo "  Run Until Schedule Complete: $run_until_schedule_complete"
                        fi
                        echo "----------------------------------------"
                        
                        python3 $parent_path/GT_grid_world.py \
                            --seed "$seed" \
                            --num-robots "$robots" \
                            --time-horizon "$T" \
                            --max-tasks "$max_task" \
                            --task-gen-strategy "$task_gen_strategy" \
                            --initial-task-assign-strategy "$initial_task_assign_strategy" \
                            --improvement-task-assign-strategy "$improvement_task_assign_strategy" \
                            --path-planning-strategy "$path_planning_strategy" \
                            --time-limit 86400 \
                            --initial-inventory "$initial_inventory" \
                            --frequency "$frequency" \
                            --inbound-outbound-ratio "$inbound_outbound_ratio" \
                            --num-skus "$num_sku" \
                            --weight-init-method "$weight_init_method" \
                            --map "$map" \
                            --cost-calculation-method "$cost_calculation_method" \
                            --removal-operator "$removal_operator" \
                            --repair-operator "$repair_operator" \
                            --acceptance-function "$acceptance_function" \
                            --T-0 "$T_0" \
                            --alpha "$alpha" \
                            --deadline-generation-method "$deadline_generation_method" \
                            --deadline-offset "$deadline_offset" \
                            $( [ "$output_intermediate_data" = "True" ] && echo --output-intermediate-data ) \
                            --intermediate-data-interval "$intermediate_data_interval" \
                            --base-cost-weight "$base_cost_weight" \
                            --deadline-weight "$deadline_weight" \
                            --sku-distribution-weight "$sku_distribution_weight" \
                            --agent-unallocated-penalty "$agent_unallocated_penalty" \
                            --solution-repair-detection-function "$solution_repair_detection_function" \
                            --solution-repair-function "$solution_repair_function" \
                            --W "$W" \
                            --B "$B" \
                            --lambda_ "$lambda_" \
                            --reallocation-task-method "$reallocation_task_method" \
                            "${precomputed_args[@]}"

                        # Optional: Add a small delay between runs
                        sleep 1
                    done
                done
            done
        done
    done
done

echo "All experiments completed!" 