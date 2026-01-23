#!/bin/bash

# Create directories if they don't exist
mkdir -p data/raw_data
mkdir -p data/buffer_data
mkdir -p data/videos

#Init absolute path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Define arrays of parameters to test
seeds=(500 501 502 503 504 505 506 507 508 509)
num_robots=(40)
time_horizons=(28800)
max_tasks=(120)
frequencies=(0.25)
inbound_outbound_ratio=(0.85)
num_skus=(30)
initial_inventory=(25.0)
weight_init_method="uniform"
task_gen_strategy="informed_uniform"
initial_task_assign_strategy="fast_greedy"
improvement_task_assign_strategy="c_lns"
cost_calculation_method="shortest_path"
path_planning_strategy="pbs"
map="data/maps/study_small_restricted"
removal_operator="shaw"
repair_operator="greedy"
acceptance_function="simulated_annealing"
T_0=1.0
alpha=0.99
deadline_generation_method="normal"
deadline_offset=180
output_intermediate_data=False
intermediate_data_interval=14400
base_cost_weight=1.0
deadline_weight=0.0
sku_distribution_weight=0.0
agent_unallocated_penalty=5
solution_repair_detection_function="Duration"
solution_repair_function="BnB"

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
                            --initial-inventory 25.0 \
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
                            --solution-repair-function "$solution_repair_function"

                        # Optional: Add a small delay between runs
                        sleep 1
                    done
                done
            done
        done
    done
done

echo "All experiments completed!" 