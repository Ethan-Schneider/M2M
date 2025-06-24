#!/bin/bash

# Create directories if they don't exist
mkdir -p data/raw_data
mkdir -p data/buffer_data
mkdir -p data/videos

#Init absolute path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Define arrays of parameters to test
seeds=(0)
num_robots=(40)
time_horizons=(1)
max_tasks=(40)
frequencies=(0.0125)
inbound_outbound_ratio=(1.0)
num_skus=(50)
initial_inventory=(25.0)
weight_init_method="uniform"
task_gen_strategy="informed_uniform"
initial_task_assign_strategy="fast_greedy"
improvement_task_assign_strategy="none"
cost_calculation_method="shortest_path"
path_planning_strategy="ecbs"
map="data/maps/symbotic_small"
removal_operator="worst"
repair_operator="greedy"

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
                            --repair-operator "$repair_operator"
                        
                        # Optional: Add a small delay between runs
                        sleep 1
                    done
                done
            done
        done
    done
done

echo "All experiments completed!" 