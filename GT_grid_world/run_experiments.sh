#!/bin/bash

# Create directories if they don't exist
mkdir -p data/raw_data
mkdir -p data/buffer_data
mkdir -p data/videos

#Init absolute path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Define arrays of parameters to test
seeds=(0)
num_robots=(3)
time_horizons=(3602)
max_tasks=(20)
frequencies=(1.0)
inbound_outbound_ratio=(1.0)
num_skus=(5)
weight_init_method="random"
task_gen_strategy="informed_uniform"
task_assign_strategy="randomized_greedy"
cost_calculation_method="manhattan"
path_planning_strategy="ecbs"
map="data/maps/symbotic_small"

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
                        echo "  Task Assign Strategy: $task_assign_strategy"
                        echo "  Path Planning Strategy: $path_planning_strategy"
                        echo "  Map: $map"
                        echo "----------------------------------------"
                        
                        python3 $parent_path/GT_grid_world.py \
                            --seed "$seed" \
                            --num-robots "$robots" \
                            --time-horizon "$T" \
                            --max-tasks "$max_task" \
                            --task-gen-strategy "$task_gen_strategy" \
                            --task-assign-strategy "$task_assign_strategy" \
                            --path-planning-strategy "$path_planning_strategy" \
                            --time-limit 86400 \
                            --initial-inventory 25.0 \
                            --frequency "$frequency" \
                            --inbound-outbound-ratio "$inbound_outbound_ratio" \
                            --num-skus "$num_sku" \
                            --weight-init-method "$weight_init_method" \
                            --map "$map" \
                            --cost-calculation-method "$cost_calculation_method"
                        
                        # Optional: Add a small delay between runs
                        sleep 1
                    done
                done
            done
        done
    done
done

echo "All experiments completed!" 