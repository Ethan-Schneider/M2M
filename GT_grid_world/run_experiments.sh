#!/bin/bash

# Create directories if they don't exist
mkdir -p data/raw_data
mkdir -p data/buffer_data
mkdir -p data/videos

#Init absolute path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Define arrays of parameters to test
seeds=(0)
num_robots=(15)
time_horizons=(10)
max_tasks=(25)
frequencies=(1.0)
inbound_outbound_ratio=(1.0)
task_gen_strategy="informed_uniform"
task_assign_strategy="lns"
path_planning_strategy="ecbs"
# Loop through all combinations
for seed in "${seeds[@]}"; do
    for robots in "${num_robots[@]}"; do
        for T in "${time_horizons[@]}"; do
            for max_task in "${max_tasks[@]}"; do
                for frequency in "${frequencies[@]}"; do
                    echo "Running experiment with:"
                    echo "  Seed: $seed"
                    echo "  Robots: $robots"
                    echo "  Time Horizon: $T"
                    echo "  Max Tasks: $max_task"
                    echo "  Frequency: $frequency"
                    echo "  Inbound Outbound Ratio: $inbound_outbound_ratio"
                    echo "  Task Gen Strategy: $task_gen_strategy"
                    echo "  Task Assign Strategy: $task_assign_strategy"
                    echo "  Path Planning Strategy: $path_planning_strategy"
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
                        --inbound-outbound-ratio "$inbound_outbound_ratio"
                    
                    # Optional: Add a small delay between runs
                    sleep 1
                done
            done
        done
    done
done

echo "All experiments completed!" 