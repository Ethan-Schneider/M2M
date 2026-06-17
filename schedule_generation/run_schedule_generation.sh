#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

output_file_name="schedule_20"
map_file_name="study_small_restricted"
init_inventory_full_percentage=0.25
total_time=1200
num_skus=30
tasks_per_timestep=2
task_release_frequency=0.5
deadline_generation_method="constant"
deadline_offset=60

json_file="${HOME}/M2M/data/schedules/${output_file_name}.json"
mkdir -p "$(dirname "$json_file")"
if [[ ! -f "$json_file" ]]; then
    echo '[]' > "$json_file"
fi

args_json=$(<"$json_file")

python3 ${parent_path}/schedule_generator.py \
    --output_file_name $output_file_name \
    --map_file_name $map_file_name \
    --init_inventory_full_percentage $init_inventory_full_percentage \
    --total_time $total_time \
    --tasks_per_timestep $tasks_per_timestep \
    --task_release_frequency $task_release_frequency \
    --num_skus $num_skus \
    --deadline-generation-method $deadline_generation_method \
    --deadline-offset $deadline_offset \
    --args_json "$args_json"

