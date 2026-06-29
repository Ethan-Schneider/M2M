#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
repo_root="$( cd "${parent_path}/.." ; pwd -P )"

output_file_name="1000_25_percent_inventory_high_frequency_adversarial"
map_file_name="study_small_restricted"
init_inventory_full_percentage=0.25
number_of_tasks=1000
num_skus=30

# Initial inventory layout: "uniform" (random) or "adversarial" (high-weight SKUs in back)
inventory_layout_mode="uniform"

# Save per-SKU weight plots to data/queues/ (true or false)
save_plots=false

# SKU tasking weights: "uniform" (equal) or "custom" (per-SKU functions from JSON)
# Custom sinusoid weight_args: [min_weight, max_weight, period_tasks, phase]
#   period_tasks is queue indices per full cycle (independent of number_of_tasks)
weight_mode="custom"
sku_weights_json_file="${repo_root}/data/queues/sku_weights_alternating_sinusoid_30min.json"

if [ "$weight_mode" = "custom" ]; then
    if [[ ! -f "$sku_weights_json_file" ]]; then
        echo "SKU weights file not found: $sku_weights_json_file" >&2
        exit 1
    fi
    sku_weights_json=$(<"$sku_weights_json_file")
else
    sku_weights_json="[]"
fi

if [ "$save_plots" = true ]; then
    save_plots_arg="--save-plots"
else
    save_plots_arg="--no-save-plots"
fi

python3 ${parent_path}/queue_generator.py \
    --output_file_name $output_file_name \
    --map_file_name $map_file_name \
    --init_inventory_full_percentage $init_inventory_full_percentage \
    --number_of_tasks $number_of_tasks \
    --num_skus $num_skus \
    --weight_mode $weight_mode \
    --inventory_layout_mode $inventory_layout_mode \
    $save_plots_arg \
    --sku_weights_json "$sku_weights_json"
