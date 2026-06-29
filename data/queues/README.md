# SKU weight configuration files

These JSON files configure **custom SKU tasking weights** for queue generation. They are used when `weight_mode="custom"` in `queue_generation/run_queue_generation.sh`.

During queue generation, each new task samples a SKU using a weighted draw. The weight for SKU `s` at task index `t` comes from the profile defined here. Higher weight means that SKU is more likely to appear in inbound and outbound tasks (subject to inventory feasibility checks).

SKUs not listed in the file default to weight `1.0`. Each SKU may appear in **at most one** entry.

---

## File structure

The file is a **JSON array**. Each element describes one or more SKUs that share the same weight function:

| Field | Type | Description |
|-------|------|-------------|
| `sku_ids` | list of int | 0-indexed SKU ids that use this weight profile |
| `weight_func` | string | `"constant"` or `"sinusoid"` (also accepts `"sinusoidal"`) |
| `weight_args` | list of numbers | Parameters for the chosen function (see below) |

---

## Weight functions

### `constant`

Fixed weight for every task index. Use for always-hot or always-cold SKUs.

**`weight_args`:** `[value]`

| Argument | Meaning |
|----------|---------|
| `value` | Tasking weight applied at every task index (must be > 0) |

**Example:** `[2.0]` — this SKU is sampled twice as often as a SKU with weight `1.0`.

---

### `sinusoid`

Weight oscillates between a minimum and maximum over a fixed number of queue indices. The period does **not** stretch or shrink when you change `number_of_tasks`; generating fewer tasks just uses a shorter segment of the same wave.

**Formula (task index `t`):**

```
weight(t) = min_weight + (max_weight - min_weight) * (sin(2π * t / period_tasks + phase) + 1) / 2
```

**`weight_args`:** `[min_weight, max_weight, period_tasks, phase]`

| Argument | Meaning |
|----------|---------|
| `min_weight` | Weight at the trough of the sine wave (coldest) |
| `max_weight` | Weight at the peak of the sine wave (hottest) |
| `period_tasks` | Queue indices per **full** cycle (peak → trough → peak) |
| `phase` | Radians offset. Use `π/2` (~1.5708) to start at max; add `π` (~3.1416) for the opposite phase |

**Phase tip:** Two groups with the same `min_weight`, `max_weight`, and `period_tasks` but phases differing by `π` (e.g. `π/2` vs `3π/2`) oscillate in opposition—when one group is hot, the other is cold.

---

## Example 1: Alternating sinusoid groups

File: [`sku_weights_alternating_sinusoid_30min.json`](sku_weights_alternating_sinusoid_30min.json)

```json
[
  {
    "sku_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "weight_func": "sinusoid",
    "weight_args": [0.25, 1.75, 2000, 1.5707963267948966]
  },
  {
    "sku_ids": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "weight_func": "sinusoid",
    "weight_args": [0.25, 1.75, 2000, 4.71238898038469]
  }
]
```

**What this does:**

- SKUs **0–14** and **15–29** each form half the catalog.
- Both groups swing between weight **0.25** (cold) and **1.75** (hot).
- One full hot→cold→hot cycle every **2000** task indices, regardless of total queue length.
- Phase `π/2` vs `3π/2` puts the two halves **out of phase**: when 0–14 is hot, 15–29 is cold, and vice versa every **1000** tasks (half period).

Point `run_queue_generation.sh` at this file:

```bash
sku_weights_json_file="${repo_root}/data/queues/sku_weights_alternating_sinusoid_30min.json"
weight_mode="custom"
```

---

## Example 2: Mixed hot, oscillating, and cold SKUs

File: [`sku_weights_example.json`](sku_weights_example.json)

```json
[
  {
    "sku_ids": [0, 1, 2, 3, 4],
    "weight_func": "constant",
    "weight_args": [2.0]
  },
  {
    "sku_ids": [5, 6, 7, 8, 9],
    "weight_func": "sinusoid",
    "weight_args": [0.2, 2.0, 1000, 0.0]
  },
  {
    "sku_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "weight_func": "constant",
    "weight_args": [0.2]
  }
]
```

**What this does:**

| SKUs | Function | Effect |
|------|----------|--------|
| 0–4 | `constant` @ 2.0 | Always hot — steady high demand |
| 5–9 | `sinusoid` 0.2→2.0, period 1000, phase 0 | Demand waves between cold and hot every 1000 tasks; starts at mid-weight (phase 0) |
| 10–29 | `constant` @ 0.2 | Always cold — rarely sampled |

This pattern is useful for testing a small hot set, a time-varying middle group, and a large cold tail.

---

## Related options

| Setting | Location | Notes |
|---------|----------|-------|
| `weight_mode="uniform"` | `run_queue_generation.sh` | Ignores this JSON; all SKUs weight 1.0 |
| `weight_mode="custom"` | `run_queue_generation.sh` | Loads a file like the examples above |
| `inventory_layout_mode="adversarial"` | `run_queue_generation.sh` | Uses mean SKU weights to place high-weight SKUs toward the **back** of the warehouse |
| `save_plots=false` | `run_queue_generation.sh` | Skips per-SKU weight PNG output in this directory |

---

## Output files (same directory)

Queue generation also writes:

- `{output_file_name}.txt` — task queue (`sku_id`, `task_type`, `deadline`)

### Queue deadlines

Set `deadline_rate` in `run_queue_generation.sh` (passed as `--deadline-rate`).
Each simulated second receives `deadline_rate / 60` tasks with the same
deadline timestep (1-based seconds, displayed as `mm:ss`):

| `deadline_rate` | First tasks' deadlines |
|-----------------|------------------------|
| 60 | `00:01`, `00:02`, …, `01:00` (one task per second) |
| 120 | `00:01`, `00:01`, `00:02`, `00:02`, …, `01:00` (two per second) |

Legacy two-column queue files remain supported; the simulator generates
deadlines at release time when the third column is absent.
- `{output_file_name}_sku_{id}_weight.png` — optional weight plots when `save_plots=true` and `weight_mode=custom`

Initial inventory is written to `data/initial_inventories/`, not here.
