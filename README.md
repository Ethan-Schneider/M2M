# symbotic_tamp

Task allocation and multi-agent path planning simulator for warehouse robotics, developed at Georgia Tech.

---

## Table of Contents

- [Installation](#installation)
  - [Python Requirements](#python-requirements)
  - [C++ Requirements](#c-requirements)
  - [Git Submodules](#git-submodules)
  - [Building the C++ Modules](#building-the-c-modules)
- [Running Experiments](#running-experiments)
  - [run_experiments.sh Parameters](#run_experimentssh-parameters)
- [Paper Experiments](#paper-experiments)

---

## Installation

### Python Requirements

Python 3.11+ is required. Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

Key packages include `numpy`, `scipy`, `pandas`, `matplotlib`, `pybind11`, and `statsmodels`.

### C++ Requirements

The C++ submodules (LNS, C-LNS, PBS, EECBS) require the following system libraries:

```bash
# Ubuntu / Debian
sudo apt-get install cmake build-essential libeigen3-dev libboost-all-dev
```

- **CMake** >= 2.6
- **Eigen3** — linear algebra library used internally by LNS
- **Boost** — `program_options`, `system`, and `filesystem` components
- **pybind11** — installed via pip (included in `requirements.txt`)

### Git Submodules

The four C++ solvers are included as git submodules. After cloning the repo, initialize them with:

```bash
git submodule update --init --recursive
```

This will populate:

| Submodule | Path | Purpose |
|-----------|------|---------|
| LNS | `GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns` | Large Neighborhood Search task allocator |
| C-LNS | `GT_grid_world/src/task_allocation_algorithms/external_algorithms/p_lns` | Conflict-aware LNS task allocator |
| PBS | `GT_grid_world/src/path_finding_algorithms/external_algorithms/PBS` | Priority-Based Search path planner |
| EECBS | `GT_grid_world/src/path_finding_algorithms/external_algorithms/EECBS` | Enhanced ECBS path planner |

> **Note:** The LNS and C-LNS submodules are hosted on `github.gatech.edu` and require Georgia Tech credentials. Make sure you have SSH access configured or use HTTPS with a personal access token.

### Building the C++ Modules

Each submodule must be compiled in-place so that Python can import the resulting `.so` files directly from the submodule directory. Build each one with `cmake` and `make`:

**LNS (task allocation):**
```bash
cd GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns
cmake .
make
```

**C-LNS (conflict-aware task allocation):**
```bash
cd GT_grid_world/src/task_allocation_algorithms/external_algorithms/p_lns
cmake .
make
```

**PBS (path planning):**
```bash
cd GT_grid_world/src/path_finding_algorithms/external_algorithms/PBS
cmake .
make
```

**EECBS (path planning):**
```bash
cd GT_grid_world/src/path_finding_algorithms/external_algorithms/EECBS
cmake .
make
```

After building, each directory should contain a shared library (e.g., `lns.cpython-*.so`, `pbs.cpython-*.so`) that Python imports at runtime.

---

## Running Experiments

Experiments are launched via the shell script in `GT_grid_world/`:

```bash
cd GT_grid_world
bash run_experiments.sh
```

The script creates output directories (`data/raw_data`, `data/buffer_data`, `data/videos`), then sweeps over all combinations of the specified parameter arrays, invoking `GT_grid_world.py` for each combination.

### run_experiments.sh Parameters

**Sweep arrays** — the script loops over every combination of values in these arrays:

| Parameter | Description |
|-----------|-------------|
| `seeds` | Random seeds for reproducibility |
| `num_robots` | Number of warehouse robots |
| `time_horizons` | Simulation duration in seconds (e.g., `28800` = 8 hours) |
| `max_tasks` | Maximum number of tasks buffered per robot |
| `frequencies` | Task arrival rate (tasks per second) |
| `num_skus` | Number of distinct SKU types in the inventory |

**Fixed parameters** — set once and shared across all runs:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `inbound_outbound_ratio` | `1.0` | Ratio of inbound to outbound tasks |
| `initial_inventory` | `60.0` | Starting inventory level per SKU |
| `weight_init_method` | `uniform` | How SKU weights are initialized |
| `task_gen_strategy` | `feedback_control` | Task generation strategy; uses inventory feedback to regulate task rate |
| `initial_task_assign_strategy` | `fast_greedy` | Algorithm used for the initial task assignment |
| `improvement_task_assign_strategy` | `c_lns` | Algorithm used for iterative improvement (C-LNS) |
| `cost_calculation_method` | `shortest_path` | How agent travel costs are estimated |
| `path_planning_strategy` | `pbs` | Multi-agent path planner (Priority-Based Search) |
| `map` | `data/maps/study_small_restricted` | Warehouse map file |
| `removal_operator` | `shaw` | LNS removal heuristic (Shaw removal) |
| `repair_operator` | `greedy` | LNS repair heuristic |
| `acceptance_function` | `simulated_annealing` | Solution acceptance criterion |
| `T_0` | `1.0` | Initial temperature for simulated annealing |
| `alpha` | `0.99` | Cooling rate for simulated annealing |
| `deadline_generation_method` | `normal` | Distribution used to sample task deadlines |
| `deadline_offset` | `180` | Mean deadline offset in seconds |
| `base_cost_weight` | `1.0` | Weight on travel cost in the objective |
| `deadline_weight` | `0.0` | Weight on deadline violations in the objective |
| `sku_distribution_weight` | `0.0` | Weight on SKU distribution balance in the objective |
| `agent_unallocated_penalty` | `5.0` | Penalty applied per unallocated agent |
| `solution_repair_detection_function` | `none` | Method for detecting when a solution needs repair |
| `solution_repair_function` | `none` | Method used to repair a degraded solution |

**Output** — results for each run are written to `data/raw_data/`. Intermediate snapshots can be enabled by setting `output_intermediate_data=True` and `intermediate_data_interval` to the desired timestep cadence.

---

## Paper Experiments

> This section describes the experiments conducted for our paper. Details to be filled in.

<!-- TODO: describe the experimental setup, baselines compared, metrics reported, and how to reproduce the main results from the paper. -->
