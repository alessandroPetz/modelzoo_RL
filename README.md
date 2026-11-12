# HPC Energy Management with Reinforcement Learning

**A PPO-based energy dispatch controller for HPC clusters, integrated into the Seanergys Model Zoo.**

## Overview

This project investigates how reinforcement learning can be used to manage the energy sources supplying a high-performance computing (HPC) cluster. HPC workloads generate time-varying power demand, while renewable generation, electricity prices, and grid carbon intensity can also vary over time.

The goal is to learn an **energy dispatch policy** that balances three objectives:

- **Operating cost:** reduce the cost of supplying energy to the cluster.
- **Carbon emissions:** reduce the emissions associated with the selected energy sources.
- **Grid power constraint:** avoid drawing more than a specified power limit from the electrical grid.

The controller does **not** schedule jobs, change CPU/GPU frequencies, or directly reduce the cluster's computational power demand. Instead, it decides **how to supply that demand** using the available energy resources.

## Energy system

The simulated infrastructure includes:

| Component | Role |
| --- | --- |
| HPC cluster | Time-varying electrical load, derived from HPC job power traces. |
| Solar and wind | Variable renewable generation, used first when available. |
| Battery energy storage | Stores surplus renewable energy and can supply power later. |
| Fuel-powered generator | Supplies additional power at an operating cost and with associated emissions. |
| Electrical grid | Supplies the remaining demand, subject to a maximum grid power threshold. |

At each simulation step, renewable energy is allocated to the HPC load first. Surplus renewable energy automatically charges the battery, subject to its physical limits. The RL agent then decides how much of the remaining deficit to cover with the battery and generator. Any remaining load is assigned to the grid up to its configured power limit; excess demand is recorded as a **constraint violation / unmet energy**.

### What the agent learns

The main challenge is **when to use energy now versus when to preserve it for later**. For example, discharging the battery may reduce current electricity purchases but leave less stored energy for a later demand peak. Using the generator may help respect the grid limit but introduce fuel costs and emissions.

A successful policy should learn to make these trade-offs over an episode rather than independently optimizing each time step. Whether the learned policy actually outperforms rule-based alternatives must be established experimentally.

## Reinforcement learning formulation

The environment is implemented as `HPCBatteryEnv` using **Gymnasium**; the controller uses **Proximal Policy Optimization (PPO)** from **Stable-Baselines3**.

### Observations

The agent receives a 14-dimensional observation vector containing:

- Current HPC power demand and the deficit after renewable generation.
- Battery state of charge (SOC) and remaining episode time.
- Current electricity price signals and grid carbon intensity.
- Time of day (sine/cosine representation).
- Current renewable power generation.
- Renewable energy and grid carbon-intensity values for 1-hour and 6-hour look-ahead windows.

The implementation normalizes observations for PPO. The look-ahead signals are currently derived from future values in the simulation data; they therefore represent **perfect-foresight information**, not independently trained forecasts.

### Actions

The agent chooses **two discrete control values** every simulation step:

1. **Battery discharge fraction:** 0% to 100% of the demand remaining after renewables.
2. **Generator contribution fraction:** 0% to 100% of the demand remaining after battery discharge.

Each fraction has 11 levels (0%, 10%, ..., 100%), giving **121 possible action combinations**. Actual energy delivery is constrained by the battery and generator models.

Battery charging from renewable surplus is automatic; it is **not** an RL action.

### Reward

The step reward penalizes energy cost, carbon emissions, and violations of the grid power limit:

```text
reward = - 1.0 * step_cost_eur
         - 0.001 * step_co2_g
         - 50.0 * violation_kwh
```

PPO aims to maximize cumulative reward over the episode. Reward values will generally be negative because supplying the workload incurs costs and emissions; **less negative reward is better**, provided the same workload and scenario are used for comparison.

## Data and simulation

The agent uses the **PM100 HPC job power dataset**, accessed through Seanergys' `PM100Dataset` and `SeanergysDataloader` components. Job-level power measurements are processed into a cluster-level load trace and split into simulation episodes.

Additional time-dependent inputs are produced using the project's models and data files for renewable generation, electricity prices, and grid carbon intensity.

**Default configuration:**

| Parameter | Default |
| --- | --- |
| PM100 time window | 2020-05-25 to 2020-06-25 (UTC) |
| Episode duration | 24 hours |
| Simulation step | 20 seconds |
| Maximum grid power | 400 kW |
| Battery capacity | 3,200 kWh |
| Maximum wind generation | 400 kW |
| Maximum solar generation | 450 kW |
| Generator maximum power | 500 kW |
| RL algorithm | PPO with an MLP policy |
| Policy hidden layers | 256, 256, 128 |

The battery model includes physical charging/discharging limits and degradation-related costs. The generator model accounts for operating constraints, fuel costs, and emissions.

## Seanergys Model Zoo integration

`RLAgentSim3HardConstraint` extends the common `SeanergysModel` interface and provides:

- `get_train_config()` and `get_test_config()` to build model/dataloader configurations.
- `train()` to train the PPO policy.
- `predict()` to run the policy on one or more episodes.
- `evaluate()` to summarize reward, cost, emissions, battery/generator use, SOC, and constraint violations.
- `save()` and `load()` for model persistence.

The implementation uses the existing Seanergys dataset and dataloader interfaces without requiring changes to the framework for the standard (non-dummy) data flow.

### Main files

```text
test_rl.py
seanergys_modelzoo/
├── datasets/pm100.py
├── dataloader/seanergys_dataloader.py
└── models/tasks/power_consumption_manage/RL/
    ├── rl_agent_sim3_hard_constraint.py
    ├── utils/
    │   ├── battery_model.py
    │   ├── generator_model.py
    │   ├── renewable_real.py
    │   ├── price_model.py
    │   └── co2.py
    ├── csvs/
    └── saved_models/
```

## Getting started

Run commands from the repository root in a Python environment with the required dependencies installed: **PyTorch, Gymnasium, Stable-Baselines3, NumPy, pandas, PyArrow, Pydantic, requests**, and the supporting Seanergys dependencies. The included `environment.yml` provides an environment starting point but may require additional packages.

### 1. Provide PM100

The default local dataset path is:

```text
seanergys_modelzoo/models/tasks/power_consumption_manage/RL/csvs/job_table.parquet
```

If the file is absent, the agent can use the PM100 Parquet hosted on Zenodo. A local copy is recommended to avoid repeated remote reads:

```bash
mkdir -p seanergys_modelzoo/models/tasks/power_consumption_manage/RL/csvs
wget -O seanergys_modelzoo/models/tasks/power_consumption_manage/RL/csvs/job_table.parquet \
  https://zenodo.org/records/10127767/files/job_table.parquet
```

### 2. Run a short training test

The entry-point script `test_rl.py` performs training, model saving, and evaluation. For the repository version in this project, set `USE_DUMMY = False` in that script to use the normal PM100 flow:

```bash
python test_rl.py
```

For a shorter test with explicit settings, the RL module also exposes a command-line interface:

```bash
python -m seanergys_modelzoo.models.tasks.power_consumption_manage.RL.rl_agent_sim3_hard_constraint \
  --mode train \
  --window-start 2020-05-25T00:00:00Z \
  --window-end 2020-05-26T00:00:00Z \
  --episode-hours 24 \
  --max-episodes 1 \
  --total-timesteps 5000 \
  --n-steps 512 \
  --batch-size 128 \
  --save-dir saved_models/rl_agent_sim3_hard_constraint
```

The defaults include an episode-based stopping callback (`max_episodes=2`). Consequently, `total_timesteps` is an upper training target, **not necessarily the number of steps actually completed**.

### 3. Inspect the results

Evaluation returns metrics such as:

- Total energy cost and CO2 emissions.
- Battery and generator energy supplied.
- Energy associated with grid-limit violations and renewable curtailment.
- Final battery SOC and capacity ratio.
- Cumulative episode reward.

Saved model artifacts can include `ppo_powercap.zip`, `vecnormalize.pkl`, `model_state.json`, and model metadata. When episode-artifact logging is enabled, an additional results CSV is written in the configured output directory.

## Current status and limitations

The dataset loading, training, saving, and evaluation workflow has been exercised successfully on a **single 24-hour PM100 episode**. This establishes a working integration test, **not** evidence that PPO already outperforms a conventional energy-management strategy.

Important limitations of the current implementation:

- A robust assessment requires comparisons against fixed-rule/greedy baselines and evaluation on **held-out time windows**. The example test currently evaluates the same time window used for training.
- Battery SOC is reset to 50% at the beginning of each episode; energy is not carried across daily episode boundaries.
- Forecast features use future simulation values rather than realistic forecast errors.
- The grid threshold limits recorded grid supply; excess demand is penalized but is **not** handled by job rescheduling or an explicit workload-shedding mechanism.
- The current PM100 dummy-filter implementation may require a compatibility fix. The standard (`is_dummy=False`) path avoids that issue without modifying the Seanergys framework.

The next research step is to evaluate whether a trained controller can reduce operating cost and emissions **while maintaining comparable or lower grid-constraint violations** than baseline dispatch strategies.

## Dataset reference

**PM100 — HPC job power consumption dataset (Marconi100, CINECA).**

- Data: https://zenodo.org/records/10127767
- Related paper: https://doi.org/10.1145/3624062.3624263
