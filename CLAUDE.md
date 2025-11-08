# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **digital twin building simulation** that couples **Mesa agent-based modeling** (ABM) with **EnergyPlus** building energy simulations. The system models occupant behavior and thermal comfort in a multi-zone building, using distributed computing with PyTorch DDP for scalability.

## Architecture

### Core Components

1. **main.py**: Orchestrates the entire simulation
   - Initializes distributed training (PyTorch DDP with NCCL/Gloo backend)
   - Manages rank-based execution (rank 0 runs EnergyPlus, all ranks run agent models)
   - Coordinates communication between Mesa agents and EnergyPlus via callbacks
   - Handles setpoint aggregation across distributed processes using `torch.distributed.all_gather`

2. **model.py**: Defines `BuildingModel` (Mesa Model)
   - Manages agent creation from JSON configuration
   - Interfaces with EnergyPlus API (pyenergyplus) when `ep_control=True`
   - Maintains zone temperature state and setpoint handles
   - Collects simulation data via Mesa `DataCollector`

3. **agent.py**: Defines agent types
   - `BaseAgent`: Parent class with thermal comfort logic
   - Agent types: `StudentAgent`, `StaffAgent`, `CleanerAgent`, `WardenAgent`, `VisitorAgent`, `PolicyAgent`
   - Each agent has `preferred_temp`, `comfort_tolerance`, and tracks `using_ac` status
   - Agents store temperature data as PyTorch tensors on the configured device (CPU/GPU)

4. **utils.py**: Helper functions for sampling agent attributes from distributions (uniform, normal)

5. **config.yaml**: Simulation configuration
   - Defines simulation steps (typically 288 for daily timesteps)
   - Lists all building zones (65+ zones in current config)
   - References agent configuration file

6. **agents.json**: Agent population configuration
   - Specifies count, gender ratio, age range for each agent type
   - Defines preferred temperature and comfort tolerance distributions

### Data Flow

```
EnergyPlus (rank 0) → callback → read zone temps → update BuildingModel.last_zone_temps
                                                   ↓
                                          step_agents() → update agent states
                                                   ↓
                                          compute_setpoint_requests() → aggregate by zone
                                                   ↓
                                          torch.distributed.all_gather → merge across ranks
                                                   ↓
                                          apply_setpoints_to_ep() → control HVAC
```

### Distributed Computing Strategy

- **Single-process mode**: Runs on CPU with Gloo backend
- **Multi-process mode**: Uses `torchrun` with NCCL (GPU) or Gloo (CPU) backend
- **Rank 0**: Runs both EnergyPlus (in separate thread) and agent model
- **Other ranks**: Run agent models only, participate in setpoint aggregation
- Setpoints are aggregated using `torch.min()` across all ranks (most conservative cooling)

## Running the Simulation

### HPC Cluster (SLURM)

```bash
# Submit job to SLURM scheduler
sbatch job.slurm

# Monitor job
squeue -u $USER
tail -f logs/boonchu_full_<jobid>.out
```

**Key SLURM parameters** (from `job.slurm`):
- Uses 2 GPUs (`--gres=gpu:2`)
- 2 tasks per node (`--ntasks=2`)
- Runs with `torchrun --standalone --nproc_per_node=2`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Single-process run (for testing)
python main.py

# Multi-process run (2 GPUs/CPUs)
torchrun --standalone --nproc_per_node=2 main.py
```

### Environment Setup

Required environment variables (set in `main.py` or externally):
```bash
ENERGYPLUS_EXE=/path/to/energyplus/energyplus
NCCL_DEBUG=INFO                    # For debugging distributed training
OMP_NUM_THREADS=1                  # Prevent CPU oversubscription
TORCH_NCCL_TIMEOUT=1200           # Extend timeout for long simulations
```

## Key Configuration Files

### config.yaml
- `mesa.steps`: Number of simulation timesteps (288 = 5-minute intervals for 1 day)
- `mesa.zones`: List of zone names (must match IDF file zone names)
- Zone name normalization: spaces → underscores, lowercase for matching

### agents.json
- `agent_types.<type>.distribution.count`: Number of agents to create
- `agent_types.<type>.attributes.preferred_temp`: Temperature preference distribution (°C)
- `agent_types.<type>.attributes.comfort_tolerance`: Acceptable temperature deviation (°C)

## EnergyPlus Integration

### IDF File Setup
- Main IDF file: `EnergyPlus_BP_Boonchoo/output/expanded.idf`
- Weather file: `EnergyPlus_BP_Boonchoo/output/in.epw`
- Zone names in IDF must match (case-insensitive, space/underscore normalized) zones in `config.yaml`

### EnergyPlus API Handles
- **Variable handle**: `get_variable_handle("Zone Air Temperature", zone_name)`
- **Actuator handle**: `get_actuator_handle("Zone Temperature Control", "SecondarySchool ClgSetp", zone_name)`
- Handles are initialized in first timestep callback (`setup_handles_first_timestep`)

### Callback Mechanism
- EnergyPlus calls `ep_agent_callback()` after each HVAC timestep
- Callback reads zone temperatures, updates agent states, computes setpoints, applies to EnergyPlus
- Runs in separate thread (`ep_thread`) on rank 0

## Output Files

### Simulation Results
- `mesa_agent_results.csv`: Agent-level data (per timestep, per agent)
  - Columns: day, hour, step, agent_id, agent_type, room, current_temp, comfort_level, using_ac, preferred_temp
- `mesa_out_result/mesa_zone_results.csv`: Zone-level aggregated data (from Mesa DataCollector)
- `outEnergyPlusBoonchoo/`: EnergyPlus output directory (eplusout.csv, eplusout.sql, etc.)

### Analysis Notebooks
- Stored in `analyze/` subdirectories (e.g., `analyze/boonchu-full/analyze.ipynb`)
- Used for post-processing simulation results

## Device Management

The code supports both CPU and GPU execution:
- Device is automatically selected based on availability: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Agent temperature tensors are stored on the configured device
- For multi-GPU: uses `LOCAL_RANK` environment variable to assign GPUs to processes
- Backend selection: NCCL for GPUs, Gloo for CPU

## Important Implementation Details

### Zone Name Matching
The code performs case-insensitive, space-normalized matching between:
1. `config.yaml` zone names (e.g., `Zone_Classroom_7302`)
2. IDF file zone names (may have different capitalization/spacing)

Normalization function in `model.py:66-67`:
```python
def normalize(name):
    return name.lower().strip().replace(" ", "_")
```

### Setpoint Aggregation
When multiple agents in the same zone request different setpoints:
- Agents with `using_ac=True` contribute their `preferred_temp`
- Zone setpoint = `min(all_agent_requests)` (most aggressive cooling)
- Distributed: `torch.min(torch.stack(gathered), dim=0)` across all ranks

### Commented Code
The codebase contains substantial commented-out code (lines 180-339 in main.py, similar patterns elsewhere). This appears to be previous implementation versions. Consider these for historical reference when debugging.

## Dependencies

Core dependencies (from `requirements.txt`):
- `mesa>=2.1`: Agent-based modeling framework
- `numpy>=1.23`, `pandas>=2.0`: Data processing
- `PyYAML>=6.0`: Configuration parsing

Additional requirements (not in requirements.txt, installed separately):
- `torch`: PyTorch for distributed computing and tensor operations
- `pyenergyplus`: EnergyPlus Python API
- `eppy`: IDF file parsing

## Troubleshooting

### Common Issues

1. **Zone handle errors**: Verify zone names match between `config.yaml` and IDF file
2. **NCCL timeout**: Increase `TORCH_NCCL_TIMEOUT` if simulation steps take >20 minutes
3. **Memory issues**: Reduce agent count in `agents.json` or simulation steps in `config.yaml`
4. **EnergyPlus crash**: Check `outEnergyPlusBoonchoo/eplusout.err` for EnergyPlus-specific errors

### Debugging Distributed Training

Enable verbose logging:
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

Check rank-specific output:
```bash
# In SLURM output files
grep "\[Rank 0\]" logs/boonchu_full_*.out
```
