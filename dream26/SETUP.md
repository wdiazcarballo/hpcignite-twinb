# DREAM'26 Experiments Setup

## ✅ Fixed Issues

### From Failed Experiment Analysis

The initial experiment (`exp1-fail1/`) revealed several issues that have been fixed:

1. **❌ Virtual environment path error**
   - **Problem:** Script tried to use `~/venv` which doesn't exist
   - **Fix:** Updated to use shared environment: `/project/lt200291-ignite/Project_chomwong/.venv/mesa_env`

2. **❌ Incorrect CPU allocation**
   - **Problem:** 8 CPUs per task (16 total for 2 tasks)
   - **Fix:** Changed to 4 CPUs per GPU (8 total) as requested

3. **❌ Complex directory structure**
   - **Problem:** `experiment1_results/run_3328140/` creates deep paths
   - **Fix:** Simplified to `exp1_3328140/` and `exp2_3328140/` directly in dream26 folder

4. **❌ Windows line endings**
   - **Problem:** Scripts had CRLF endings causing bash errors
   - **Fix:** Converted all scripts to UNIX (LF) format

5. **❌ Scattered output locations**
   - **Problem:** Logs in one place, results in another
   - **Fix:** All outputs now in dream26 folder: `exp1_*.out`, `exp1_*.err`, `exp1_*/`

## 📋 Prerequisites

### Required: Shared Python Environment

**✅ Already set up by pwongta!**

The shared environment is located at:
```
/project/lt200291-ignite/Project_chomwong/.venv/
```

Activate with:
```bash
source /project/lt200291-ignite/Project_chomwong/.venv/bin/activate
```

Verification:
```bash
ls -ld /project/lt200291-ignite/Project_chomwong/.venv/bin
# Should show: drwxrwsr-x+ 2 pwongta lt200291 4096 Oct 23 16:52 ...
```

## 🚀 Running Experiments

### Quick Start

```bash
cd dream26

# Experiment 1: Platform benchmarking (~30 min)
sbatch experiment1_platform_capability.slurm

# Experiment 2: Baseline simulation (~2 hours)
sbatch experiment2_baseline_simulation.slurm

# Monitor
squeue -u $USER
```

### Check Results (Simple Paths!)

```bash
# All outputs are in dream26 folder - no more nested subdirectories!

# Check logs
tail -f dream26/exp1_3328140.out
tail -f dream26/exp2_3328140.err

# List results
ls -la dream26/exp1_3328140/
ls -la dream26/exp2_3328140/
```

### Analyze Results

```bash
cd dream26

# Option 1: Automatic (finds most recent)
./run_analysis.sh

# Option 2: Specify job ID
./run_analysis.sh 3328140

# Option 3: Direct
python analyze_baseline.py exp2_3328140
```

## 📁 New Directory Structure

```
dream26/
├── experiment1_platform_capability.slurm
├── experiment2_baseline_simulation.slurm
├── analyze_baseline.py
├── run_analysis.sh
│
├── exp1_3328140.out          # Experiment 1 stdout
├── exp1_3328140.err          # Experiment 1 stderr
├── exp1_3328140/             # Experiment 1 results
│   ├── gpu_compute_results.json
│   ├── memory_bandwidth_results.json
│   └── ...
│
├── exp2_3328150.out          # Experiment 2 stdout
├── exp2_3328150.err          # Experiment 2 stderr
└── exp2_3328150/             # Experiment 2 results
    ├── EXPERIMENT_SUMMARY.md
    ├── twinb_baseline_profile.nsys-rep
    └── ...
```

**Benefits:**
- ✅ No more `logs/` and `experiment1_results/` subdirectories
- ✅ All files in one place (dream26 folder)
- ✅ Shorter shell prompts
- ✅ Easier to navigate
- ✅ Easy to clean up: `rm -rf dream26/exp*`

## 🔧 Resource Allocation

Both experiments now use:
- **GPUs:** 2× NVIDIA A100 (40GB each)
- **CPUs:** 4 CPUs per GPU = 8 total
- **Memory:** 64GB RAM
- **Runtime:** 30 min (exp1), 2 hours (exp2)

## 📝 Notes

- All scripts use **UNIX (LF) line endings**
- Shared Python environment from `/project/lt200291-ignite/Project_chomwong/.venv/mesa_env`
- Scripts will fail early if environment not found
- No more deep directory nesting - everything flat in dream26/

## ❓ Troubleshooting

**Environment not found:**
```
ERROR: Shared virtual environment not found at /project/lt200291-ignite/Project_chomwong/.venv/mesa_env
```
→ Ask pwongta to run the setup commands above

**Permission denied:**
```
chmod: changing permissions of '/project/.../': Operation not permitted
```
→ pwongta needs to set group permissions (step 4 above)

**Short paths confirmed:**
```bash
# Check your prompt length
cd dream26/exp1_3328140
pwd  # Should be reasonable length
```
