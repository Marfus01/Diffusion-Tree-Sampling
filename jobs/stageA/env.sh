# Environment for stage-A jobs on n46 (sourced by the sbatch scripts).
source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate dts
export DTS_ROOT=$HOME/run/dts
export HF_HOME=$DTS_ROOT/scratch/.cache/huggingface
export TORCH_HOME=$DTS_ROOT/scratch/.cache/torch
export DTS_CACHE_DIR=$DTS_ROOT/scratch/.cache
export DTS_DATA_DIR=$DTS_ROOT/data/torchvision
export DTS_BASE_DIR=$DTS_ROOT/experiments
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd $DTS_ROOT/Diffusion-Tree-Sampling
echo "[$(date)] host=$(hostname) commit=$(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
