#!/bin/bash
#SBATCH -J A-M3-sd
#SBATCH -p gpu_a800
#SBATCH --gpus=1
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
# A-R009: SD-v1.5 + ImageReward on the 20 GeneVal prompts of prompts/geneval_sub20.json.
# Array task i runs the prompt indices in word i of TASKS (e.g. TASKS="0,1,2 3,4"), through
#   main: Best-of-N, FK (diff potential, lambda=1), DTS
#   star: DTS* (--maximize)
#   max:  FK-max (max potential, lambda=10, best final particle)
# then the offline metrics of eval/eval_sd.py. Submit with --array=0..(#words - 1) and --time.
set -euo pipefail
export DTS_CODE=$HOME/run/dts/wt-m3
source $DTS_CODE/jobs/stageA/env.sh

STAGE=${STAGE:?}
read -ra WORDS <<< "${TASKS:?}"
IDS=${WORDS[$SLURM_ARRAY_TASK_ID]//,/ }
VARIANTS=${VARIANTS:-main star max}
NFE=${NFE:-1000 3000 10000}
EVAL=${EVAL:-1}
PROMPTS=prompts/geneval_sub20.json

# Per-job TensorRT timing cache (the library default is one shared file in the node's /tmp),
# seeded from the cache of an earlier job so that engines build in seconds instead of minutes.
export TMPDIR=/tmp/$USER-dts-$SLURM_JOB_ID
TRT_LOCAL=$TMPDIR/torch_tensorrt_engine_cache/timing_cache.bin
TRT_SHARED=$DTS_ROOT/scratch/trt/timing_cache.bin
mkdir -p $(dirname $TRT_LOCAL) $(dirname $TRT_SHARED)
[ -f $TRT_SHARED ] && cp $TRT_SHARED $TRT_LOCAL
trap 'rm -rf $TMPDIR' EXIT

COMMON="--prompts_file $PROMPTS --prompt_ids $IDS --sd_model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --guiding_reward_fns ImageReward --compute_budget $NFE --population_size 16 --seed 0 --no_progress_bar \
  --base_dir $DTS_BASE_DIR ${SD_EXTRA:-}"
TAG=${STAGE}_t${SLURM_ARRAY_TASK_ID}
echo "[$(date)] stage=$STAGE prompts=[$IDS] variants=[$VARIANTS] nfe=[$NFE]"

for V in $VARIANTS; do
  echo "[$(date)] variant $V"
  case $V in
    main) python -m scripts.run_sd $COMMON --do_bon --do_fkd --do_mcts_rollout --exp_name m3_${TAG}_main ;;
    star) python -m scripts.run_sd $COMMON --do_mcts_rollout --maximize --exp_name m3_${TAG}_star ;;
    max)  python -m scripts.run_sd $COMMON --do_fkd --potential_type max --lambda 10 --exp_name m3_${TAG}_max ;;
  esac
done

if [ ! -f $TRT_SHARED ] && [ -f $TRT_LOCAL ]; then
  cp $TRT_LOCAL $TRT_SHARED.$SLURM_JOB_ID && mv -n $TRT_SHARED.$SLURM_JOB_ID $TRT_SHARED
  rm -f $TRT_SHARED.$SLURM_JOB_ID
fi

if [ "$EVAL" = 1 ]; then
  for V in $VARIANTS; do
    RUN=$DTS_BASE_DIR/sd/geneval_sub20_m3_${TAG}_$V
    echo "[$(date)] offline metrics $RUN"
    python -m eval.eval_sd --image_dir $RUN/images/ImageReward --prompts_file $PROMPTS \
      --out $RUN/metrics/offline_eval.csv
  done
fi
echo "[$(date)] done"
