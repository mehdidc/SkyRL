#!/bin/bash -x
#SBATCH --job-name=skyrl_discover
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --partition=dc-gpu
#SBATCH --account=zam
#SBATCH --time=05:00:00
set -euo pipefail

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "Usage: sbatch run_skyrl_discover.sbatch <ac1|circle_packing|spherical_code> [skyrl overrides...]" >&2
  exit 1
fi
shift

ml Stages/2025 CUDA/12
ml git
export UV_NO_SYNC=1
export UV_LINK_MODE=copy

export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_MODE=${WANDB_MODE:-offline}

export UV_PROJECT_ENVIRONMENT=/p/project1/laionize/cherti1/skyrl/.venv
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook # make sure Ray uses uv

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
if [[ "$MASTER_NODE" == *"jrc"* ]]; then
    MASTER_NODE_I="${MASTER_NODE}i"
else
    MASTER_NODE_I="$MASTER_NODE"
fi
export MASTER_ADDR=$(nslookup $MASTER_NODE_I | grep 'Address' | tail -n1 | awk '{print $2}')
echo "Master node: $MASTER_NODE, master node with i: $MASTER_NODE_I"
MASTER_IP=$(getent hosts $MASTER_NODE_I | awk '{ print $1 }')
echo "Master address: $MASTER_ADDR"
echo "Master IP: $MASTER_IP"
MASTER_PORT=$(shuf -i 10000-60000 -n 1)
echo "Master port: $MASTER_PORT"

export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export MODEL_PATH="${MODEL_PATH:-/p/data1/mmlaion/shared/models/Qwen/$MODEL_NAME}"

bash start_ray_server.sh
export RAY_ADDRESS=$(cat ray_address.txt)
echo "Ray address: $RAY_ADDRESS"
uv run ray status --address "$RAY_ADDRESS"


set -x

NUM_GPUS="${NUM_GPUS:-4}"
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout

case "$TASK" in
  ac1)
    TASK_NAME="ac1"
    DATA_DIR="${DATA_DIR:-data/ac_inequalities_ac1}"
    ENTRYPOINT="examples.train.ac_inequalities.main_ac1"
    ENV_CLASS="ac1"
    PROJECT_NAME="ac1_megatron"
    RUN_NAME_PREFIX="ac1"
    CKPT_PATH="${CKPT_PATH:-ckpts/ac1_megatron_ckpt}"
    EXPORT_PATH="${EXPORT_PATH:-exports/ac1}"
    SAVE_PATH_PREFIX="ac1"
    DATASET_HINT="uv run --isolated -m examples.train.ac_inequalities.prepare_dataset --output_dir $DATA_DIR --train_size 1 --test_size 1 --timeout_s 60 --budget_s 60"
    ;;
  circle_packing)
    NUM_CIRCLES="${NUM_CIRCLES:-26}"
    TASK_NAME="circle_packing_${NUM_CIRCLES}"
    DATA_DIR="${DATA_DIR:-data/circle_packing_${NUM_CIRCLES}}"
    ENTRYPOINT="examples.train.circle_packing.main_circle_packing"
    ENV_CLASS="circle_packing"
    PROJECT_NAME="circle_packing_megatron"
    RUN_NAME_PREFIX="circle_packing_${NUM_CIRCLES}"
    CKPT_PATH="${CKPT_PATH:-ckpts/circle_packing_${NUM_CIRCLES}_megatron_ckpt}"
    EXPORT_PATH="${EXPORT_PATH:-exports/circle_packing_${NUM_CIRCLES}}"
    SAVE_PATH_PREFIX="circle_packing_${NUM_CIRCLES}"
    DATASET_HINT="uv run --isolated -m examples.train.circle_packing.prepare_dataset --output_dir $DATA_DIR --num_circles $NUM_CIRCLES --train_size 1 --test_size 1 --timeout_s 60"
    ;;
  spherical_code)
    NUM_POINTS="${NUM_POINTS:-12}"
    DIM="${DIM:-3}"
    TASK_NAME="spherical_code_${NUM_POINTS}_${DIM}"
    DATA_DIR="${DATA_DIR:-data/spherical_code_${NUM_POINTS}_${DIM}}"
    ENTRYPOINT="examples.train.spherical_code.main_spherical_code"
    ENV_CLASS="spherical_code"
    PROJECT_NAME="spherical_code_megatron"
    RUN_NAME_PREFIX="spherical_code_${NUM_POINTS}_${DIM}"
    CKPT_PATH="${CKPT_PATH:-ckpts/spherical_code_${NUM_POINTS}_${DIM}_megatron_ckpt}"
    EXPORT_PATH="${EXPORT_PATH:-exports/spherical_code_${NUM_POINTS}_${DIM}}"
    SAVE_PATH_PREFIX="spherical_code_${NUM_POINTS}_${DIM}"
    DATASET_HINT="uv run --isolated -m examples.train.spherical_code.prepare_dataset --output_dir $DATA_DIR --num_points $NUM_POINTS --dim $DIM --train_size 1 --test_size 1 --timeout_s 60"
    ;;
  *)
    echo "Unsupported task '$TASK'. Supported tasks: ac1, circle_packing, spherical_code" >&2
    exit 1
    ;;
esac

if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/validation.parquet" ]]; then
  echo "Missing $TASK dataset at $DATA_DIR" >&2
  echo "Create it with:" >&2
  echo "$DATASET_HINT" >&2
  exit 1
fi

mkdir -p "$EXPORT_PATH"
echo "Export path: $EXPORT_PATH"

INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}" # currently only vllm is supported for megatron

MEGATRON_TP="${MEGATRON_TP:-4}"
MEGATRON_PP="${MEGATRON_PP:-4}"
MEGATRON_CP="${MEGATRON_CP:-1}"

# torch profiler config
ENABLE_TORCH_PROFILER="${ENABLE_TORCH_PROFILER:-false}"
RANKS_TO_PROFILE="${RANKS_TO_PROFILE:-[0]}"
SAVE_PATH="${SAVE_PATH:-megatron_prof/${SAVE_PATH_PREFIX}_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}}"

NUM_NODES="${SLURM_NNODES:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_GENERATE_LENGTH="${MAX_GENERATE_LENGTH:-16384}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DISCOVER_STATE_EVOLUTION="${DISCOVER_STATE_EVOLUTION:-false}"

uv run --extra megatron -m "$ENTRYPOINT" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_NODES \
  generator.inference_engine.tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
  trainer.policy.megatron_config.torch_profiler_config.enable=$ENABLE_TORCH_PROFILER \
  trainer.policy.megatron_config.torch_profiler_config.ranks=$RANKS_TO_PROFILE \
  trainer.policy.megatron_config.torch_profiler_config.save_path=$SAVE_PATH \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.remove_microbatch_padding=true \
  trainer.epochs=50 \
  trainer.eval_batch_size=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.discover_state_evolution=$DISCOVER_STATE_EVOLUTION \
  environment.env_class=$ENV_CLASS \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="${RUN_NAME_PREFIX}_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  "$@"
