#!/usr/bin/env bash
# 微调 DW0.5 世界模型（SO101，仿真 + 真机混训）。
#
# 用法（仓根）：
#     CUDA_VISIBLE_DEVICES=0 bash script/so101/train_dw05_so101.sh compute_norm_stats   先算归一化统计
#     CUDA_VISIBLE_DEVICES=0,1,2 bash script/so101/train_dw05_so101.sh train [字段路径=值 ...]
#
# DeepSpeed ZeRO-1：DW0.5 是 5B 规模，AdamW 优化器状态全量约 72 GB，单卡放不下；
# ZeRO-1 把它切到各卡上。N 卡时每卡约 72/N GB 优化器状态，另加权重 12 GB、梯度 12 GB 与激活。
# 上游的 script/dw/accelerate_zero1_ds.yaml 就是这个配置。
set -euo pipefail

TASK=${1:?用法: train_dw05_so101.sh <compute_norm_stats|train|smoke> [字段路径=值 ...]}
shift
: "${CUDA_VISIBLE_DEVICES:?用 CUDA_VISIBLE_DEVICES 指定训练用的卡，例如 0,1,2}"
export CUDA_VISIBLE_DEVICES
NPROC=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

# accelerate 配置里的 `deepspeed_config_file: script/dw/ds_zero1_config.json` 是仓根相对路径，
# 所以必须在仓根启动。
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source script/so101/env.sh
use_staged_datasets

# deepspeed 在导入时就会找 nvcc。
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"

# 底座目录（vae / text_encoder / tokenizer 都在里面）。
export DIFFSYNTH_MODEL_BASE_PATH="$DW05_BUNDLE"
# 底座 DiT（Wan2.2-TI2V-5B，约 30 GB）上游默认从 ModelScope 取；网络条件不同时
# 两个源的速度能差一个量级，这个开关是上游自带的。
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-huggingface}"
# 训练路径会去找 Action Expert 的初始化权重，而发布包里没有这个文件。置空后走推理分支
# 同一条随机初始化；Action Expert 的真实权重在 model.pt 里，由 resume 覆盖。
export DW05_ACTION_DIT_PRETRAINED_PATH="${DW05_ACTION_DIT_PRETRAINED_PATH-}"
# 多卡起步慢，心跳超时与上游脚本一致。
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

# wandb 默认在线，训练中就能看曲线；离线跑设 WANDB_MODE=offline，完事 `wandb sync`。
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

if [ "$TASK" = "compute_norm_stats" ] || [ "$TASK" = "smoke" ]; then
  # 统计要在一份数据上串行累计，多进程各算一份会得到不完整的统计。
  exec "$PY" -m dexbotic.so101.dw05_exp --task "$TASK" "$@"
fi

# 端口显式给：与 DM0.5 同机并行时两边默认都是 29500，accelerate 撞了会自己挪，
# 只留一行 UserWarning，两次启动挪到的端口还可能不同。
exec "$PY" -m accelerate.commands.launch \
  --config_file script/dw/accelerate_zero1_ds.yaml \
  --num_processes "$NPROC" \
  --main_process_port "${DW05_MASTER_PORT:-29520}" \
  -m dexbotic.so101.dw05_exp \
  --task "$TASK" \
  "$@"
