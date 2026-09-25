#!/usr/bin/env bash
# LoRA 微调 DM0.5（SO101，仿真 + 真机混训）。
#
# 用法（仓根）：CUDA_VISIBLE_DEVICES=0,1 bash script/so101/train_dm05_so101.sh [tyro 覆盖参数...]
#
# 配方按等效 batch 定义，不按卡数定义：`卡数 × 每卡 batch × 累积 = 48` 恒定。
# 累积步数从卡数现算：手工凑这个乘积时改了卡数忘了改累积，等效 batch 会静默变成几倍，
# 训练照跑、loss 也不明显异常，只是已经不是这个配方了。乘不出 48 就报错停下。
# 2 卡 → 累积 3；6 卡 → 累积 1；4 卡时 48/32 不整除，设 PER_DEVICE=6 即可（累积 2）。
#
# 卡号是唯一的声明处，进程数从它数出来 —— 两者分别给就可能不一致。
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?用 CUDA_VISIBLE_DEVICES 指定训练用的卡，例如 0,1}"
export CUDA_VISIBLE_DEVICES
NPROC=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-48}
PER_DEVICE=${PER_DEVICE:-8}
GRAD_ACCUM=$(( EFFECTIVE_BATCH / (NPROC * PER_DEVICE) ))
if [ "$GRAD_ACCUM" -lt 1 ] || [ $(( GRAD_ACCUM * NPROC * PER_DEVICE )) -ne "$EFFECTIVE_BATCH" ]; then
  echo "[配方] $NPROC 卡 × 每卡 $PER_DEVICE × 累积 ? 凑不出等效 batch $EFFECTIVE_BATCH；改 PER_DEVICE。" >&2
  exit 1
fi
echo "[配方] $NPROC 卡（$CUDA_VISIBLE_DEVICES）× 每卡 $PER_DEVICE × 累积 $GRAD_ACCUM = 等效 batch $EFFECTIVE_BATCH"

# 入口与 config/ 都相对仓根定位。
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source script/so101/env.sh
use_staged_datasets

# wandb 默认在线，训练中就能看曲线；离线跑设 WANDB_MODE=offline，完事 `wandb sync`。
export WANDB_MODE=${WANDB_MODE:-online}
export TOKENIZERS_PARALLELISM=false

# 上游 DM0 / DM0.5 的训练入口都走 torchrun（见 docs/DM0.md）。`python -m` 调用，
# 不依赖 PATH 里是哪个环境的 torchrun。
# 入口是 `tyro.cli`，覆盖走短横线旗标（`--trainer-config.x-y`），
# 不是 DW0.5 那边的 `字段路径=值`。两个入口的覆盖语法不同。
# 端口由调用方声明：同一台机器上跑两个 torchrun 时，默认的 29500 第二个会起不来（EADDRINUSE）。
exec "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port "${DM05_MASTER_PORT:-29500}" \
  playground/dm05_so101_xbotics.py \
  --task train \
  --trainer-config.per-device-train-batch-size "$PER_DEVICE" \
  --trainer-config.gradient-accumulation-steps "$GRAD_ACCUM" \
  "$@"
