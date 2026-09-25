#!/usr/bin/env bash
# 把仿真三个场景与真机九个任务转进同一份 dexdata，供 DM0.5 与 DW0.5 混训。
#
# 进同一个目录：`JsonlDataset` 只吃一个 jsonl_dir，而同一台机器人、同样 6 维动作空间、
# 同样两路相机、同样 30 fps，本来就该共享一份归一化统计。不同任务靠每帧的 `prompt` 区分。
#
# 输入（目录名是约定，见 docs/so101.md）：
#   $SO101_DATASETS_DIR/so101-sim-640-v2/{cube40,cube20,cylinder40}
#   $SO101_DATASETS_DIR/so101-real/<任务名>/
# 输出：$SO101_DATASETS_DIR/so101-dexdata
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
D=$SO101_DATASETS_DIR

# 仿真目录名里写死分辨率，让「拿错版本」在 `ls` 那一刻就看得见：
# 分辨率不一致既不报错也不影响收敛，靠跑起来发现不了。
SIM=$D/so101-sim-640-v2
REAL=$D/so101-real
for dir in "$SIM" "$REAL"; do
  [ -d "$dir" ] || { echo "[convert] 数据不在：$dir" >&2; exit 1; }
done

ARGS=()
for scene in cube40 cube20 cylinder40; do
  ARGS+=(--source "$SIM/$scene" --name "sim_$scene")
done
# 目录名就是任务名；README.md 之类的文件不是任务，`*/` 只取目录。
for task in "$REAL"/*/; do
  ARGS+=(--source "${task%/}" --name "real_$(basename "$task")")
done

echo "[convert] 共 $(( ${#ARGS[@]} / 4 )) 个来源 → $D/so101-dexdata"
exec "$PY" "$HERE/lerobot_v3_to_dexdata.py" --out "$D/so101-dexdata" --image-root "$D" "${ARGS[@]}"
