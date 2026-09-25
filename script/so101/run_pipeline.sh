#!/usr/bin/env bash
# SO101 一条龙：数据验收 → 转 dexdata → 对齐校验 → 文本嵌入 → 剔机器人专属权重
# →（可选）搬内存 → DW0.5 归一化统计 → DM0.5 与 DW0.5 并行开训。
#
# 用法（仓根）：
#     DM_GPUS=0,1 DW_GPUS=2,3,4 LEROBOT_PY=<装了 lerobot 的 python> bash script/so101/run_pipeline.sh
#
# 环境变量：
#     DM_GPUS      DM0.5 训练用的卡（必填）。等效 batch 48 凑不齐时再给 PER_DEVICE，
#                  规则见 train_dm05_so101.sh
#     DW_GPUS      DW0.5 训练用的卡（必填）
#     PREP_GPU     文本嵌入与归一化统计用的卡，默认 DW_GPUS 的第一张
#     LEROBOT_PY   跑对齐校验的解释器（必填）：真值取自 lerobot 自己的数据集 API
#     STAGE_SHM=1  开训前把数据与 DW0.5 权重搬进内存
#     PY 与目录约定见 layout.py / env.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
source script/so101/env.sh
: "${DM_GPUS:?用 DM_GPUS 指定 DM0.5 训练用的卡}"
: "${DW_GPUS:?用 DW_GPUS 指定 DW0.5 训练用的卡}"
: "${LEROBOT_PY:?用 LEROBOT_PY 指定一个装了 lerobot 的 python，对齐校验要用}"
PREP_GPU=${PREP_GPU:-${DW_GPUS%%,*}}
D=$SO101_DATASETS_DIR
LOG=$SO101_RUNS_ROOT/logs
DW05_INIT=$SO101_WEIGHTS_DIR/DW05-Robotwin-noactionhead/model.pt
mkdir -p "$LOG"

echo "[pipeline] ① 数据验收：任务数 / 集数 / 帧数 / 全量视频分辨率"
# 期望值对应 HF so101-sim-pickplace-v2@92ca801c 的三个场景与
# ModelScope so101-pick-place-tasks@f64493c4 的九个任务。
"$PY" - "$D" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
want = {"so101-sim-640-v2": (3, 1498, 495544), "so101-real": (9, 2200, 784963)}
for name, expected in want.items():
    infos = [json.loads(p.read_text()) for p in sorted((d / name).glob("*/meta/info.json"))]
    got = (len(infos), sum(i["total_episodes"] for i in infos), sum(i["total_frames"] for i in infos))
    print(f"  {name}: {got}  期望 {expected}")
    if got != expected:
        sys.exit(f"[pipeline] {name} 对不上")
PY
# 分辨率不一致既不报错也不影响收敛，只能在这里逐个查。
bad=$(find "$D/so101-sim-640-v2" "$D/so101-real" -name '*.mp4' -print0 |
  xargs -0 -n1 ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 |
  grep -vc '^640,480$' || true)
[ "$bad" = 0 ] || { echo "[pipeline] 有 $bad 个视频不是 640×480" >&2; exit 1; }

echo "[pipeline] ② 转 dexdata"
bash script/so101/convert_all.sh > "$LOG/convert.log" 2>&1

echo "[pipeline] ③ 画面与动作对齐（仿真、真机各抽一份）"
for pair in "so101-sim-640-v2/cube20 sim_cube20" \
            "so101-real/pick_up_a_can_and_place_in_the_bin real_pick_up_a_can_and_place_in_the_bin"; do
  read -r src name <<< "$pair"
  "$LEROBOT_PY" script/so101/verify_dexdata_alignment.py \
    --dexdata "$D/so101-dexdata" --lerobot "$D/$src" --repo-id "local/$name" --name "$name" | tail -3
done

echo "[pipeline] ④ DW0.5 文本嵌入"
CUDA_VISIBLE_DEVICES=$PREP_GPU "$PY" script/so101/precompute_text_embeds.py \
  --jsonl-dir "$D/so101-dexdata/jsonl" --out "$D/so101-dexdata/text_embeddings" \
  --bundle "$DW05_BUNDLE" 2>&1 | tail -3

echo "[pipeline] ⑤ DW0.5 起点：剔掉机器人专属权重"
if [ -f "$DW05_INIT" ]; then
  echo "  已有：$DW05_INIT"
else
  "$PY" script/so101/strip_robot_specific_weights.py --in "$DW05_BUNDLE/model.pt" --out "$DW05_INIT"
fi

if [ "${STAGE_SHM:-0}" = 1 ]; then
  echo "[pipeline] ⑥ 数据与 DW0.5 权重搬内存"
  bash script/so101/stage_to_shm.sh | tail -5
  # DW0.5 按 mmap 缺页读权重，在网络盘上是随机小读，比顺序拷贝慢一个量级，
  # 所以先顺序拷进 tmpfs 再装。只拷训练读的：发行包里的 model.pt 不读，起点是剔过的那份。
  W=$SO101_SHM_ROOT/weights
  mkdir -p "$W/DW05-Robotwin/Wan-AI/Wan2.2-TI2V-5B" "$W/DW05-Robotwin-noactionhead"
  (cd "$DW05_BUNDLE" &&
    cp -r --preserve=timestamps config.json norm_stats.json text_encoder vae tokenizer "$W/DW05-Robotwin/" &&
    cp --preserve=timestamps Wan-AI/Wan2.2-TI2V-5B/*.json Wan-AI/Wan2.2-TI2V-5B/*.safetensors \
      "$W/DW05-Robotwin/Wan-AI/Wan2.2-TI2V-5B/")
  cp --preserve=timestamps "$(dirname "$DW05_INIT")"/* "$W/DW05-Robotwin-noactionhead/"
  export DW05_BUNDLE=$W/DW05-Robotwin
  DW05_INIT=$W/DW05-Robotwin-noactionhead/model.pt
fi

echo "[pipeline] ⑦ DW0.5 归一化统计"
CUDA_VISIBLE_DEVICES=$PREP_GPU bash script/so101/train_dw05_so101.sh compute_norm_stats \
  > "$LOG/dw05-norm-stats.log" 2>&1
[ -f "$SO101_RUNS_ROOT/dw05_so101/norm_stats/norm_stats.json" ] ||
  { echo "[pipeline] DW0.5 norm_stats 没产出，见 $LOG/dw05-norm-stats.log" >&2; exit 1; }

echo "[pipeline] ⑧ 并行开训"
# setsid 让训练脱离当前会话：终端或 ssh 断开时不会被 SIGHUP 带走。
CUDA_VISIBLE_DEVICES=$DW_GPUS setsid bash script/so101/train_dw05_so101.sh train \
  trainer_config.resume="$DW05_INIT" \
  > "$LOG/dw05.log" 2>&1 < /dev/null &
# wandb 默认写进 cwd（仓根）下的 wandb/，指到产物目录里。
mkdir -p "$SO101_RUNS_ROOT/dm05_so101"
CUDA_VISIBLE_DEVICES=$DM_GPUS WANDB_DIR="$SO101_RUNS_ROOT/dm05_so101" setsid bash script/so101/train_dm05_so101.sh \
  > "$LOG/dm05.log" 2>&1 < /dev/null &
echo "[pipeline] 已起：日志 $LOG/dw05.log  $LOG/dm05.log"
