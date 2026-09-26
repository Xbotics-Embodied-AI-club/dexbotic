#!/usr/bin/env bash
# DM0.5 某个存点在三个仿真场景上的闭环成功率：每个场景一个推理服务 + 一个仿真进程，三路并行。
#
# 用法（仓根）：
#     SIM_PY=<仿真环境的 python> bash script/so101/eval_dm05_so101.sh \
#         <checkpoint 目录> <每场景局数> <输出目录> <推理服务的三张卡，逗号分隔>
#
# 环境变量：
#     SIM_PY     仿真侧解释器（必填）。仿真与训练两边的 torch 版本冲突，装不进同一个环境
#     SIM_GPUS   仿真渲染用的三张卡，默认同推理服务
#     REPLAN     一个动作块执行几步就重新请求，默认 25
#     PORT_BASE  三个服务的起始端口，默认 7891
#
# 一个场景一个服务：同一个模型不接并发请求（Flask 开发服务器默认多线程，
# 同卡同模型并发推理的结果没有保证）。
#
# `--action-mode absolute`：DM0.5 按 RELATIVE 训练时，推理服务的输出变换
# （dm05_exp.py 里的 ActionAbsolute）已经把状态加回去、返回绝对角。harness 再加一次
# 会静默翻倍，只表现为很低的成功率。
#
# 判据：每场景 Clopper-Pearson 95% 下界都 > 70%（每场景 50 局时至少 42 局成功）。
set -uo pipefail

CKPT=${1:?用法: eval_dm05_so101.sh <ckpt> <局数> <输出目录> <gpus>}
N=${2:?}
OUT=${3:?}
IFS=, read -r -a GPUS <<< "${4:?第四个参数给推理服务的三张卡，例如 0,1,2}"
# 渲染的卡与推理服务的卡分开给：与别的作业共用的卡可能渲出成片纯黑块，策略看到的是
# 训练里没有的脏图，成功率被压低且不报错。rollout_so101.py 在首帧会拦下这种卡。
IFS=, read -r -a SIM_GPUS <<< "${SIM_GPUS:-$4}"
: "${SIM_PY:?用 SIM_PY 指定仿真环境的 python（见 script/so101/sim_env/pyproject.toml）}"
PORT_BASE=${PORT_BASE:-7891}
SCENES=(cube40 cube20 cylinder40)
LABEL=dm05-$(basename "$CKPT")
cd "$(dirname "$0")/../.."
source script/so101/env.sh
mkdir -p "$OUT"
[ -f "$CKPT/adapter_model.safetensors" ] || { echo "[eval] $CKPT 里没有 LoRA 适配器" >&2; exit 1; }

pids=()
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" -m dexbotic.so101.dm05_exp --task inference \
    --model-config.model-name-or-path "$CKPT" --inference-config.port "$((PORT_BASE + i))" \
    > "$OUT/server_${SCENES[$i]}.log" 2>&1 &
  pids+=($!)
done
trap 'kill "${pids[@]}" 2>/dev/null' EXIT

for i in 0 1 2; do
  for _ in $(seq 180); do
    curl -s -o /dev/null -m 2 "http://127.0.0.1:$((PORT_BASE + i))/" && break
    kill -0 "${pids[$i]}" 2>/dev/null || { echo "[eval] ${SCENES[$i]} 的服务起不来："; tail -20 "$OUT/server_${SCENES[$i]}.log"; exit 1; }
    sleep 10
  done
done
echo "[eval] 三个服务就绪 $(date -Is)"

sims=()
for i in 0 1 2; do
  # 仿真环境里没装 dexbotic；rollout 要用 dexbotic.so101.client，按仓根导入（它只依赖 numpy / av）。
  CUDA_VISIBLE_DEVICES=${SIM_GPUS[$i]} PYTHONPATH="$PWD" "$SIM_PY" -W ignore script/so101/rollout_so101.py \
    --out "$OUT/${SCENES[$i]}" --endpoint "http://127.0.0.1:$((PORT_BASE + i))/v1/infer" \
    --label "$LABEL" --action-mode absolute --episodes "$N" --scenes "${SCENES[$i]}" --replan "${REPLAN:-25}" \
    --wandb-mode offline > "$OUT/rollout_${SCENES[$i]}.log" 2>&1 &
  sims+=($!)
done
rc=0
for p in "${sims[@]}"; do wait "$p" || rc=1; done

"$SIM_PY" - "$OUT" "$LABEL" <<'PY'
import json, pathlib, sys
from scipy.stats import beta
out, label = pathlib.Path(sys.argv[1]), sys.argv[2]

def exact_low(k, n):
    """Clopper-Pearson 95% 双侧区间的下界。"""
    return 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))

rows = {}
for scene in ("cube40", "cube20", "cylinder40"):
    f = out / scene / f"rollout_{label}.json"
    if not f.is_file():
        print(f"  {scene}: 没跑完"); continue
    s = json.loads(f.read_text())["scenes"][scene]
    s["ci95_low"] = exact_low(s["successes"], s["episodes"])
    rows[scene] = s
    print(f"  {scene:10s} {s['successes']:3d}/{s['episodes']:3d} = {s['success_rate']:.3f}  95% 下界 {s['ci95_low']:.3f}")
passed = len(rows) == 3 and all(r["ci95_low"] > 0.70 for r in rows.values())
(out / "summary.json").write_text(json.dumps({"label": label, "scenes": rows, "passed_70pct_lower_bound": passed},
                                             ensure_ascii=False, indent=2))
print(f"[eval] 三项 95% 下界都 > 70%：{'是' if passed else '否'}")
PY
exit $rc
