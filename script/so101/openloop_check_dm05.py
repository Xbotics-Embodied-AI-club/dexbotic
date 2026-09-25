"""开环自检：拿训练数据的画面与状态问 DM0.5 服务，预测应贴近数据里真实的后续动作。

闭环成功率很低时要先分清：是策略还没学会，还是推理这条链配错了（动作口径、图像槽位、
单位、指令）。后者在开环上就会露馅 —— 喂的是训练时见过的输入，输出却连「原地不动」
这个基线都赢不了。

判据：逐关节平均绝对误差，模型 vs「保持当前状态」基线，模型低于基线的 60% 算链路通。

用法（仓根，仿真侧环境即可，只用 numpy / av / urllib）：
    $SIM_PY script/so101/openloop_check_dm05.py --endpoint http://127.0.0.1:7891/v1/infer \\
        --jsonl $SO101_DATASETS_DIR/so101-dexdata/jsonl/sim_cube40_ep00000.jsonl --out <json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import av
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from layout import DATASETS_DIR

# 同一个请求函数：量的就是评测走的那条链。
from rollout_so101 import request_actions


def frames_at(path: pathlib.Path, wanted: set[int]) -> dict[int, np.ndarray]:
    out = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                out[index] = frame.to_ndarray(format="rgb24")
            if index >= max(wanted):
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--jsonl", required=True, type=pathlib.Path, nargs="+")
    ap.add_argument(
        "--image-root",
        default=DATASETS_DIR,
        type=pathlib.Path,
        help="dexdata 视频 url 的相对基准，与转换时的 --image-root 同值",
    )
    ap.add_argument("--starts", type=int, nargs="+", default=[0, 60, 120, 180, 240])
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()

    rows = []
    for jsonl in args.jsonl:
        recs = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        state = np.asarray([r["state"] for r in recs], dtype=np.float32)
        action = np.asarray([r["action"] for r in recs], dtype=np.float32)
        starts = [s for s in args.starts if s + args.horizon <= len(recs)]
        cams = {}
        for slot, name in (("images_1", "top"), ("images_2", "wrist")):
            wanted = {recs[s][slot]["frame_idx"] for s in starts}
            got = frames_at(args.image_root / recs[0][slot]["url"], wanted)
            cams[name] = {s: got[recs[s][slot]["frame_idx"]] for s in starts}
        for s in starts:
            pred = request_actions(
                args.endpoint, {k: cams[k][s] for k in cams}, state[s], recs[s]["prompt"], 120
            )
            h = min(len(pred), args.horizon)
            truth = action[s : s + h]
            rows.append(
                {
                    "clip": jsonl.stem,
                    "start": s,
                    "model_mae": np.abs(pred[:h] - truth).mean(axis=0).round(2).tolist(),
                    "hold_mae": np.abs(state[s][None] - truth).mean(axis=0).round(2).tolist(),
                    "pred_first": pred[0].round(1).tolist(),
                    "truth_first": truth[0].round(1).tolist(),
                    "state": state[s].round(1).tolist(),
                }
            )
            r = rows[-1]
            print(
                f"{jsonl.stem} t={s:3d}  model {np.mean(r['model_mae']):6.2f}  hold {np.mean(r['hold_mae']):6.2f}"
                f"  | pred0 {r['pred_first']}  truth0 {r['truth_first']}",
                flush=True,
            )
    model = float(np.mean([np.mean(r["model_mae"]) for r in rows]))
    hold = float(np.mean([np.mean(r["hold_mae"]) for r in rows]))
    args.out.write_text(json.dumps({"model_mae": model, "hold_mae": hold, "rows": rows}, indent=2))
    print(
        f"\n[开环] 平均绝对误差 模型 {model:.2f}° vs 保持不动 {hold:.2f}° ⇒ {'链路通' if model < 0.6 * hold else '可疑'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
