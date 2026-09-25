"""把 DW0.5 的动作专家包成 HTTP 策略服务，契约与 DM0.5 的推理服务一致。

`rollout_so101.py` 要成功率，就得有服务出动作。DM0.5 上游自带 Flask 服务
（`--task inference`，`POST /v1/infer`），DW0.5 的 `inference()` 只渲一段 mp4 存盘。
但 DW0.5 是 world-action 模型，动作专家同时在训（loss 里有 `loss_action`），
`infer_action` 是现成的，包一层就能让同一套 harness 量两个模型的成功率。

## 返回绝对关节角

delta → 绝对角的换算在服务内部做，harness 用 `--action-mode absolute`：夹爪那一维
不做 delta（`ROBOTWIN_NON_DELTA_DIMS`），只有 policy 这一侧知道是哪几维。

## 与训练口径一致

状态排布、不做 delta 的维、本体状态不归一化、prompt 不套模板，都经
`dw05_sim_check.patch_policy_for_so101` 换成 SO101 的训练口径，与离线自检同一份。
三路视图照训练时的 `["images_1", "images_2", "images_2"]`（top + wrist + wrist）拼；
顺序不同不报错，只是画面布局与训练时不同。

用法（仓根，训练侧环境）：
    CUDA_VISIBLE_DEVICES=0 $PY script/so101/dw05_policy_server.py \\
        --checkpoint <weights/step_xxxxxx.pt> --port 7892
"""

from __future__ import annotations

import argparse
import base64
import io
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from playground.dw05_so101_exp import DW05_NORM_STATS
from script.so101.dw05_sim_check import patch_policy_for_so101
from script.so101.layout import DW05_BUNDLE


def decode_image(payload: str) -> np.ndarray:
    """base64 图片 → (H, W, 3) uint8。"""
    from PIL import Image

    raw = base64.b64decode(payload)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def build_app(policy, action_horizon: int):
    """建 Flask 应用。

    Args:
        policy: 已加载并打过 SO101 补丁的 `DW05RobotWinPolicy`。
        action_horizon: 一次出多少步动作；执行几步由 harness 的 `--replan` 决定。

    Returns:
        Flask 应用，暴露 `POST /v1/infer` 与 `GET /healthz`。
    """
    import torch
    from flask import Flask, jsonify, request

    from dexbotic.policy.dw05_policy import ROBOTWIN_NON_DELTA_DIMS

    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "action_horizon": action_horizon, "returns": "absolute_qpos"})

    @app.post("/v1/infer")
    def infer():
        body = request.get_json(force=True)
        observation = body.get("observation")
        if not isinstance(observation, dict):
            return jsonify({"error": "observation 必须是 JSON 对象"}), 400
        images_raw = observation.get("images")
        if not isinstance(images_raw, dict):
            return jsonify({"error": "observation.images 必须是 JSON 对象"}), 400

        # harness 按 1 基槽位发：1=top、2=wrist。三路视图由 wrist 复用一次凑齐。
        slots = {int(key): decode_image(value) for key, value in images_raw.items()}
        missing = [index for index in (1, 2) if index not in slots]
        if missing:
            return jsonify({"error": f"缺图像槽位 {missing}，收到 {sorted(slots)}"}), 400
        views = [slots[1], slots[2], slots[2]]

        state = np.asarray(observation.get("state"), dtype=np.float32).reshape(-1)
        if state.shape[0] != policy.raw_state_dim:
            return jsonify(
                {"error": f"state 维度要 {policy.raw_state_dim}，收到 {state.shape[0]}"}
            ), 400

        image_tensor = policy._image_tensor_from_arrays(views)
        proprio = policy.normalize_state(state)
        with torch.no_grad():
            pred = policy.model.infer_action(
                prompt=policy.format_prompt(observation.get("prompt", "")),
                input_image=image_tensor,
                action_horizon=action_horizon,
                proprio=proprio,
                num_inference_steps=int(policy.config.num_inference_steps),
                sigma_shift=policy.config.sigma_shift,
                seed=policy.config.seed,
                rand_device=policy.config.rand_device,
                tiled=bool(policy.config.tiled),
            )
        chunk = policy.denormalize_action(pred["action"])[0]

        # delta → 绝对角。夹爪那几维本来就是绝对量，不加基准 ——
        # 这一步的口径只有 policy 类知道，所以放在这一侧做，不交给 harness。
        absolute = []
        for step in range(chunk.shape[0]):
            action = chunk[step] + state
            for dim in ROBOTWIN_NON_DELTA_DIMS:
                if dim < action.shape[0]:
                    action[dim] = chunk[step, dim]
            absolute.append(action.astype(np.float32).tolist())
        return jsonify({"actions": absolute, "metadata": {"returns": "absolute_qpos"}})

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle", default=DW05_BUNDLE, help="发行包目录，vae / text_encoder / tokenizer 都在里面"
    )
    ap.add_argument("--checkpoint", required=True, help="SO101 上微调出的检查点")
    ap.add_argument("--norm-stats", default=DW05_NORM_STATS)
    ap.add_argument("--port", type=int, default=7892)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--action-horizon", type=int, default=32)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    dw05_policy = patch_policy_for_so101()
    print(f"[dw05] 底座 {args.bundle}  检查点 {args.checkpoint}  设备 {args.device}")
    policy = dw05_policy.DW05RobotWinPolicy(
        dw05_policy.DW05RobotWinPolicyConfig(
            checkpoint_path=args.checkpoint,
            norm_stats_path=args.norm_stats,
            model_base_path=args.bundle,
            device=args.device,
            mixed_precision="bf16",
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            raw_state_dim=6,
            raw_action_dim=6,
        )
    )
    app = build_app(policy, args.action_horizon)
    print(f"[dw05] 服务就绪：POST http://0.0.0.0:{args.port}/v1/infer  返回绝对关节角")
    print(
        f"[dw05] harness 那边要用 --action-mode absolute --endpoint http://127.0.0.1:{args.port}/v1/infer"
    )
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
