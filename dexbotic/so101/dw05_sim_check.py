"""在训练集外的仿真轨迹上验 DW0.5：照真动作推演的未来要比两个基线更像真值。

轨迹来自 `rollout_so101.py` 跑 DM0.5 时另存的那批（top/wrist 两路 mp4 + 逐步状态 npz），
训练集里没有它们。每条轨迹从第 0 帧起连推 `--rollouts` 轮（每轮 32 步动作出 9 帧），
与真值逐帧比，三个参照：

    真动作      该轨迹真实的下一帧状态序列（训练时动作也是这么从相邻状态导出的）
    倒放动作      同一段沿时间轴翻转 —— 值域不变，只变时序
    复制起始帧    条件帧原样重复 —— 什么都不预测的分数

三组都从第 1 帧起计分：第 0 帧是条件帧，「复制起始帧」在这一帧与真值逐像素相同、
PSNR 顶到上限（约 80 dB），算进平均会把这个基线抬高约 2 dB。

**判据**：全部轨迹上真动作的平均 PSNR 同时高于「复制起始帧」与「倒放动作」，
且逐条轨迹上真动作胜过倒放动作的占多数。只高过复制起始帧不够 —— 那可能只是学会了
「画面会动」而没按动作动。

## SO101 的排布怎么进上游 policy

`DW05RobotWinPolicy` 把 RobotWin 的状态排布与「不做 delta 的维」写成模块常量。
这里在构造前把它们换成训练时的 SO101 值（与 `playground/dw05_so101_exp.py` 同一份），
上游代码不改。换错不报错，只会把关节喂进错的槽位 —— 所以直接从训练配置 `dw05_exp` 导入，不抄一份。

用法：
    CUDA_VISIBLE_DEVICES=0 python -m dexbotic.so101.dw05_sim_check --checkpoint <weights/step_xxxxxx.pt> \\
        --rollout-dir <rollout_so101.py 的 --out> --out <目录>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

from dexbotic.so101.client import read_frames
from dexbotic.so101.dw05_exp import (
    DW05_NORM_STATS,
    SO101_NON_DELTA_MASK,
    SO101_STATE_ARRANGEMENT,
)
from dexbotic.so101.layout import DW05_BUNDLE

ACTION_HORIZON = 32
FPS_STRIDE = 4
NUM_VIDEO_FRAMES = 9
SEED = 1234
#: 训练时的三路视图：top + wrist + wrist，见 dw05_so101_exp.py 的 images_keys。
VIEWS = ("top", "wrist", "wrist")


def patch_policy_for_so101():
    """把上游 policy 的 RobotWin 常量换成 SO101 的训练口径。

    Returns:
        打过补丁的 `dexbotic.policy.dw05_policy` 模块。
    """
    from dexbotic.policy import dw05_policy

    # 训练时 70% 的样本 prompt 就是原样的任务指令；不套 RobotWin 的模板。
    os.environ["DEPLOY_USE_DEFAULT_PROMPT"] = "0"

    dw05_policy.ROBOTWIN_STATE_ARRANGEMENT = list(SO101_STATE_ARRANGEMENT)
    # NON_DELTA_MASK 给的是排布后的槽位；policy 在原始维上判，换回原始维号。
    dw05_policy.ROBOTWIN_NON_DELTA_DIMS = [
        SO101_STATE_ARRANGEMENT[slot] for slot in SO101_NON_DELTA_MASK
    ]
    dw05_policy.ROBOTWIN_VALID_ARRANGED_DIMS = [
        i for i, src in enumerate(SO101_STATE_ARRANGEMENT) if src >= 0
    ]

    # 本体状态（proprio）在训练时**没有归一化**：`ActionNormMultiDataset` 只归一化 norm_stats
    # 里有的键，而 compute_norm_stats 只产了 `action`；proprio 由 `AddProprioTrajectory`
    # 直接从排布后的原始状态（度 + 终止位）切窗口。上游 policy 却按 RobotWin 的习惯把 state
    # 做分位数归一化 —— 照搬就是喂给模型一个训练时从没见过的量纲，不报错。
    # ⇒ 推理这一侧也不归一化：排布 → 补终止位 0 → 取模型宽度，与训练逐步一致。
    import torch

    def normalize_state_like_training(self, raw_state):
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        with_term = np.concatenate(
            [dw05_policy._arrange_raw_state(raw_state), np.zeros(1, dtype=np.float32)]
        )
        value = dw05_policy._select_model_dims(with_term, self.proprio_dim)
        return (
            torch.from_numpy(value)
            .unsqueeze(0)
            .to(device=self.model.device, dtype=self.model.torch_dtype)
        )

    dw05_policy.DW05RobotWinPolicy.normalize_state = normalize_state_like_training
    # policy 构造时硬要求 stats 里有 `state`；它在上面已被绕开、不会被读，给一个占位让构造通过。
    original_load = dw05_policy._load_norm_stats

    def load_with_state_placeholder(path):
        stats = original_load(path)
        stats.setdefault("state", stats["action"])
        return stats

    dw05_policy._load_norm_stats = load_with_state_placeholder
    return dw05_policy


def as_rgb(frame) -> np.ndarray:
    return (
        frame if isinstance(frame, np.ndarray) else np.asarray(frame.convert("RGB"), dtype=np.uint8)
    )


def score(prediction, truth) -> dict[str, float]:
    """逐帧 PSNR / SSIM，指标实现用上游自己的那份。"""
    import torch

    from dexbotic.exp.utils import video_psnr, video_ssim

    n = min(len(prediction), len(truth))
    pred, gt = (
        torch.from_numpy(np.stack([as_rgb(f) for f in seq[:n]]).astype(np.float32) / 255.0).permute(
            3, 0, 1, 2
        )
        for seq in (prediction, truth)
    )
    return {
        "psnr": float(video_psnr(pred, gt, data_range=1.0)),
        "ssim": float(video_ssim(pred, gt, data_range=1.0)),
        "frames_compared": n,
    }


def strip(frames, columns: int = 9) -> np.ndarray:
    picks = np.linspace(0, len(frames) - 1, min(columns, len(frames))).astype(int)
    return np.concatenate([as_rgb(frames[i]) for i in picks], axis=1)


def labelled_grid(rows):
    """几条 strip 上下摆好并写上标签 —— 没标签分不出哪行是真值。"""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(np.concatenate([row for _, row in rows], axis=0))
    draw = ImageDraw.Draw(canvas)
    offset = 0
    for label, row in rows:
        draw.rectangle((4, offset + 4, 12 + 8 * len(label), offset + 22), fill=(0, 0, 0))
        draw.text((8, offset + 8), label, fill=(255, 255, 0))
        offset += row.shape[0]
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm-stats", default=DW05_NORM_STATS)
    ap.add_argument("--bundle", default=DW05_BUNDLE)
    ap.add_argument("--rollout-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--per-scene", type=int, default=4, help="每个场景验几条轨迹")
    ap.add_argument("--rollouts", type=int, default=3, help="每条连推几轮（每轮 32 步）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dw05_policy = patch_policy_for_so101()

    needed = args.rollouts * ACTION_HORIZON
    picked: list[pathlib.Path] = []
    for scene in ("cube40", "cube20", "cylinder40"):
        clips = [
            p
            for p in sorted((args.rollout_dir / "videos").glob(f"*_{scene}_ep*.npz"))
            if len(np.load(p)["state"]) > needed
        ]
        picked += clips[: args.per_scene]
    if not picked:
        raise SystemExit(f"{args.rollout_dir}/videos 下没有长于 {needed} 步的轨迹")

    policy = dw05_policy.DW05RobotWinPolicy(
        dw05_policy.DW05RobotWinPolicyConfig(
            checkpoint_path=args.checkpoint,
            norm_stats_path=args.norm_stats,
            model_base_path=args.bundle,
            device=args.device,
            mixed_precision="bf16",
            action_horizon=ACTION_HORIZON,
            num_inference_steps=10,
            num_video_frames=NUM_VIDEO_FRAMES,
            seed=SEED,
            raw_state_dim=6,
            raw_action_dim=6,
        )
    )
    size_hw = policy.config.image_size_hw
    layout = policy.config.image_layout
    print(
        f"[dw05] 归一化 {policy.normalization_mode} · 动作条件 {policy.action_condition_mode} · 布局 {layout}"
    )

    rows, grids = [], []
    for npz in picked:
        stem = npz.stem
        data = np.load(npz)
        states, prompt = data["state"], str(data["prompt"])
        cams = {
            "top": read_frames(npz.with_suffix(".mp4")),
            "wrist": read_frames(npz.with_name(stem + "_wrist.mp4")),
        }

        truth_idx = [0] + [
            r * ACTION_HORIZON + j * FPS_STRIDE
            for r in range(args.rollouts)
            for j in range(1, NUM_VIDEO_FRAMES)
        ]
        truth = [
            dw05_policy.compose_robotwin_image(
                [cams[v][i] for v in VIEWS], layout=layout, image_size_hw=size_hw
            )
            for i in truth_idx
        ]
        init = policy._image_tensor_from_arrays([cams[v][0] for v in VIEWS])
        # 整局都传进去，只推前 `--rollouts` 轮：上游按序列长度定「局尾」终止位，
        # 只传前 96 步会把第 81 步起标成局尾，而训练时这些步离局尾还远。
        # 状态也逐步传：每轮的动作增量与本体状态都以该轮起点为基准，与训练时一致。
        action_real = states[1:].copy()
        action_reversed = np.concatenate([action_real[:needed][::-1], action_real[needed:]])
        preds = {}
        for name, action in (
            ("real_action", action_real),
            ("reversed_action", action_reversed),
        ):
            preds[name] = policy.rollout_video_with_actions(
                prompt=prompt,
                init_image_tensor=init,
                action_abs=action,
                state_abs=states[: len(action)],
                max_rollouts=args.rollouts,
                fps_stride=FPS_STRIDE,
            )
        row = {
            "clip": stem,
            "prompt": prompt,
            "real_action": score(preds["real_action"][1:], truth[1:]),
            "reversed_action": score(preds["reversed_action"][1:], truth[1:]),
            "copy_first_frame": score([truth[0]] * (len(truth) - 1), truth[1:]),
        }
        rows.append(row)
        print(
            f"[{stem}] PSNR 真动作 {row['real_action']['psnr']:.2f} · 倒放动作 {row['reversed_action']['psnr']:.2f}"
            f" · 复制起始帧 {row['copy_first_frame']['psnr']:.2f}",
            flush=True,
        )
        labels = (
            "simulator ground truth",
            "DW0.5 imagined, DM0.5's actions",
            "DW0.5 imagined, time-reversed actions",
        )
        seqs = (truth, preds["real_action"], preds["reversed_action"])
        labelled_grid([(lab, strip(seq)) for lab, seq in zip(labels, seqs, strict=True)]).save(
            args.out / f"dw05_sim_{stem}.png"
        )
        # 同一内容的视频：三行上下对齐逐帧播放。每行左上角写明是什么，免得被当成别的片子。
        import imageio.v3 as iio

        n = min(len(q) for q in seqs)
        video = [
            np.asarray(
                labelled_grid(
                    [
                        (f"{lab}  [{stem}] frame {t + 1}/{n}", as_rgb(seq[t]))
                        for lab, seq in zip(labels, seqs, strict=True)
                    ]
                )
            )
            for t in range(n)
        ]
        iio.imwrite(str(args.out / f"dw05_sim_{stem}.mp4"), np.stack(video), fps=4, codec="libx264")
        grids.append(stem)

    mean = {
        k: float(np.mean([r[k]["psnr"] for r in rows]))
        for k in ("real_action", "reversed_action", "copy_first_frame")
    }
    wins = int(sum(r["real_action"]["psnr"] > r["reversed_action"]["psnr"] for r in rows))
    passed = (
        mean["real_action"] > mean["copy_first_frame"]
        and mean["real_action"] > mean["reversed_action"]
        and wins * 2 > len(rows)
    )
    report = {
        "checkpoint": args.checkpoint,
        "clips": len(rows),
        "mean_psnr": mean,
        "real_beats_reversed": f"{wins}/{len(rows)}",
        "passed": passed,
        "rows": rows,
    }
    (args.out / "dw05_sim_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n[结果] 平均 PSNR：真实动作 {mean['real_action']:.2f} dB"
        f" · 倒放动作 {mean['reversed_action']:.2f} dB"
        f" · 复制起始帧 {mean['copy_first_frame']:.2f} dB · 逐条比较真实动作胜倒放 {wins}/{len(rows)}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
