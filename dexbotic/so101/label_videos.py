"""把 rollout 存下的单局录像做成 top | wrist 并排的片子，顶上横幅写明来历。

横幅写清哪个模型、哪个存点、哪个场景、第几局、成没成、是策略在仿真里跑的 ——
没有这行字，策略 rollout 与数据集里同样两路并排的演示录像肉眼分不开。

用法：
    python -m dexbotic.so101.label_videos --rollout-dir <rollout 的输出目录> --out <目录> [--per-kind 2]
"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dexbotic.so101.client import read_frames


def banner(width: int, text: str, ok: bool) -> np.ndarray:
    img = Image.new("RGB", (width, 44), (20, 110, 40) if ok else (150, 30, 30))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((12, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollout-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--per-kind", type=int, default=2, help="成功、失败各出几段")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for kind in ("success", "fail"):
        tops = sorted(p for p in (args.rollout_dir / "videos").glob(f"*_{kind}.mp4"))[: args.per_kind]
        for top_path in tops:
            wrist = read_frames(top_path.with_name(top_path.stem + "_wrist.mp4"))
            top = read_frames(top_path)
            n = min(len(top), len(wrist))
            label, _, rest = top_path.stem.partition("_")  # dm05-checkpoint-5000_cube40_ep003_success
            scene, episode, _ = rest.rsplit("_", 2)
            where = "on the real SO-101" if scene == "real" else "in so101_sim"
            text = (
                f"{label} policy rollout {where} | {scene} {episode} | "
                f"{'SUCCESS' if kind == 'success' else 'FAIL'} | {n} steps @30fps | left: top  right: wrist"
            )
            head = banner(top.shape[2] + wrist.shape[2], text, kind == "success")
            frames = [
                np.concatenate([head, np.concatenate([top[t], wrist[t]], axis=1)], axis=0) for t in range(n)
            ]
            dst = args.out / f"{label}_{scene}_{episode}_{kind}.mp4"
            iio.imwrite(str(dst), np.stack(frames), fps=args.fps, codec="libx264")
            made.append(dst.name)
    print("\n".join(made) or "没有可用的录像")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
