"""DW0.5 推演未来：拿 rollout 在仿真里真实跑出的轨迹，按真实动作推演画面，再与两个参照比。

    python -m dexbotic.so101.imagine --rollout-dir <rollout 的输出目录>

对照三行：仿真真值 / 按真实动作推演 / 按时间倒放的同一串动作推演。三种动作与「复制起始帧」
各自的平均 PSNR 写在 `<输出目录>/dw05_sim_check.json`，三行对照视频也在输出目录下。
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    from dexbotic.so101 import dw05_sim_check
    from dexbotic.so101.layout import DW05_BUNDLE, ROOT, WEIGHTS_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(ROOT) / "outputs" / "imagine")
    ap.add_argument("--per-scene", type=int, default=2, help="每个场景取几条轨迹")
    args = ap.parse_args()
    weights = pathlib.Path(WEIGHTS_DIR) / "so101-dw05"
    sys.argv = [
        "imagine",
        "--checkpoint",
        str(weights / "model.pt"),
        "--norm-stats",
        str(weights / "norm_stats.json"),
        "--bundle",
        DW05_BUNDLE,
        "--rollout-dir",
        str(args.rollout_dir),
        "--out",
        str(args.out),
        "--per-scene",
        str(args.per_scene),
    ]
    # dw05_sim_check 的退出码是给研究流水线用的判定；这里只看结果，不因数字高低报失败。
    dw05_sim_check.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
