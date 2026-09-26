"""把下载好的仿真与真机数据转成 DM0.5 / DW0.5 训练用的统一格式（dexdata）。

    python -m dexbotic.so101.prepare_data

输入（`dexbotic.so101.download --data` 下好的位置）：
    <数据目录>/so101-sim-640-v2/{cube40,cube20,cylinder40}   仿真三个场景
    <数据目录>/so101-real/<任务名>/                           真机九个任务
输出：<数据目录>/so101-dexdata/（jsonl 标注，画面仍引用原视频）

仿真与真机进同一份：同一台机器人、同样的六维动作、同样两路相机和帧率，本来就该共用一份
归一化统计；不同任务靠每一帧的指令区分。
"""

from __future__ import annotations

import pathlib
import sys

SIM_SCENES = ("cube40", "cube20", "cylinder40")


def main() -> int:
    from dexbotic.so101 import lerobot_v3_to_dexdata
    from dexbotic.so101.layout import DATASETS_DIR

    root = pathlib.Path(DATASETS_DIR)
    sim, real = root / "so101-sim-640-v2", root / "so101-real"
    for directory in (sim, real):
        if not directory.is_dir():
            raise SystemExit(f"数据不在 {directory}；先运行 python -m dexbotic.so101.download --data")
    argv = ["--out", str(root / "so101-dexdata"), "--image-root", str(root)]
    for scene in SIM_SCENES:
        argv += ["--source", str(sim / scene), "--name", f"sim_{scene}"]
    # 目录名就是任务名；README 之类的文件不是任务。
    for task in sorted(p for p in real.iterdir() if p.is_dir()):
        argv += ["--source", str(task), "--name", f"real_{task.name}"]
    print(f"共 {len(argv) // 4} 个来源 → {root / 'so101-dexdata'}")
    sys.argv = ["lerobot_v3_to_dexdata", *argv]
    return lerobot_v3_to_dexdata.main()


if __name__ == "__main__":
    raise SystemExit(main())
