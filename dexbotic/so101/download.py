"""下载权重与训练数据，落到工作区目录（默认 `~/so101_workspace`，改 `SO101_ROOT` 换位置）。

    python -m dexbotic.so101.download --weights    # 两份发布的权重 + DW0.5 推理用的基座组件
    python -m dexbotic.so101.download --data       # 训练数据 3698 集（仿真 1498 + 真机 2200）

各项钉死版本：数据换了版本，训练结果就和讲义里的不可比。
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

DM05_ADAPTER_REPO = "Harrysunshine/so101-dm05-lora-sim-real-10task"
DW05_REPO = "Harrysunshine/so101-dw05-sim-real-10task"
SIM_DATA_REPO, SIM_DATA_REVISION = (
    "Harrysunshine/so101-sim-pickplace-v2",
    "92ca801c614cda2481029122b75ed20c72f39495",
)
REAL_DATA_REPO, REAL_DATA_REVISION = "zhuzhuangtian/so101-pick-place-tasks", "f64493c4"
SIM_SCENES = ("cube40", "cube20", "cylinder40")


def tool(name: str) -> str:
    """同一环境里的命令行工具（`hf` / `modelscope`），不依赖 PATH 里装的是哪一个。"""
    return str(pathlib.Path(sys.executable).with_name(name))


def hf(repo: str, local_dir: pathlib.Path, *extra: str) -> None:
    subprocess.run([tool("hf"), "download", repo, "--local-dir", str(local_dir), *extra], check=True)


def weights() -> None:
    from dexbotic.so101.layout import DW05_BUNDLE, WEIGHTS_DIR

    root = pathlib.Path(WEIGHTS_DIR)
    # DM0.5 的基座 Dexmal/DM05 不用单独下：适配器里记着它，第一次加载时自动取到 HF 缓存。
    hf(DM05_ADAPTER_REPO, root / "so101-dm05-lora")
    hf(DW05_REPO, root / "so101-dw05")
    # 基座包自带一份 12 GB 的 model.pt；推理用我们微调后的权重，所以不下它。
    hf("Dexmal/DW05-Robotwin", pathlib.Path(DW05_BUNDLE), "--exclude", "model.pt")
    # 扩散主干不在基座包里，放进包内约定的位置。
    # 一条命令只给一个 --include：同一条命令给多个时实测只取回其中一部分，而且不报错。
    for pattern in ("*.json", "*.safetensors"):
        hf(
            "Wan-AI/Wan2.2-TI2V-5B",
            pathlib.Path(DW05_BUNDLE) / "Wan-AI" / "Wan2.2-TI2V-5B",
            "--include",
            pattern,
        )
    print(f"权重就位：{root}")


def data() -> None:
    from dexbotic.so101.layout import DATASETS_DIR

    root = pathlib.Path(DATASETS_DIR)
    for scene in SIM_SCENES:
        hf(
            SIM_DATA_REPO,
            root / "so101-sim-640-v2",
            "--repo-type",
            "dataset",
            "--revision",
            SIM_DATA_REVISION,
            "--include",
            f"{scene}/*",
        )
    subprocess.run(
        [
            tool("modelscope"),
            "download",
            "--dataset",
            REAL_DATA_REPO,
            "--revision",
            REAL_DATA_REVISION,
            "--local_dir",
            str(root / "so101-real"),
        ],
        check=True,
    )
    print(f"训练数据就位：{root}；下一步 python -m dexbotic.so101.prepare_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--data", action="store_true")
    args = ap.parse_args()
    if not (args.weights or args.data):
        ap.error("至少给 --weights 或 --data 之一")
    if args.weights:
        weights()
    if args.data:
        data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
