"""起 DM0.5 推理服务：收两路画面、关节状态和指令，返回接下来 50 步的绝对关节角。

    python -m dexbotic.so101.serve                       # 用发布的权重
    python -m dexbotic.so101.serve --checkpoint <目录>    # 用自己训的检查点

第一次运行会把 DM0.5 的基座下到 HF 缓存，要几分钟。服务就绪后另开一个终端跑 rollout。
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from dexbotic.so101 import dm05_exp
    from dexbotic.so101.layout import WEIGHTS_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=f"{WEIGHTS_DIR}/so101-dm05-lora")
    ap.add_argument("--port", type=int, default=7891)
    args = ap.parse_args()
    sys.argv = [
        "serve",
        "--task",
        "inference",
        "--model-config.model-name-or-path",
        args.checkpoint,
        "--inference-config.port",
        str(args.port),
    ]
    dm05_exp.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
