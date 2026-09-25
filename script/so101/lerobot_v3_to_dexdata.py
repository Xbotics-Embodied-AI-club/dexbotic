"""已移到 `dexbotic.so101.lerobot_v3_to_dexdata`；保留这个入口给现有的 shell 流水线。"""

from dexbotic.so101.lerobot_v3_to_dexdata import main

if __name__ == "__main__":
    raise SystemExit(main())
