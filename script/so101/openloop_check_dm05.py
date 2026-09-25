"""已移到 `dexbotic.so101.openloop_check`；保留这个入口给现有的 shell 流水线。"""

from dexbotic.so101.openloop_check import main

if __name__ == "__main__":
    raise SystemExit(main())
