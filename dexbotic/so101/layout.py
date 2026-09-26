"""SO101 工作区的目录约定：一个根目录推出数据、权重、产物三处落点。

两个 playground 入口与 `script/so101/` 下的脚本都从这里取默认值，每一项都可以用
同名环境变量单独覆盖。shell 脚本经 `env.sh` 执行本文件，拿到同一份 export 语句，
所以两种语言只有这一处定义。

    $SO101_ROOT/
    ├── datasets/   SO101_DATASETS_DIR  原始 LeRobot 数据与转换出的 dexdata
    │                                  （SO101_DEXDATA_DIR，默认其下 so101-dexdata/）
    ├── weights/    SO101_WEIGHTS_DIR   下载的底座权重（DW05_BUNDLE 默认在这下面）
    └── runs/       SO101_RUNS_ROOT     训练产物：存点、归一化统计、日志

产物放在代码检出之外：`git clean` 会连带删掉检出里的未追踪文件。
"""

import os
import shlex

ROOT = os.environ.get("SO101_ROOT", os.path.expanduser("~/so101_workspace"))
DATASETS_DIR = os.environ.get("SO101_DATASETS_DIR", f"{ROOT}/datasets")
#: 转换出的 dexdata；两个模型的训练都读它。jsonl 在其下 jsonl/，视频 url 相对 DATASETS_DIR。
DEXDATA_DIR = os.environ.get("SO101_DEXDATA_DIR", f"{DATASETS_DIR}/so101-dexdata")
WEIGHTS_DIR = os.environ.get("SO101_WEIGHTS_DIR", f"{ROOT}/weights")
RUNS_ROOT = os.environ.get("SO101_RUNS_ROOT", f"{ROOT}/runs")
DW05_BUNDLE = os.environ.get("DW05_BUNDLE", f"{WEIGHTS_DIR}/DW05-Robotwin")
#: `stage_to_shm.sh` 把热数据与大权重搬进来的 tmpfs 根目录。
SHM_ROOT = os.environ.get("SO101_SHM_ROOT", "/dev/shm/so101")

if __name__ == "__main__":
    for name, value in (
        ("SO101_ROOT", ROOT),
        ("SO101_DATASETS_DIR", DATASETS_DIR),
        ("SO101_DEXDATA_DIR", DEXDATA_DIR),
        ("SO101_WEIGHTS_DIR", WEIGHTS_DIR),
        ("SO101_RUNS_ROOT", RUNS_ROOT),
        ("DW05_BUNDLE", DW05_BUNDLE),
        ("SO101_SHM_ROOT", SHM_ROOT),
    ):
        print(f"export {name}={shlex.quote(value)}")
