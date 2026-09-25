# 由 script/so101/ 下的 shell 脚本 source：解释器默认值与目录约定。
# 目录约定只在 layout.py 里定义，这里执行它拿到 export 语句。
PY=${PY:-python}
# 先落进变量再 eval：`eval "$(...)"` 里的命令失败时 eval 仍返回 0，`set -e` 拦不住。
_so101_layout=$("$PY" -m dexbotic.so101.layout) ||
  { echo "[env] 用 PY=$PY 执行 layout.py 失败" >&2; exit 1; }
eval "$_so101_layout"

# 训练优先读内存里那份数据。只认 stage_to_shm.sh 逐项核过指纹后写下的标记，
# 不自己推断「看起来齐了」：jsonl 到位而视频没搬完时，训练会逐样本找不到文件。
# 没有标记就读原位数据，并打印读的是哪一份 —— 两者吞吐能差一倍，不打印就看不出原因。
use_staged_datasets() {
  if [ -f "$SO101_SHM_ROOT/datasets/.stage_complete" ]; then
    export SO101_DATASETS_DIR="$SO101_SHM_ROOT/datasets"
    echo "[data] 读内存里那份：$SO101_DATASETS_DIR"
  else
    echo "[data] 内存里没有完整副本，读原位数据：$SO101_DATASETS_DIR（先跑 stage_to_shm.sh 会快很多）"
  fi
}
