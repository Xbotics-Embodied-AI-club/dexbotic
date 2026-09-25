#!/usr/bin/env bash
# 把训练要反复读的数据搬进内存（tmpfs），训练脚本见到完成标记就改读这一份。
#
# dexdata 的读法是随机小读：每一步都要从共享 mp4 里按帧号解出若干帧，而 h264 的随机访问
# 要从最近的关键帧开始解。这种模式在网络存储上最吃亏，放进内存后吞吐能差一倍。
# 整份数据约 11 GB，先确认 `df -h /dev/shm` 放得下。
#
# 搬的是输入数据的只读副本。目录树逐级保留：dexdata 里视频的 `url` 相对数据根，
# 树一致才能只换根就生效。幂等：已在内存里且一致的不重搬；重启后 tmpfs 清空，重跑即可。
#
# 用 `--preserve=timestamps` 而不是 `-a`：tmpfs 不支持 ACL，`cp -a` 会为每个文件报
# "preserving permissions: Operation not supported" 并以非零退出，在 `set -e` 下
# 把已经拷好的搬运判成失败。
set -euo pipefail

source "$(dirname "$0")/env.sh"
SRC=$SO101_DATASETS_DIR
DST=$SO101_SHM_ROOT/datasets
SETS=${SETS:-"so101-dexdata so101-sim-640-v2 so101-real"}

# 按「文件数 + 总字节数」判要不要搬。
# 目录存在不等于完整：半途被杀会留下只有一半视频的目录。
# 文件数相同也不等于内容相同：源数据原地重渲后文件名与个数不变、只有内容变了，
# 而改分辨率必然改体积，总字节数是抓住这种情况最便宜的量。
_fingerprint() {  # 输出 "<文件数> <总字节>"，目录不存在时输出 "0 0"
  [ -d "$1" ] || { echo "0 0"; return; }
  find "$1" -type f -printf '%s\n' | awk '{n++; b+=$1} END {print n+0, b+0}'
}

# 训练在跑时不许删重搬：重搬的第一步是 `rm -rf`，训练会在删掉到拷回之间读不到文件，
# 崩在 dataloader 里，等去看时文件又回来了。一致时什么都不动，所以训练中途跑一次做校验是安全的。
_training_running() {
  pgrep -f "dm05_so101_xbotics.py" >/dev/null 2>&1 || pgrep -f "dw05_so101_exp.py" >/dev/null 2>&1
}

mkdir -p "$DST"
for name in $SETS; do
  [ -d "$SRC/$name" ] || { echo "[stage] 源不存在，跳过：$SRC/$name"; continue; }
  want=$(_fingerprint "$SRC/$name")
  have=$(_fingerprint "$DST/$name")
  if [ "$want" = "$have" ]; then
    echo "[stage] 已完整在内存里：$name（$want 文件/字节）"
    continue
  fi
  if [ "$have" != "0 0" ]; then
    if _training_running; then
      echo "[stage] 拒绝：$name 内存里是 [$have]、源是 [$want]，要删重搬，但训练正在跑。先停训练再来。" >&2
      exit 1
    fi
    echo "[stage] $name 内存里是 [$have]、源是 [$want]，不一致 ⇒ 删掉重搬"
    rm -rf "${DST:?}/$name"
  fi
  echo "[stage] 搬 $name（$want 文件/字节）..."
  cp -r --preserve=timestamps "$SRC/$name" "$DST/"
done

# 搬完逐个复核指纹再写标记。标记是唯一的「可以用了」声明，训练脚本只认它；
# 内容是各数据集的指纹，下一次能看出内存里那份对应源数据的哪个版本。
MARKER="$DST/.stage_complete"
rm -f "$MARKER"
ok=1
report=""
for name in $SETS; do
  [ -d "$SRC/$name" ] || continue
  a=$(_fingerprint "$SRC/$name")
  b=$(_fingerprint "$DST/$name")
  if [ "$a" = "$b" ]; then
    echo "[stage] $name  [$b]  一致"
    report="$report$name $b"$'\n'
  else
    echo "[stage] $name  内存 [$b] ≠ 源 [$a]  不一致"
    ok=0
  fi
done
if [ "$ok" = 1 ]; then
  printf '%s' "$report" > "$MARKER"
  echo "[stage] 全部一致，已写标记 $MARKER"
else
  echo "[stage] 有不一致的，不写标记 —— 训练会读原位数据，而不是一份对不上的副本" >&2
  exit 1
fi
du -sh "$DST"/* 2>/dev/null
df -h "$DST" | tail -1
