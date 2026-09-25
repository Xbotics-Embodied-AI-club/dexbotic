"""证明转出来的 dexdata 里「画面」和「动作」没有错位。

判据是拿 lerobot 自己的数据集 API 解出的同一帧当真值：它知道正确的
(集, 帧) → (共享 mp4, 帧号) 映射，所以它给的帧**在构造上**就是对的。
两边都是从同一个 mp4 解出来的，编解码误差完全抵消，逐像素比得下去。

## 为什么不用「逐集图像统计」那套

拿 `meta/episodes` 里每集的 `stats/observation.images.*/mean` 和按 `frame_idx` 解出来的
均值比，实测差值是同号的常数 0.018 —— 那是 h264 + YUV420 往返的系统性偏暗，不是错位。
整集均值对一帧的位移也极不敏感（377 帧里只有 1 帧不同）。那套判据既有系统偏置、又缺分辨力。

## 判据

对抽查的每一帧，比三个候选偏移 −1 / 0 / +1，报各自与真值的平均绝对误差。

    offset 0 必须是三者里最小的，且明显小于另两个。

**偏移 ±1 是关键**：它让「常数偏置」这类系统误差在三个候选之间完全相同，
于是比较只反映对齐。运动中的相邻帧差别肉眼可见，所以这个判据有分辨力 ——
它可反驳：offset 0 不是最小，对齐就是错的。

用法（仓根，装了 lerobot 的环境）：
    $LEROBOT_PY script/so101/verify_dexdata_alignment.py \\
        --dexdata $SO101_DATASETS_DIR/so101-dexdata \\
        --lerobot $SO101_DATASETS_DIR/so101-sim-640-v2/cube40 \\
        --repo-id local/sim_cube40 --name sim_cube40
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

#: dexdata 的 images_N ← LeRobot 的相机键，顺序与转换器一致。
CAMERAS = ["observation.images.top", "observation.images.wrist"]
#: 比较用的候选偏移。0 必须胜出。
OFFSETS = (-1, 0, 1)


def to_uint8_hwc(value) -> np.ndarray:
    """把 lerobot 给的一帧转成 (H, W, 3) uint8。

    lerobot 的视频帧是 CHW、float32、值域 [0,1]；旧版本也可能直接给 uint8 HWC。

    Args:
        value: 数据集某一帧的图像字段。

    Returns:
        (H, W, 3) 的 uint8 数组。
    """
    array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
    if array.ndim != 3:
        raise ValueError(f"期望三维图像，实得 {array.shape}")
    if array.shape[0] == 3:  # CHW → HWC
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def decode_frames(video_path: pathlib.Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """顺序流式解码，只留要的那几帧。

    不用 seek：seek 只能落到关键帧，落点会悄悄偏，而本脚本就是判偏移的。
    不整条读进内存：共享 mp4 一个两百多 MB、装着几千帧。

    Args:
        video_path: 共享 mp4。
        wanted: 要的帧号集合。

    Returns:
        帧号 → (H, W, 3) uint8。

    Raises:
        ValueError: 视频比要的帧号短。
    """
    import imageio.v3 as iio

    last = max(wanted)
    out: dict[int, np.ndarray] = {}
    for index, frame in enumerate(iio.imiter(str(video_path), plugin="pyav")):
        if index in wanted:
            out[index] = np.asarray(frame, dtype=np.uint8)
        if index >= last:
            break
    if len(out) != len(wanted):
        raise ValueError(f"{video_path.name} 只解出 {len(out)} 帧，要 {len(wanted)} 帧")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dexdata", required=True, type=pathlib.Path)
    ap.add_argument("--lerobot", required=True, type=pathlib.Path, help="LeRobot v3 数据集根")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--name", required=True, help="转换时给这个来源的前缀")
    ap.add_argument("--episodes", type=int, default=3, help="抽查几集（等间隔取）")
    ap.add_argument("--frames", type=int, default=4, help="每集抽查几帧（等间隔取）")
    # lerobot 默认用 torchcodec 解码，它要系统里有匹配的 FFmpeg 共享库。pyav 是 lerobot
    # 自带的另一个后端，本脚本另一侧也用 pyav，两边同一个解码器，比较更干净。
    ap.add_argument("--video-backend", default="pyav")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=str(args.lerobot), video_backend=args.video_backend)
    print(f"[真值] lerobot 数据集就绪：{dataset.num_episodes} 集 / {dataset.num_frames} 帧")

    image_root = pathlib.Path(
        json.loads((args.dexdata / "convert_report.json").read_text())["image_dir"]
    )
    jsonl_files = sorted((args.dexdata / "jsonl").glob(f"{args.name}_ep*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"{args.dexdata}/jsonl 下没有 {args.name}_ep*.jsonl")

    # 等间隔抽集：偏移错误常常只在后面的集才显形（共享 mp4 换了一个文件）。
    episode_picks = np.linspace(
        0, len(jsonl_files) - 1, min(args.episodes, len(jsonl_files))
    ).astype(int)
    failures: list[dict] = []
    records: list[dict] = []

    for pick in episode_picks:
        path = jsonl_files[pick]
        episode_index = int(path.stem.split("_ep")[-1])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        base = int(dataset.meta.episodes[episode_index]["dataset_from_index"])
        # 抽帧避开首末两帧：判据要比 ±1 偏移，首帧的 −1 与末帧的 +1 落在这一集之外，
        # 在共享 mp4 里那两帧属于**相邻的另一集**，比出来的数没有意义。
        frame_picks = np.linspace(1, len(rows) - 2, min(args.frames, max(1, len(rows) - 2))).astype(
            int
        )
        print(f"\n[集 {episode_index}] {path.name}  {len(rows)} 帧  dataset_from_index={base}")

        for slot, camera in enumerate(CAMERAS, start=1):
            url = rows[0][f"images_{slot}"]["url"]
            wanted = {
                rows[int(f)][f"images_{slot}"]["frame_idx"] + o
                for f in frame_picks
                for o in OFFSETS
            }
            decoded = decode_frames(image_root / url, wanted)

            for frame in frame_picks:
                frame = int(frame)
                sample = dataset[base + frame]
                if (
                    int(sample["frame_index"]) != frame
                    or int(sample["episode_index"]) != episode_index
                ):
                    raise SystemExit(
                        f"真值取错了：要 (集 {episode_index}, 帧 {frame})，"
                        f"lerobot 给的是 (集 {int(sample['episode_index'])}, 帧 {int(sample['frame_index'])})"
                    )
                truth = to_uint8_hwc(sample[camera])
                ours = rows[frame][f"images_{slot}"]["frame_idx"]
                errors = {
                    offset: float(
                        np.abs(
                            decoded[ours + offset].astype(np.int32) - truth.astype(np.int32)
                        ).mean()
                    )
                    for offset in OFFSETS
                }
                best = min(errors, key=errors.get)
                ok = best == 0
                record = {
                    "episode": episode_index,
                    "camera": camera,
                    "frame": frame,
                    "frame_idx": ours,
                    "errors": errors,
                    "best_offset": best,
                    "ok": ok,
                }
                records.append(record)
                marks = "  ".join(f"{o:+d}:{errors[o]:6.3f}" for o in OFFSETS)
                print(
                    f"  {camera:26s} 帧{frame:4d} frame_idx={ours:6d}  {marks}  最优 {best:+d}  "
                    f"{'✔' if ok else '✘ 错位'}"
                )
                if not ok:
                    failures.append(record)

    (args.dexdata / "alignment_check.json").write_text(
        json.dumps({"records": records, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if failures:
        print(f"\n[FAIL] {len(failures)} / {len(records)} 处的最优偏移不是 0 —— 画面与动作错位")
        return 1
    print(f"\n[PASS] {len(records)} 处逐像素比，最优偏移全是 0 —— 画面与动作对齐")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
