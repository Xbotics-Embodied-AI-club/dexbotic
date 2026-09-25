"""把 LeRobot v3 数据集转成 Dexbotic 的 dexdata 格式（SO101，单臂 6 维）。

## 为什么不能直接用上游那个转换器

上游有 `script/convert_data/convert_lerobot_to_dexdata.py`，外层骨架（遍历、写 jsonl、
搬视频）是通用的，dexdata 的格式定义也有权威文档（`docs/web_docs/5. Use Custom Data.md`）。
但它是 **v2 时代**写给 galaxea 双臂移动底盘的，三处对不上：

1. 字段：它读 `observation.state.left_arm` / `right_arm` / `chassis` / `torso`；
   SO101 数据是扁平的 `observation.state` / `action`，各 6 维（5 关节 + 夹爪）。
2. 任务表：它读 `meta/tasks.jsonl`；v3 换成了 `meta/tasks.parquet`。
3. **文件粒度 —— 这条最要命。** 它假设「一集一个 parquet、一集一个 mp4」；
   v3 是「多集共用一个 parquet、多集共用一个 mp4」。实测 cube40：465 集全在
   `data/chunk-000/file-000.parquet` 一个文件里，视频侧每路相机 512 个 mp4、
   每个装若干集。⇒ 照 v2 的写法算 `frame_idx`，画面与动作会**整段错位**，
   而训练照样收敛、只是评测时莫名其妙 —— 这类错没有任何一步会报错。

## 时间对齐（本文件的核心）

每集在共享 mp4 里的位置由 `meta/episodes` 那张表给出：
`videos/<相机>/{chunk_index,file_index,from_timestamp,to_timestamp}`。
所以

    frame_idx = round(from_timestamp * fps) + 行内的 frame_index

**不复制、不重编码视频** —— dexdata 的图像字段原生支持
`{"type":"video","url":...,"frame_idx":...}`，指向共享 mp4 就够了。

判据（每集都查，不抽样）：`round((to - from) * fps)` 必须等于 `length - 1`（闭区间，
`to_timestamp` 是末帧时刻）或 `length`（开区间，`to_timestamp` 在末帧之后）。
两份数据集的约定不同 —— HF 上的仿真数据集是闭区间，ModelScope 上的真机数据集是开区间，
所以两者都放行，并把实际用的约定记进转换报告。别的值就说明上面那个公式在这份数据上
不适用，当场报错停下：继续转下去只会产出一份安静错位的数据集。

偏移本身只用 `from_timestamp`，不受这条差异影响；对齐的最终证据是
`verify_dexdata_alignment.py` 拿 lerobot 自己的数据集 API 解出同一帧做逐像素比。

## 相机顺序

`dexbotic/data/dataset_dm05/so101.py` 注册的是 `images_1` / `images_2`，
提示词是 `["Head", "Left wrist"]`。所以 **`images_1` = top，`images_2` = wrist**。
换顺序不报错，只是每个视角的语义和预训练时对不上。

## 一条记录同时喂两个模型（并集写法）

DM0.5 与 DW0.5 吃的**不是同一个 schema**，所以每条记录两套字段都写，互不干扰：

    DM0.5 读扁平的  prompt · state · action · is_robot · images_N
    DW0.5 读嵌套的  type · robot{prompt,state,subtask} · worldmodel{caption}
                    · robot_task_success · images_N

两处缺字段的报错都不直接指向原因：

1. 少了 `type` ⇒ DW0.5 逐样本抛 `Expected frame type list`。
2. 只把 `state` / `prompt` 放在顶层 ⇒ 变换链第一步 `ToList` 就 `NoneType has no len()`：
   它按 `worldmodel` / `robot` / `conversations` 里任意一个的长度来定帧数，三个都没有就崩。

还有一件容易误判的：**DW0.5 不读顶层的 `action`**。它的动作是变换链从相邻帧的
`robot.state` 现算的（`AddAction` → `AddTrajectory` → `DeltaAction`）。顶层那份 `action`
只有 DM0.5 在用，两者算出来的量不必相同，各按自己的口径走。

## 混训：多个来源写进同一个输出目录

`JsonlDataset` 只吃**一个** `jsonl_dir`，所以仿真与真机要混训就写进同一个目录，
共享一份归一化统计 —— 同一台机器人、同样 6 维动作空间、同样两路相机、同样 30 fps，
本来就该共享。不同任务靠每帧的 `prompt` 区分。
jsonl 文件名带来源前缀，避免两边的 `episode_00000` 撞名。

视频 `url` 是相对 `image_dir` 的，所以 `--image-root` 要给一个能同时覆盖两个来源的
公共父目录，脚本会自己算出各来源相对它的前缀。

日常用 `convert_all.sh` 一次转全部来源；单独调用时：
    D=$SO101_DATASETS_DIR
    $PY script/so101/lerobot_v3_to_dexdata.py --image-root $D --out $D/so101-dexdata \\
        --source $D/so101-sim-640-v2/cube40  --name sim_cube40 \\
        --source $D/so101-sim-640-v2/cube20  --name sim_cube20
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

#: dexdata 的 images_N ← LeRobot 的相机键。顺序即语义，见模块 docstring。
CAMERA_ORDER = ["observation.images.top", "observation.images.wrist"]

#: 每帧的 `type`。DW0.5 按这个列表决定加载哪几支：`action` 走机器人数据（归一化统计
#: 与动作专家都要它），`wm` 走未来画面。两者是独立分支、结果合并，所以同时给。
SAMPLE_TYPES = ["action", "wm"]

#: DW0.5 的视频分支要一句 caption。与 `dw05_policy.ROBOTWIN_PROMPT_FORMAT` 同形，
#: 所以和预训练时的文本分布一致。
CAPTION_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {instruction}"

#: 两份数据集都是成功轨迹（脚本化生成 + 遥操作演示），所以恒为 1。
#: 有失败样本进来时这里必须按真值写 —— 恒 1 会让世界模型把失败也当成该学的未来。
ROBOT_TASK_SUCCESS = 1


def load_info(root: pathlib.Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def load_episodes(root: pathlib.Path):
    """把 meta/episodes 下所有分片拼成一张表，只取对齐要用的那几列。"""
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"{root} 下没有 meta/episodes/*.parquet —— 这不是 LeRobot v3 数据集"
        )
    columns = ["episode_index", "length", "tasks"]
    for camera in CAMERA_ORDER:
        columns += [
            f"videos/{camera}/{field}"
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ]
    frames = [pq.read_table(f, columns=columns).to_pandas() for f in files]
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def load_tasks(root: pathlib.Path) -> dict[int, str]:
    """`task_index` → 任务指令。

    一律走 `meta/tasks.parquet`，不用 data 里那一列 `task` —— 那一列是可选的：
    HF 上的仿真数据集有，ModelScope 上的真机数据集没有。走任务表则两边同一条路径。

    Args:
        root: LeRobot v3 数据集根。

    Returns:
        task_index → 指令字符串。
    """
    table = pq.read_table(root / "meta" / "tasks.parquet").to_pandas().reset_index()
    columns = set(table.columns)
    if not {"task_index", "task"} <= columns:
        raise ValueError(
            f"{root}/meta/tasks.parquet 缺 task_index 或 task 列，实有 {sorted(columns)}"
        )
    return {int(row["task_index"]): str(row["task"]) for _, row in table.iterrows()}


def load_data(root: pathlib.Path):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root} 下没有 data/**/*.parquet")
    columns = ["action", "observation.state", "frame_index", "episode_index", "task_index"]
    frames = [pq.read_table(f, columns=columns).to_pandas() for f in files]
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def video_relpath(
    root: pathlib.Path, image_root: pathlib.Path, camera: str, chunk: int, file_index: int
) -> str:
    absolute = (
        root / "videos" / camera / f"chunk-{int(chunk):03d}" / f"file-{int(file_index):03d}.mp4"
    )
    if not absolute.is_file():
        raise FileNotFoundError(f"视频不存在：{absolute}")
    return absolute.relative_to(image_root).as_posix()


def convert_source(
    root: pathlib.Path,
    name: str,
    image_root: pathlib.Path,
    out_jsonl_dir: pathlib.Path,
) -> dict:
    info = load_info(root)
    fps = float(info["fps"])
    episodes = load_episodes(root)
    tasks = load_tasks(root)
    data = load_data(root)
    grouped = data.groupby("episode_index", sort=True)

    written = 0
    total_frames = 0
    conventions: set[str] = set()
    # 列名里有点和斜杠，`itertuples` 会把它们改写成位置名 `_3` 这类东西 ——
    # 按名字取字段会静默拿错列。所以 episodes 走 dict（只有几百行），
    # data 走显式列序的纯元组，取值靠这里定的顺序，不靠 pandas 的改写规则。
    for row in episodes.to_dict("records"):
        episode_index = int(row["episode_index"])
        length = int(row["length"])

        # 每路相机在共享 mp4 里的起始帧。判据不成立就停下——见模块 docstring。
        offsets, urls = [], []
        for camera in CAMERA_ORDER:
            start = float(row[f"videos/{camera}/from_timestamp"])
            end = float(row[f"videos/{camera}/to_timestamp"])
            span = round((end - start) * fps)
            # `to_timestamp` 有两种约定：闭区间（末帧时刻，span = length−1）与
            # 开区间（末帧之后，span = length）。两种都合法，别的值就是算法不成立。
            # 偏移本身只用 `from_timestamp`，不受这条差异影响。
            if span not in (length - 1, length):
                raise ValueError(
                    f"{name} 第 {episode_index} 集 {camera}：按时间戳算出跨度 {span} 帧，"
                    f"表里写的是 {length} 帧（差 {span - length}）—— 既不是闭区间也不是开区间，"
                    "frame_idx 的算法在这份数据上不成立，停下"
                )
            conventions.add("closed" if span == length - 1 else "open")
            offsets.append(int(round(start * fps)))
            urls.append(
                video_relpath(
                    root,
                    image_root,
                    camera,
                    row[f"videos/{camera}/chunk_index"],
                    row[f"videos/{camera}/file_index"],
                )
            )

        episode_rows = grouped.get_group(episode_index).sort_values("frame_index")
        if len(episode_rows) != length:
            raise ValueError(
                f"{name} 第 {episode_index} 集：data 里 {len(episode_rows)} 行，"
                f"episodes 表里 {length} 帧 —— 两处对不上，停下"
            )

        lines = []
        columns = ["frame_index", "observation.state", "action", "task_index"]
        for frame_index, state, action, task_index in episode_rows[columns].itertuples(
            index=False, name=None
        ):
            if int(task_index) not in tasks:
                raise ValueError(
                    f"{name} 第 {episode_index} 集出现未登记的 task_index {task_index}"
                )
            record = {}
            for slot, (url, offset) in enumerate(zip(urls, offsets, strict=True), start=1):
                record[f"images_{slot}"] = {
                    "type": "video",
                    "url": url,
                    "frame_idx": offset + int(frame_index),
                }
            instruction = tasks[int(task_index)]
            state_list = np.asarray(state, dtype=np.float32).tolist()
            # 扁平字段给 DM0.5。
            record["prompt"] = instruction
            record["state"] = state_list
            record["action"] = np.asarray(action, dtype=np.float32).tolist()
            record["is_robot"] = True
            # 嵌套字段给 DW0.5 —— 它的训练格式与 DM0.5 的**不是一个 schema**，见模块 docstring。
            record["type"] = SAMPLE_TYPES
            record["robot"] = {"prompt": instruction, "state": state_list, "subtask": instruction}
            record["worldmodel"] = {"caption": CAPTION_TEMPLATE.format(instruction=instruction)}
            record["robot_task_success"] = ROBOT_TASK_SUCCESS
            lines.append(json.dumps(record, ensure_ascii=False))

        (out_jsonl_dir / f"{name}_ep{episode_index:05d}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        written += 1
        total_frames += len(lines)

    return {
        "source": str(root),
        "name": name,
        "episodes": written,
        "frames": total_frames,
        "fps": fps,
        "to_timestamp_convention": sorted(conventions),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        required=True,
        type=pathlib.Path,
        help="dexdata 输出目录（jsonl 都写这里面的 jsonl/）",
    )
    ap.add_argument(
        "--image-root",
        required=True,
        type=pathlib.Path,
        help="视频 url 的相对基准；必须是所有来源的公共父目录（= 训练配置里的 image_dir）",
    )
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        type=pathlib.Path,
        help="一个 LeRobot v3 数据集根",
    )
    ap.add_argument(
        "--name",
        action="append",
        required=True,
        help="与 --source 一一对应的来源名，用作 jsonl 文件名前缀",
    )
    args = ap.parse_args()

    if len(args.source) != len(args.name):
        raise SystemExit(
            f"--source 给了 {len(args.source)} 个，--name 给了 {len(args.name)} 个，必须一一对应"
        )

    out_jsonl_dir = args.out / "jsonl"
    out_jsonl_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for root, name in zip(args.source, args.name, strict=True):
        print(f"[转] {name} ← {root}")
        report = convert_source(root.resolve(), name, args.image_root.resolve(), out_jsonl_dir)
        print(f"     {report['episodes']} 集 / {report['frames']} 帧")
        reports.append(report)

    summary = {
        "image_dir": str(args.image_root.resolve()),
        "jsonl_dir": str(out_jsonl_dir),
        "camera_order": CAMERA_ORDER,
        "sources": reports,
        "episodes_total": sum(r["episodes"] for r in reports),
        "frames_total": sum(r["frames"] for r in reports),
    }
    (args.out / "convert_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n[done] 共 {summary['episodes_total']} 集 / {summary['frames_total']} 帧 → {out_jsonl_dir}"
    )
    print("       训练配置里 jsonl_dir 指这里，image_dir 指 --image-root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
