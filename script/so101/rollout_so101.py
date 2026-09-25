"""在三个仿真场景上跑策略，出成功率与每局 rollout 视频。

## 为什么要跨进程

仿真与推理装不进同一个环境：仿真那套是 ManiSkill/SAPIEN + torch 2.8，
Dexbotic 要 torch 2.11 + transformers 5.3。所以本脚本只跑在仿真那一侧，
动作向一个 HTTP 推理服务要 —— 上游自带这个服务（`--task inference` 起 Flask，
`POST /v1/infer`）。

    # 训练侧环境：起推理服务
    CUDA_VISIBLE_DEVICES=0 $PY playground/dm05_so101_xbotics.py --task inference \\
        --model-config.model-name-or-path <checkpoint>
    # 仿真侧环境：跑评测
    CUDA_VISIBLE_DEVICES=1 $SIM_PY script/so101/rollout_so101.py --out <目录> \\
        --action-mode absolute --episodes 50

## 动作口径

仿真环境要绝对关节角（`control_mode="pd_joint_pos"`，数据集录的就是绝对角）。
服务返回的是绝对角还是相对发请求那一刻状态的增量，由 `--action-mode` 声明，没有默认值。
DM0.5 按 RELATIVE 训练时，推理服务已经把状态加回去、返回绝对角，所以用 `absolute`；
`dw05_policy_server.py` 同样返回绝对角。

这一项配错不报错：绝对角被当成增量会再加一次状态，目标角约成两倍；增量被当成绝对角，
手臂几乎不动。两种都只表现为很低的成功率，会被误读成「策略没学会」。
本脚本把它写进报告，事后能分清那一轮用的是哪个。

## 三个场景都要跑

只测 cube40 会漏掉小物体（cube20）与圆面抓取（cylinder40）两种失效模式。

## 判据

成功率取环境自己的 `is_success`（物体落在料箱内 ∧ 夹爪已松开 ∧ 物体静止 ∧ 机器人静止），
不另立判据。同时存每局 mp4 与逐步状态，供事后查看与 DW0.5 自检复用。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import time

import numpy as np

TASKS = {
    "cube40": "SO101PickPlaceCube40-v1",
    "cube20": "SO101PickPlaceCube20-v1",
    "cylinder40": "SO101PickPlaceCylinder40-v1",
}
#: 数据集录的是绝对关节角，环境必须用这个控制模式。
CONTROL_MODE = "pd_joint_pos"
#: dexdata 的 images_1 = top、images_2 = wrist，与训练时的槽位一致。顺序错了不报错。
IMAGE_SLOTS = ("top", "wrist")


def encode_image(image: np.ndarray) -> str:
    """把一帧 RGB 编成推理服务要的 base64 PNG。"""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def request_actions(
    endpoint: str, images: dict, state: np.ndarray, prompt: str, timeout: float
) -> np.ndarray:
    """向推理服务要一个动作块。

    Args:
        endpoint: 形如 `http://127.0.0.1:7891/v1/infer`。
        images: 相机名 → RGB 帧。按 `IMAGE_SLOTS` 的顺序填 1 基槽位。
        state: 当前关节角（绝对，度）。
        prompt: 任务指令。
        timeout: 单次请求超时（秒）。

    Returns:
        形状 (chunk, dof) 的动作块。

    Raises:
        RuntimeError: 服务返回非 200，或响应里没有 `actions`。
    """
    import urllib.error
    import urllib.request

    payload = {
        "observation": {
            "images": {
                str(slot): encode_image(images[name])
                for slot, name in enumerate(IMAGE_SLOTS, start=1)
            },
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "prompt": prompt,
        }
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"推理服务返回 {error.code}：{error.read()[:300]!r}") from error
    actions = body.get("actions")
    if actions is None:
        raise RuntimeError(f"响应里没有 actions，实收键 {sorted(body)}")
    return np.asarray(actions, dtype=np.float32).reshape(-1, np.shape(state)[-1])


def to_absolute(chunk: np.ndarray, state: np.ndarray, action_mode: str) -> np.ndarray:
    """把动作块换成环境要的绝对关节角。

    Args:
        chunk: 推理服务给的动作块。
        state: 发出请求时的关节角。
        action_mode: 推理服务返回的口径，`"relative"` 或 `"absolute"`。

    Returns:
        绝对关节角序列，形状与 `chunk` 相同。

    Raises:
        ValueError: `action_mode` 不是这两个值之一。
    """
    if action_mode == "absolute":
        return chunk
    if action_mode != "relative":
        raise ValueError(f"action_mode 只能是 relative 或 absolute，收到 {action_mode!r}")
    # 相对量是「相对发出请求那一刻的状态」的位移，整块共用同一个基准 ——
    # 逐步累加会把一个动作块内的位移叠成好几倍。
    return np.asarray(state, dtype=np.float32)[None, :] + chunk


def run_episode(
    env, endpoint: str, prompt: str, action_mode: str, replan: int, max_steps: int, timeout: float
) -> tuple[bool, dict, int]:
    """跑一集。

    Args:
        env: `So101SimEnv` 实例。
        endpoint: 推理服务地址。
        prompt: 任务指令。
        action_mode: 见 `to_absolute`。
        replan: 一个动作块最多执行几步就重新请求（其余丢弃）。
        max_steps: 单集步数上限。
        timeout: 单次推理请求超时。

    Returns:
        `(是否成功, 轨迹, 实际走的步数)`。轨迹含逐帧两路画面、执行前状态与所发绝对动作，
        供 DW0.5 在没见过的仿真轨迹上做「真动作 vs 基线」的验证。
    """
    obs, _ = env.reset()
    traj = {name: [obs["pixels"][name].copy()] for name in IMAGE_SLOTS}
    traj["state"], traj["action"] = [], []
    success = False
    steps = 0
    while steps < max_steps:
        chunk = request_actions(endpoint, obs["pixels"], obs["agent_pos"], prompt, timeout)
        absolute = to_absolute(chunk, obs["agent_pos"], action_mode)
        for action in absolute[:replan]:
            traj["state"].append(np.asarray(obs["agent_pos"], dtype=np.float32).copy())
            traj["action"].append(np.asarray(action, dtype=np.float32))
            obs, _, terminated, truncated, info = env.step(action)
            for name in IMAGE_SLOTS:
                traj[name].append(obs["pixels"][name].copy())
            steps += 1
            success = success or bool(info["is_success"])
            if terminated or truncated or steps >= max_steps:
                break
        if success or steps >= max_steps:
            break
    return success, traj, steps


def save_mp4(path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
    """存一集的 rollout 视频。"""
    import imageio.v3 as iio

    iio.imwrite(str(path), np.stack(frames), fps=fps, codec="libx264")


def read_frames(path: pathlib.Path) -> np.ndarray:
    """把 `save_mp4` 存下的视频整段解成 (T, H, W, 3) uint8。

    直接用 av：imageio 的 pyav 插件在新版 av 上读完即抛
    `VideoCodecContext has no attribute close`。
    """
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--endpoint", default="http://127.0.0.1:7891/v1/infer")
    ap.add_argument("--label", default="dm05", help="报告与 wandb run 名里的模型标识")
    ap.add_argument(
        "--action-mode",
        required=True,
        choices=["relative", "absolute"],
        help="推理服务返回的动作口径；配错不报错，只是成功率很低",
    )
    ap.add_argument("--episodes", type=int, default=20, help="每个场景跑几集")
    # 同一存点三场景实测：每块执行 25 步 11/30，8 步 3/30。
    ap.add_argument("--replan", type=int, default=25, help="一个动作块执行几步就重新请求")
    ap.add_argument("--max-steps", type=int, default=500, help="单局步数上限")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--fps", type=int, default=30, help="存视频的帧率，与数据集一致")
    ap.add_argument("--scenes", nargs="*", default=sorted(TASKS))
    ap.add_argument("--wandb-project", default="xbotics_so101_rollout")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument(
        "--log-videos",
        type=int,
        default=4,
        help="每个场景往 wandb 传几集视频（成功与失败各尽量取一半）",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    video_dir = args.out / "videos"
    video_dir.mkdir(exist_ok=True)

    import so101_sim  # noqa: F401  —— import 即注册三个场景
    from so101_sim.lerobot_env import So101SimEnv

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.label}-rollout",
            config={k: str(v) for k, v in vars(args).items()} | {"control_mode": CONTROL_MODE},
            mode=args.wandb_mode,
        )
        print(f"[wandb] {run.url}")

    report = {
        "label": args.label,
        "action_mode": args.action_mode,
        "control_mode": CONTROL_MODE,
        "episodes_per_scene": args.episodes,
        "replan": args.replan,
        "scenes": {},
    }
    for scene in args.scenes:
        if scene not in TASKS:
            raise SystemExit(f"不认识的场景 {scene}；可选 {sorted(TASKS)}")
        # 不传分辨率：用环境自己标定的 640×480，与训练数据同构。
        # auto_reset=False：成功那一步环境不许自己换场景，集的边界由本脚本管。
        env = So101SimEnv(
            task=TASKS[scene],
            obs_type="pixels_agent_pos",
            control_mode=CONTROL_MODE,
            episode_length=args.max_steps,
            auto_reset=False,
        )
        # 指令取环境自己那张表（与数据集 tasks 逐字相同）。`env.task` 是 ManiSkill 的环境 id，
        # 拿它当 prompt 不报错，只会量出「策略没见过这句话」时的成功率。
        prompt = env.task_description
        # 渲染自检：与别的作业共用的卡可能渲出成片纯黑块，干净画面这一数是 0~5。
        # 脏图不报错、只把成功率压低，所以在第一帧就拦下，换卡重跑。
        probe, _ = env.reset()
        speckle = max(
            int((probe["pixels"][name].astype(int).sum(-1) < 20).sum()) for name in IMAGE_SLOTS
        )
        if speckle > 1000:
            raise SystemExit(
                f"[{scene}] 渲染带黑斑（{speckle} 个纯黑像素），这张卡不能用来评测，换 SIM_GPUS"
            )
        successes, step_counts = [], []
        # 传上去的视频要**成败都有**：只传成功的会让人以为没有失败模式。
        clips: dict[str, list[pathlib.Path]] = {"success": [], "fail": []}
        for episode in range(args.episodes):
            started = time.monotonic()
            success, traj, steps = run_episode(
                env,
                args.endpoint,
                prompt,
                args.action_mode,
                args.replan,
                args.max_steps,
                args.timeout,
            )
            successes.append(success)
            step_counts.append(steps)
            tag = "success" if success else "fail"
            clip_path = video_dir / f"{args.label}_{scene}_ep{episode:03d}_{tag}.mp4"
            save_mp4(clip_path, traj["top"], args.fps)
            save_mp4(clip_path.with_name(clip_path.stem + "_wrist.mp4"), traj["wrist"], args.fps)
            np.savez(
                clip_path.with_suffix(".npz"),
                state=np.stack(traj["state"]),
                action=np.stack(traj["action"]),
                prompt=prompt,
            )
            if len(clips[tag]) < max(1, args.log_videos // 2):
                clips[tag].append(clip_path)
            print(
                f"[{scene}] 第 {episode:3d} 集  {tag:7s} {steps:4d} 步  "
                f"{time.monotonic() - started:5.1f}s  累计成功率 {np.mean(successes):.3f}"
            )
        env.close()
        rate = float(np.mean(successes))
        report["scenes"][scene] = {
            "task": TASKS[scene],
            "episodes": len(successes),
            "successes": int(np.sum(successes)),
            "success_rate": rate,
            "mean_steps": float(np.mean(step_counts)),
        }
        print(f"[{scene}] 成功率 {int(np.sum(successes))}/{len(successes)} = {rate:.3f}")
        if run is not None:
            payload = {
                f"success_rate/{scene}": rate,
                f"mean_steps/{scene}": float(np.mean(step_counts)),
            }
            for tag, paths in clips.items():
                for index, path in enumerate(paths):
                    payload[f"rollout/{scene}/{tag}_{index}"] = wandb.Video(
                        str(path), fps=args.fps, format="mp4"
                    )
            run.log(payload)

    rates = [entry["success_rate"] for entry in report["scenes"].values()]
    report["success_rate_mean_over_scenes"] = float(np.mean(rates)) if rates else 0.0
    (args.out / f"rollout_{args.label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n三场景平均成功率 {report['success_rate_mean_over_scenes']:.3f} → {args.out}")
    if run is not None:
        run.log({"success_rate/mean": report["success_rate_mean_over_scenes"]})
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
