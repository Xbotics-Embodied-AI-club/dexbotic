"""SO-101 这一侧与 DM0.5 推理服务、与抓放录像文件打交道的几个小函数。

录像文件的格式只在这里定义一次：写的一方（控制循环）和读的一方（DW0.5 推演自检、出片）都用这一份。
一局存成三个文件，共用一个文件名主干 `<标签>_<场景简称>_ep<局号>_<success|fail>`：
主干`.mp4` 是顶视、主干`_wrist.mp4` 是腕部、主干`.npz` 存逐步的关节状态 `state`、所发动作 `action`
与指令 `prompt`。
"""

from __future__ import annotations

import base64
import io
import json
import pathlib

import numpy as np

#: dexdata 的 images_1 = top、images_2 = wrist，与训练时的槽位一致。顺序错了不报错。
IMAGE_SLOTS = ("top", "wrist")

#: 仿真场景的注册名 → 简称。简称用在录像文件名、数据集目录名和按场景分组的地方。
SCENES = {
    "SO101PickPlaceCube40-v1": "cube40",
    "SO101PickPlaceCube20-v1": "cube20",
    "SO101PickPlaceCylinder40-v1": "cylinder40",
}


def infer_endpoint(port: int, host: str = "127.0.0.1") -> str:
    """DM0.5 推理服务的请求地址。"""
    return f"http://{host}:{port}/v1/infer"


def episode_stem(label: str, scene: str, episode: int, success: bool) -> str:
    """一局录像的文件名主干，见模块说明。"""
    return f"{label}_{scene}_ep{episode:03d}_{'success' if success else 'fail'}"


def parse_episode_stem(stem: str) -> tuple[str, str, str, str]:
    """`episode_stem` 的逆：`(标签, 场景简称, 'ep<局号>', 'success'|'fail')`。

    标签里可以有下划线，所以从右往左拆。
    """
    label, scene, episode, tag = stem.rsplit("_", 3)
    return label, scene, episode, tag


def save_episode(
    directory: pathlib.Path, stem: str, top: list, wrist: list, state, action, prompt: str, fps: int
) -> None:
    """存一局：顶视与腕部两路录像，以及逐步的关节状态、所发动作与指令。"""
    import imageio.v3 as iio

    iio.imwrite(str(directory / f"{stem}.mp4"), np.stack(top), fps=fps, codec="libx264")
    iio.imwrite(str(directory / f"{stem}_wrist.mp4"), np.stack(wrist), fps=fps, codec="libx264")
    np.savez(directory / f"{stem}.npz", state=np.stack(state), action=np.stack(action), prompt=prompt)


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


def read_frames(path: pathlib.Path) -> np.ndarray:
    """把一段录像整段解成 (T, H, W, 3) uint8。

    直接用 av：imageio 的 pyav 插件在新版 av 上读完即抛
    `VideoCodecContext has no attribute close`。
    """
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
