"""SO-101 这一侧与 DM0.5 推理服务、与录像文件打交道的几个小函数。

仿真评测、开环自检、出片脚本与工作坊的控制循环都用这一份，不各自再写。
"""

from __future__ import annotations

import base64
import io
import json
import pathlib

import numpy as np

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


def read_frames(path: pathlib.Path) -> np.ndarray:
    """把 `save_mp4` 存下的视频整段解成 (T, H, W, 3) uint8。

    直接用 av：imageio 的 pyav 插件在新版 av 上读完即抛
    `VideoCodecContext has no attribute close`。
    """
    import av

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
