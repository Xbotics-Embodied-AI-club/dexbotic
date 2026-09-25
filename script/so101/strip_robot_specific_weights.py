#!/usr/bin/env python3
"""从 DW0.5 检查点里剔掉宽度绑定机器人的那些权重，另存一份供换机型微调 resume。

## 剔哪些、为什么

凡是宽度等于「机器人动作/本体维度」的张量都要剔 —— 换机型时它们没有可迁移的语义，
本来就要重学。DW05-Robotwin 是双臂、宽 14，与 SO101 配置的宽度对不上。两个位置会撞：

1. `payload["mot"]` 里的动作头 —— `mixtures.action.action_encoder.*` 与
   `mixtures.action.head.*`。
2. `payload["proprio_encoder"]` —— 本体编码器，键名就是裸的 `weight`，形状 `[4096, 14]`。

第 2 个尤其要注意：上游 `load_checkpoint`（`dexbotic/model/dw05/dw05_core.py`）对 mot 用
`strict=False`、对 proprio_encoder 用 `strict=True`。只剔动作头的话它会在 proprio 上报
一句没有模块前缀的 `size mismatch for weight`。

整个 `proprio_encoder` 键直接删掉即可：上游对「没有这个键」有明确分支，
打一条 `keeping current proprio_encoder params` 的 warning 然后保留随机初始化，
正是换机型要的。

## 为什么不改上游、也不靠 strict=False

`strict=False` 容忍**缺键与多键**，但**不容忍形状不符** —— 形状不符照样抛。
所以必须让这些键**根本不出现在 payload 里**：不出现就是缺键，被容忍，
对应参数保持模型自己的随机初始化。

剔的动作放在**离线的一次转换**里，而不是改上游的加载函数：上游那个文件是模型代码，
改它会让每次升级都要重新合并；而且"哪些张量被丢掉了"应当是一件写在产物旁边、
能事后查证的事，不是一个藏在加载路径里的静默行为。

## 用法

    $PY script/so101/strip_robot_specific_weights.py \
        --in  $DW05_BUNDLE/model.pt \
        --out $SO101_WEIGHTS_DIR/DW05-Robotwin-noactionhead/model.pt

然后训练时 `trainer_config.resume=` 指向 `--out` 那份。剔了哪些、每个原来的形状是
什么，都打印出来并写进产物旁边的 `STRIPPED.json` —— 产物要能自己说明它被改过什么。
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch

#: `payload["mot"]` 里要剔的键（前缀匹配）。
#:
#: 两个都属于动作专家的输入投影与输出头，宽度随机器人的动作维度变。动作专家**内部**的
#: 层（注意力、FFN）宽度只跟 `action_hidden_dim` 有关、与机器人无关，所以不剔 ——
#: 那些才是真正值得迁移的部分。
MOT_PREFIXES = (
    "mixtures.action.action_encoder.",
    "mixtures.action.head.",
)

#: 整个删掉的顶层键。本体编码器的输入宽度就是本体维度，换机型必然对不上。
DROP_TOP_LEVEL = ("proprio_encoder",)


def strip(
    payload: dict,
    mot_prefixes: tuple[str, ...] = MOT_PREFIXES,
    drop_top_level: tuple[str, ...] = DROP_TOP_LEVEL,
) -> tuple[dict, dict[str, list[int] | str]]:
    """剔掉宽度绑定机器人的权重。

    Args:
        payload: `torch.load` 出来的检查点内容，必须含 `mot` 键。
        mot_prefixes: `payload["mot"]` 里按前缀剔的键。
        drop_top_level: 整个删掉的顶层键。

    Returns:
        `(新的 payload, {被剔的键: 形状或说明})`。输入的 payload 不被修改。

    Raises:
        KeyError: payload 里没有 `mot` —— 那不是 DW0.5 的 MoT 检查点。
        SystemExit: 一个都没剔中 —— 见下面的说明。
    """
    if "mot" not in payload:
        raise KeyError(
            f"检查点里没有 `mot` 键，只有 {sorted(payload)}。"
            "DW0.5 的 MoT 权重必须在 `mot` 下；`dit` 是只有视频专家的旧格式。"
        )
    stripped: dict[str, list[int] | str] = {}
    kept_mot = {}
    for key, tensor in payload["mot"].items():
        if key.startswith(mot_prefixes):
            stripped[f"mot.{key}"] = list(tensor.shape)
        else:
            kept_mot[key] = tensor

    new_payload = {k: v for k, v in payload.items() if k not in drop_top_level}
    new_payload["mot"] = kept_mot
    for key in drop_top_level:
        if key in payload:
            inner = payload[key]
            shapes = (
                {k: list(v.shape) for k, v in inner.items()}
                if isinstance(inner, dict)
                else "非 dict"
            )
            stripped[key] = json.dumps(shapes, ensure_ascii=False)

    if not stripped:
        # 空剔一遍会产出一份和输入一模一样的文件，那比报错更难发现 —— resume 指过去
        # 照样炸，而你以为已经处理过了。
        raise SystemExit(
            f"没有任何键匹配 {mot_prefixes} 或 {drop_top_level} —— "
            "要么这份检查点的命名不同，要么已经剔过了。"
        )
    return new_payload, stripped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=pathlib.Path)
    ap.add_argument("--out", dest="dst", required=True, type=pathlib.Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的输出。默认拒绝 —— resume 指向的权重被静默换掉最难查。",
    )
    args = ap.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"输入不存在：{args.src}")
    if args.dst.exists() and not args.force:
        raise SystemExit(f"输出已存在，确认后加 --force：{args.dst}")

    print(f"[load] {args.src}")
    payload = torch.load(args.src, map_location="cpu")
    new_payload, stripped = strip(payload)

    print(f"[strip] 剔掉 {len(stripped)} 项：")
    for key, shape in stripped.items():
        print(f"        {key}  {shape}")
    print(f"[keep]  保留 {len(new_payload['mot'])} 个 mot 张量，顶层键 {sorted(new_payload)}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_payload, args.dst)
    manifest = args.dst.parent / "STRIPPED.json"
    manifest.write_text(
        json.dumps(
            {
                "source": str(args.src),
                "stripped": stripped,
                "why": (
                    "宽度绑定机器人的权重换机型时没有可迁移语义，本来就要重学："
                    "RobotWin 是双臂 14 维，与 SO101 配置的宽度对不上。剔掉后动作头走 "
                    "strict=False 的缺键分支、proprio_encoder 走上游的"
                    "「没有这个键就保留当前参数」分支，都是随机初始化重学。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save]  {args.dst}")
    print(f"[save]  {manifest}")
    return 0


def _selftest() -> None:
    """最小自检：剔该剔的、留该留的、空剔报错、缺 mot 报错、输入不被就地改。"""
    fake = {
        "mot": {
            "mixtures.action.action_encoder.weight": torch.zeros(4, 14),
            "mixtures.action.head.weight": torch.zeros(14, 4),
            "mixtures.action.head.bias": torch.zeros(14),
            "mixtures.action.blocks.0.ffn.weight": torch.zeros(4, 4),
            "mixtures.video.blocks.0.ffn.weight": torch.zeros(4, 4),
        },
        "proprio_encoder": {"weight": torch.zeros(4096, 14)},
        "optimizer": {"fake": 1},
    }
    new, stripped = strip(fake)
    assert set(stripped) == {
        "mot.mixtures.action.action_encoder.weight",
        "mot.mixtures.action.head.weight",
        "mot.mixtures.action.head.bias",
        "proprio_encoder",
    }, stripped
    assert stripped["mot.mixtures.action.head.weight"] == [14, 4]
    assert "proprio_encoder" not in new, "本体编码器必须整个删掉，不是留一个空 dict"
    # 动作专家内部的层与视频专家都必须留下 —— 它们的宽度与机器人无关。
    assert set(new["mot"]) == {
        "mixtures.action.blocks.0.ffn.weight",
        "mixtures.video.blocks.0.ffn.weight",
    }, new["mot"].keys()
    assert new["optimizer"] == {"fake": 1}, "顶层的其他键要原样保留"
    assert len(fake["mot"]) == 5, "输入不该被就地修改"
    assert "proprio_encoder" in fake, "输入不该被就地修改"
    try:
        strip({"mot": {"a": torch.zeros(1)}})
    except SystemExit:
        pass
    else:
        raise AssertionError("一个都没剔中时必须报错，不能产出一份与输入相同的文件")
    try:
        strip({"dit": {}})
    except KeyError:
        pass
    else:
        raise AssertionError("没有 mot 键必须报 KeyError")
    print("自检通过")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
