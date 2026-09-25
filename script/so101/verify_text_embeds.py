#!/usr/bin/env python3
"""证明缓存里的文本嵌入，就是训练/推理时那个编码器会算出来的东西。

## 为什么需要这一步

缓存是**离线**算的，训练时只按文件名去查、查到就用，**不校验内容**。所以一份用错编码器
（或错 dtype、错长度约定）算出来的缓存，训练照样吃下去、照样收敛，只是文本条件那一路
接的是另一个向量空间的东西 —— 全程不报错。「文件存在」不等于「内容对」。

这个脚本做的就是那个缺失的校验：现场用同一条加载路径把编码器取回来，重新编码一遍，
和缓存文件逐元素比。

## 一致性靠什么保证

`_load_registered_model` 认权重的方式是**文件内容哈希**：算出 `hash_model_file(path)`，
在 `WAN22_MODEL_REGISTRY` 里找同时匹配 hash 与 model_name 的条目，找不到就直接
`ValueError: Cannot detect model type`。所以"指到了另一个 T5"这件事**做不到静默发生**。

本脚本与训练走的是同一个 `_resolve_configs` → `_load_registered_model`，且同一个
`DIFFSYNTH_MODEL_BASE_PATH`，所以拿到的必然是同一份权重与同一个 tokenizer。

用法（仓根，训练侧环境）：
    $PY script/so101/verify_text_embeds.py \
        --cache  $SO101_DATASETS_DIR/so101-dexdata/text_embeddings \
        --bundle $DW05_BUNDLE \
        --prompt "Pick up a cube and place in the bin" --prompt "<另一条 prompt>"

`--prompt` 必须显式给：文件名是哈希，反推不出原文。至少给两条，第二条当负对照。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from script.so101.precompute_text_embeds import cache_filename, to_fixed_length


def cos_on_valid(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """有效 token 上的平均逐 token 余弦相似度；没有有效 token 时返回 NaN。"""
    if int(mask.sum()) == 0:
        return float("nan")
    return float(
        torch.nn.functional.cosine_similarity(a.float()[mask], b.float()[mask], dim=-1).mean()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=pathlib.Path)
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    ap.add_argument("--prompt", required=True, action="append", help="要验的 prompt 原文，可给多次")
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--enc-id", default="wan22ti2v5b")
    ap.add_argument(
        "--device", default="cpu", help="默认 CPU —— 卡上通常有训练在跑，不去挤它的显存"
    )
    args = ap.parse_args()

    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(args.bundle))
    from dexbotic.model.modules.wan22.helpers.loader import (
        _load_registered_model,
        _resolve_configs,
    )
    from dexbotic.model.modules.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    _dit, text_config, _vae, tok_config = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
    )
    print(f"[encoder] 权重 {text_config.path}")
    print(f"[encoder] tokenizer {tok_config.path}")
    encoder = (
        _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch.bfloat16,
            device=args.device,
        )
        .to(args.device)
        .eval()
    )
    tokenizer = HuggingfaceTokenizer(name=tok_config.path, seq_len=512, clean="whitespace")

    all_ok = True
    with torch.no_grad():
        for prompt in args.prompt:
            path = args.cache / cache_filename(prompt, args.context_len, args.enc_id)
            print(f"\n=== {prompt!r}")
            print(f"    文件 {path.name}")
            if not path.is_file():
                print("    ✗ 缓存里没有这个文件 —— 训练会回落到零向量")
                all_ok = False
                continue
            payload = torch.load(path, map_location="cpu")
            cached, cached_mask = payload["context"], payload["mask"].bool()

            ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
            emb = encoder(ids.to(args.device), mask.to(args.device, dtype=torch.bool))[0]
            fresh, fresh_mask, valid = to_fixed_length(
                emb.float().cpu(), mask[0].cpu(), args.context_len
            )

            same_shape = cached.shape == fresh.shape
            same_dtype = cached.dtype == fresh.dtype
            same_mask = torch.equal(cached_mask, fresh_mask)

            # 判据用余弦相似度，不用绝对差：缓存在 GPU 上算、这里在 CPU 上重算，归约顺序
            # 不同，而两边都以 bfloat16 存 —— bf16 相对精度 2^-8≈0.0039，幅值 ~10 的元素
            # 量化台阶就有 0.04 上下，绝对差量出来的是存储精度而不是「是不是同一个东西」。
            #
            # 同时做负对照：拿另一条 prompt 的缓存来比。判据成立要求同一条 prompt 的相似度
            # 接近 1，并且不同 prompt 的相似度明显低 —— 连不同 prompt 都算一致，
            # 这个检查就什么都没证明。
            cos_same = cos_on_valid(cached, fresh, cached_mask)
            ok = same_shape and same_dtype and same_mask and cos_same > 0.999

            print(f"    形状 {tuple(cached.shape)} == {tuple(fresh.shape)} : {same_shape}")
            print(f"    dtype {cached.dtype} == {fresh.dtype} : {same_dtype}")
            print(f"    有效 token {int(cached_mask.sum())} / {valid}，mask 一致 : {same_mask}")
            print(f"    逐 token 余弦相似度（同一 prompt）: {cos_same:.6f}")
            print(
                f"    最大逐元素差 {float((cached.float() - fresh.float()).abs().max()):.4f}"
                f"（幅值最大元素 {float(cached.float().abs().max()):.3f}，"
                f"bf16 在该幅值的量化台阶约 {2**-8 * float(cached.float().abs().max()):.4f}）"
            )

            # 负对照：随便另一条 prompt。取不到就说明只给了一条，明说而不是静默跳过。
            others = [q for q in args.prompt if q != prompt]
            if others:
                other_path = args.cache / cache_filename(others[0], args.context_len, args.enc_id)
                if other_path.is_file():
                    other = torch.load(other_path, map_location="cpu")["context"]
                    cos_other = cos_on_valid(cached, other, cached_mask)
                    print(f"    负对照 与 {others[0]!r} 的相似度: {cos_other:.6f}")
                    if not (cos_other < 0.99):
                        print("    ⚠️ 负对照没能拉开差距 —— 这个判据不构成证据")
                        ok = False
            else:
                print("    ⚠️ 只给了一条 prompt，没有负对照，本次判定分辨力不足")
                ok = False

            print(f"    {'✓ 一致' if ok else '✗ 不一致'}")
            all_ok = all_ok and ok

    print(
        f"\n{'[PASS] 缓存内容与编码器现算一致' if all_ok else '[FAIL] 有不一致，训练的文本条件不可信'}"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
