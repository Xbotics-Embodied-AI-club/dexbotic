#!/usr/bin/env python3
"""预先算好 DW0.5 要的文本嵌入缓存。

## 为什么要自己写

上游 `TextEmbeddingCache` 在缓存缺失时的报错点名了一个 precompute 脚本，但上游仓里
没有这个脚本，所以照着读取端的约定自己产出。

不补的后果：`missing_text_embedding="zero"` 会把 context 与 context_mask 全部置零，
所有任务的语言条件完全相同，世界模型分不清「把方块放进箱子」和「把方块叠到罐子上」，
只能从画面猜 —— 训练全程只打一条 warning，不报错。

## 读取端的约定（照 `dexbotic/data/dataset/dw05/transform/text.py`，改了就对不上）

- 文件名：``{sha256(prompt)}.t5_len{context_len}.{enc_id}.pt``，
  默认 `context_len=128`、`enc_id="wan22ti2v5b"`。
- 内容：``{"context": [context_len, 4096] bfloat16, "mask": [context_len] bool}``。
- 位置：`cache_dir` 下，或其 `prompt/` `rewrite_prompt/` `subtask/` `subtask_rewrites/`
  `caption/` 任一子目录。本脚本写在 `cache_dir` 根下，查找第一站就是它。

## 要算哪些字符串

`PromptSelector.select` 在 `use_rewrite_prompt_and_subtask=False`（默认）且 subtask
非空时，对同一条样本只可能产出**两种**字符串：

1. ``subtask``（原样，不经 refine）—— 概率 1 - prompt_add_prob；
2. ``refine_text(f"{refine_text(caption)} {subtask}")`` —— 概率 prompt_add_prob。

`apply_prompt_format` 在 `meta_data.prompt_format` 为 None 时是恒等（RobotWin 的
meta 里正是 None）。所以从数据里取出所有 (caption, subtask) 对，各展开这两种，
就是完整的 prompt 空间 —— 不是采样，是穷举。

两种都要算：漏掉第二种会让 30% 的样本仍然吃零向量，而那同样只打 warning。

## 用法

    $PY script/so101/precompute_text_embeds.py \
        --jsonl-dir $SO101_DATASETS_DIR/so101-dexdata/jsonl \
        --out       $SO101_DATASETS_DIR/so101-dexdata/text_embeddings \
        --bundle    $DW05_BUNDLE

`--out` 要与训练配置里的 `text_embedding_cache_dir` 同值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from dexbotic.data.dataset.dw05.transform.text import refine_text


def collect_prompts(jsonl_dir: pathlib.Path) -> set[str]:
    """穷举数据里所有可能被送进文本编码器的字符串。

    Args:
        jsonl_dir: dexdata 的 jsonl 目录，每集一个文件、每帧一行。

    Returns:
        去重后的 prompt 集合。

    Raises:
        SystemExit: 目录里没有 jsonl，或一条 prompt 都没抽出来 —— 空缓存会让训练
            静默回落到零向量，比报错难发现得多。
    """
    files = sorted(jsonl_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"{jsonl_dir} 下没有 jsonl")

    pairs: set[tuple[str, str]] = set()
    for path in files:
        # 同一集里每帧的 caption/subtask 相同，读第一行即可 —— 全读要扫一百多万行
        # 才得到十几个不同的值。
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
        if not first.strip():
            continue
        record = json.loads(first)
        caption = (record.get("worldmodel") or {}).get("caption") or ""
        subtask = (record.get("robot") or {}).get("subtask") or ""
        pairs.add((str(caption), str(subtask)))

    # 产超集，不赌具体是哪一种写法。这条链上对同一段文字有几种相近的处理：有的地方
    # 直接用原串，有的用 `refine_text`（去首尾空白 + 补句号），有的只 `.strip()`。
    # 数据里有带前导空格的 `robot.subtask`，读取端查的是去空格那版。
    #
    # 每个嵌入约 1 MB，多产几种的代价可以忽略；漏产一种的代价是一批数据静默失去语言条件。
    def variants(text: str) -> set[str]:
        text = str(text)
        return {text, text.strip(), refine_text(text)}

    prompts: set[str] = set()
    for caption, subtask in pairs:
        if subtask:
            prompts |= variants(subtask)
            for cap in variants(caption):
                if cap:
                    for sub in variants(subtask):
                        prompts.add(refine_text(f"{refine_text(cap)} {sub}"))
        elif caption:
            prompts |= variants(caption)
    prompts.discard("")
    if not prompts:
        raise SystemExit(
            f"从 {len(files)} 个 jsonl 里一条 prompt 都没抽出来 —— "
            "检查 worldmodel.caption 与 robot.subtask 是否真的写进去了。"
        )
    return prompts


def cache_filename(prompt: str, context_len: int, enc_id: str) -> str:
    """按读取端的命名规则拼文件名。"""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{digest}.t5_len{context_len}.{enc_id}.pt"


def to_fixed_length(embedding: torch.Tensor, mask: torch.Tensor, context_len: int):
    """把变长的编码结果裁剪/补零到固定的 context_len。

    Args:
        embedding: `[T, 4096]`，编码器输出。
        mask: `[T]` bool，哪些位置是真 token。
        context_len: 读取端约定的固定长度。

    Returns:
        `(context [context_len, 4096] bfloat16, mask [context_len] bool)`。
    """
    valid = int(mask.sum().item())
    context = torch.zeros((context_len, embedding.shape[-1]), dtype=torch.bfloat16)
    out_mask = torch.zeros(context_len, dtype=torch.bool)
    keep = min(valid, context_len)
    if keep:
        context[:keep] = embedding[:keep].to(torch.bfloat16)
        out_mask[:keep] = True
    return context, out_mask, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument(
        "--bundle",
        required=True,
        type=pathlib.Path,
        help="含 Wan-AI/Wan2.2-TI2V-5B 的权重包，用它自带的 umT5 编码器",
    )
    ap.add_argument("--context-len", type=int, default=128)
    ap.add_argument("--enc-id", default="wan22ti2v5b")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = sorted(collect_prompts(args.jsonl_dir))
    print(f"[prompts] 穷举出 {len(prompts)} 条不同的 prompt")
    for p in prompts:
        print(f"          {p[:100]}")

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [
        p
        for p in prompts
        if not (args.out / cache_filename(p, args.context_len, args.enc_id)).is_file()
    ]
    if not todo:
        print("[skip] 全部已在缓存里")
        return 0
    print(f"[encode] 要算 {len(todo)} 条")

    # 用模型自己那份编码器，嵌入才和预训练时同分布。DIFFSYNTH_MODEL_BASE_PATH 是
    # 加载器找权重的根，和训练脚本里设的是同一个变量。
    import os

    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(args.bundle))
    from dexbotic.model.modules.wan22.helpers.loader import (
        _load_registered_model,
        _resolve_configs,
    )
    from dexbotic.model.modules.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    # 不走 `load_wan22_ti2v_5b_components`：那个公开入口**强制要求一份完整的
    # `dit_config`**（键按 `WanVideoDiT.__init__` 的签名逐个校验，少一个必填就报错），
    # 而这里连 DiT 都不需要 —— 它还会顺带加载 VAE。为了拿一个文本编码器去凑一份
    # 三十多项的 DiT 配置，本身就是个会随上游签名漂移的负担。
    #
    # 这里直接用它内部那两步：解析权重路径 → 按文件哈希匹配注册表并加载。
    # 两个函数带下划线，但它们正是公开入口内部调用的同一条路径；
    # 签名变了会当场报错，而不是静默拿到别的东西。
    _dit_cfg, text_config, _vae_cfg, tokenizer_config = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
    )
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
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean="whitespace")

    written = 0
    with torch.no_grad():
        for prompt in todo:
            ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to(args.device)
            mask = mask.to(args.device, dtype=torch.bool)
            emb = encoder(ids, mask)[0].float().cpu()
            context, out_mask, valid = to_fixed_length(emb, mask[0].cpu(), args.context_len)
            if valid > args.context_len:
                # 截断会丢掉句子末尾。真发生了要知道，别让它悄悄过去。
                print(
                    f"  ⚠️ {valid} token 超过 context_len={args.context_len}，已截断：{prompt[:60]}"
                )
            path = args.out / cache_filename(prompt, args.context_len, args.enc_id)
            torch.save({"context": context, "mask": out_mask}, path)
            written += 1
            print(f"  [{written}/{len(todo)}] {valid:3d} token  {path.name}")

    print(f"[done] 写了 {written} 个到 {args.out}")
    return 0


def _selftest() -> None:
    """自检：文件名规则与定长裁剪，两者错了训练都只会静默回落到零向量。"""
    # 文件名必须与读取端逐字一致 —— 它是 sha256(prompt) 而不是别的摘要。
    name = cache_filename("hello", 128, "wan22ti2v5b")
    expect = hashlib.sha256(b"hello").hexdigest() + ".t5_len128.wan22ti2v5b.pt"
    assert name == expect, name

    # 短于 context_len：前 valid 位为真、其余补零且 mask 为假。
    emb = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    mask = torch.tensor([True, True, False])
    ctx, m, valid = to_fixed_length(emb, mask, 5)
    assert valid == 2
    assert ctx.shape == (5, 4)
    assert m.tolist() == [True, True, False, False, False]
    assert torch.allclose(ctx[:2].float(), emb[:2])
    assert ctx[2:].abs().sum() == 0

    # 长于 context_len：截断，且 mask 全真。
    emb2 = torch.ones(10, 4)
    ctx2, m2, valid2 = to_fixed_length(emb2, torch.ones(10, dtype=torch.bool), 3)
    assert valid2 == 10
    assert ctx2.shape == (3, 4)
    assert m2.all()

    # 穷举逻辑：subtask 非空时应当给出两条，且第一条是原样的 subtask。
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "a.jsonl"
        p.write_text(
            json.dumps(
                {
                    "worldmodel": {"caption": "A video of a robot"},
                    "robot": {"subtask": "pick up the cube"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        got = collect_prompts(pathlib.Path(d))
    assert "pick up the cube" in got, got
    assert "A video of a robot. pick up the cube." in got, got

    # 带前导空格的 subtask：去空格那版必须在里面，读取端查的是它。
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "b.jsonl"
        p.write_text(
            json.dumps(
                {
                    "worldmodel": {"caption": "A video of a robot"},
                    "robot": {"subtask": " Stack the cube on the can"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        got2 = collect_prompts(pathlib.Path(d))
    assert "Stack the cube on the can" in got2, got2
    assert " Stack the cube on the can" in got2, got2
    assert "Stack the cube on the can." in got2, got2
    print("自检通过")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
