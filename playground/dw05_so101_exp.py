"""在 SO101 数据上微调 DW0.5 世界模型。

所有配置都走 `DW05DataConfig` 已有的扩展点，**上游模型与训练代码一行未改**：给出
`annotations` 之后 `dw05_exp.py` 的 `_build_recipe_entries` 会拼一条内联 recipe，
并把 `dataset_meta_overrides` 合并进 `ROBOTWIN_META` 的副本。

## 与 DM0.5 共用同一份 dexdata

DW0.5 的索引是 `<annotations>/**/*.jsonl` 递归扫，DM0.5 的 `JsonlDataset` 是扫一个目录，
两者都吃 `convert_all.sh` 产出的那个扁平 `jsonl/` 目录，一份数据两个模型共用。

## 状态排布：6 维填进 RobotWin 的 16 槽

RobotWin 是双臂、每臂八槽（六个位姿维 + 一个占位 + 一个夹爪）。SO101 是单臂、
五个关节 + 一个夹爪。所以只填第一块的前五槽，夹爪落在夹爪槽 7 上，第二块整体留空 ——
保留预训练权重见过的维度语义，比把六维平铺到前六槽更贴近原分布。

## 三路视图：wrist 用两次

`concat_views` 在相机少于三路时**只用第一路**，其余被忽略；三路时才拼成预训练见过的
「上大下二」画面。SO101 只有 top 与 wrist 两路，所以把 wrist 复用一次凑成三路，
让画面布局与预训练一致，同时保留腕部细节（抓取那一刻最关键的信息）。

## 底座

从 `DW05-Robotwin` 起，而不是 `DW05-Base`：前者已经是一个可用的动作条件操作世界模型
（实测真动作 30.5 dB、比复制首帧基线高 11.5 dB），迁到同类抓放任务上
起点比通用底座高。它的动作头与本体编码器绑定双臂维度，要先用
`script/so101/strip_robot_specific_weights.py` 剔掉，再以 `trainer_config.resume=` 起步。

用法（仓根）：
    CUDA_VISIBLE_DEVICES=0 bash script/so101/train_dw05_so101.sh compute_norm_stats
    CUDA_VISIBLE_DEVICES=0,1,2 bash script/so101/train_dw05_so101.sh train \\
        trainer_config.resume=<剔过的 model.pt>
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from dexbotic.exp.base_dw_exp import apply_dotlist_overrides
from dexbotic.exp.dw05_exp import (
    DW05DataConfig as _DW05DataConfig,
)
from dexbotic.exp.dw05_exp import (
    DW05Exp as _DW05Exp,
)
from dexbotic.exp.dw05_exp import (
    DW05InferenceConfig as _DW05InferenceConfig,
)
from dexbotic.exp.dw05_exp import (
    DW05NormStatsConfig as _DW05NormStatsConfig,
)
from dexbotic.exp.dw05_exp import (
    DW05TrainerConfig as _DW05TrainerConfig,
)
from dexbotic.exp.dw05_trainer import DW05Trainer as _DW05Trainer
from dexbotic.model.dw05 import DW05ModelConfig as _DW05ModelConfig
from script.so101.layout import DATASETS_DIR, RUNS_ROOT

DEXDATA_ROOT = f"{DATASETS_DIR}/so101-dexdata"
DW05_RUN_DIR = f"{RUNS_ROOT}/dw05_so101"
#: `--task compute_norm_stats` 的产物；训练与 `dw05_sim_check.py` 都读它。
DW05_NORM_STATS = f"{DW05_RUN_DIR}/norm_stats/norm_stats.json"

#: SO101 的六维状态填进 RobotWin 的十六槽，见模块 docstring。
SO101_STATE_ARRANGEMENT = [0, 1, 2, 3, 4, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1]
#: 夹爪那一维不做 delta 编码，与 RobotWin 对夹爪槽的处理一致。
SO101_NON_DELTA_MASK = [7]

# 关于宽度：这条链上有两个不同的宽度，别把它们弄混。
#
# - **17 = 归一化的宽度**。`ArrangeState` 输出 `len(state_arrangement)` = 16 宽
#   （槽位数，不是有效维数；无效槽填 0 并在 `action_dim_mask` 里标掉），
#   `AddTerminationState` 再追加一位终止标志。`compute_norm_stats` 产出的
#   norm_stats.json 每一项长度正是 17 —— 归一化发生在补零**之前**。
# - **32 = 模型看到的宽度**。变换链最后是 `PadState(ndim=32)` 与 `PadAction(ndim=32)`
#   （`dw_dataset.py` 里写死的），补零到 32 才进模型。
#
# 所以 `DW05ModelConfig` 默认的 `action_dim=32` / `proprio_dim=32` **是对的**，这里不改。
# 按 17 配会在第一个 batch 上报 ``sample['proprio'] last dim must be 17, got 32``。
#
# 而 DW05-Robotwin 检查点里那些宽 14 的张量是它训练时更早的一套排布，与现在这两个宽度
# 都无关 ⇒ 从它起步必须先剔掉与机器人绑定的权重，见
# `script/so101/strip_robot_specific_weights.py`。


def parse_args():
    """解析 `--task` 与 dotlist 覆盖参数。

    Returns:
        `argparse.Namespace`，未识别的参数收进 `overrides` 交给 dotlist 覆盖。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="smoke",
        choices=["train", "inference", "compute_norm_stats", "smoke"],
    )
    args, unknown = parser.parse_known_args()
    # 上游的 `apply_dotlist_overrides` 把不含 `=` 的 token 静默丢弃。写成
    # `--data_config.annotations /path` 这种 argparse 风格时覆盖不生效、也不报错，
    # 训练照旧用默认路径跑起来。所以在进入上游之前先拦：格式只有 `a.b.c=value` 一种。
    malformed = [token for token in unknown if "=" not in token]
    if malformed:
        raise SystemExit(
            f"覆盖参数格式不对：{malformed}。只接受 `字段路径=值`（下划线、等号、不带 --），"
            "例如 `data_config.annotations=/path/to/jsonl`。"
            "上游会把不含等号的 token 静默丢弃，所以这里必须拦下来。"
        )
    args.overrides = unknown
    return args


@dataclass
class So101DataConfig(_DW05DataConfig):
    """指向共用的那份 dexdata，并把状态排布改成单臂六维。"""

    # recipe 只是个名字：给出 annotations 后真正的来源走内联条目（名为 dw05_local），
    # 但 base_dw_exp.build_data 仍要求 recipe 非空，所以这里对齐成同一个名字。
    recipe: str = field(default="dw05_local")
    annotations: str = field(default=f"{DEXDATA_ROOT}/jsonl")
    # 记录里的视频 url 是相对这个根的，与转换时的 `--image-root` 必须同值。
    data_path_prefix: str = field(default=DATASETS_DIR)
    index_path_prefix: str = field(default="")
    images_keys: list[str] = field(default_factory=lambda: ["images_1", "images_2", "images_2"])
    # 文本嵌入由 `script/so101/precompute_text_embeds.py` 预先算好放在这个目录里。
    #
    # `missing="zero"` 是有代价的兜底：查不到就返回全零 context 与全零 mask，
    # 训练照跑照收敛，只是那批样本失去语言条件，全程只有一条 warning。
    # 不改成 `"error"`：缺失可以在线补（补出文件放进训练读的缓存目录，下次查找即命中，
    # 不必重启），而 `"error"` 会让一条没见过的 prompt 在任意时刻打断整轮训练。
    # 所以训练日志里 `Missing text embedding` 的条数应当是 0，非零就补算缓存。
    text_embedding_cache_dir: str = field(default=f"{DEXDATA_ROOT}/text_embeddings")
    missing_text_embedding: str = field(default="zero")
    # 由 `--task compute_norm_stats` 先生成；维度必须等于 len(state_arrangement)+1 = 17。
    norm_stats_path: str = field(default=DW05_NORM_STATS)
    dataset_meta_overrides: dict | None = field(
        default_factory=lambda: {
            "state_arrangement": SO101_STATE_ARRANGEMENT,
            "non_delta_mask": SO101_NON_DELTA_MASK,
        }
    )


@dataclass
class So101TrainerConfig(_DW05TrainerConfig):
    """训练预算与存档节流：20000 步。

    上游有三个默认值在这份数据上不合适：

    - `wandb_enabled` 上游默认 False，不改就什么都不上报。
    - `num_epochs=5` 在这份约 128 万帧的数据上远超预算，改按 `max_steps` 收口。
    - `save_every=2500` 时每次存档约 183 GB（优化器状态 160 GB + 权重 23 GB），
      间隔拉大到一万步。
    """

    output_dir: str = field(default=DW05_RUN_DIR)
    wandb_project: str = field(default="xbotics_dw05_so101")
    wandb_name: str = field(default="dw05-so101-mixed")
    wandb_enabled: bool = field(default=True)
    # 在线：训练中就能看曲线。启动脚本里的 WANDB_MODE 会覆盖它。
    wandb_mode: str = field(default="online")
    batch_size: int = field(default=1)
    num_workers: int = field(default=4)
    max_steps: int | None = field(default=20000)
    save_every: int = field(default=10000)
    # 单次 eval 约 10-15 秒，200 步训练约 150 秒，开销在 10% 上下，
    # 换来全程 100 段对比视频（每段约 0.1 MB）。
    eval_every: int = field(default=200)


@dataclass
class So101NormStatsConfig(_DW05NormStatsConfig):
    norm_save_path: str = field(default=f"{DW05_RUN_DIR}/norm_stats")
    batch_size: int = field(default=8)
    num_workers: int = field(default=4)
    max_batches: int | None = field(default=400)


@dataclass
class So101InferenceConfig(_DW05InferenceConfig):
    output_mp4: str = field(default=f"{DW05_RUN_DIR}/rollout.mp4")


class So101Trainer(_DW05Trainer):
    """在上游的 eval 指标之外，把那段对比视频也送进 wandb。

    上游 `build_eval_wandb_payload` 只挑 `isinstance(value, (int, float))` 的项，
    于是 `metrics["video_path"]` 被过滤掉 —— 视频只落在
    `runs/<run>/eval/step_XXXXXX_rank_XXX.mp4`，看曲线的人看不到画面。而这个模型
    要判断的恰恰是"预测的未来像不像"，那是**指标之外的东西**：PSNR 掉了能看出退步，
    但看不出它是把机械臂画歪了还是把任务做错了。

    视频本身是三行拼起来的（`dw05_trainer.evaluate` 里 `torch.cat(..., dim=2)`）：
    **上=模型预测 · 中=VAE 重建的真值 · 下=真值原片**。中间那行是天花板 ——
    预测再好也不会超过 VAE 重建，所以它是判断"差距来自世界模型还是来自编解码"的参照。

    只有主进程有 wandb run（`_wandb_log` 在非主进程是空操作），所以只传 rank 0 那一份。
    """

    def build_eval_wandb_payload(self, metrics: dict) -> dict:
        payload = super().build_eval_wandb_payload(metrics)
        video_path = metrics.get("video_path")
        # 上游在个别分支里不产视频；缺了就只报指标。
        if not video_path or not os.path.isfile(video_path):
            return payload
        try:
            import wandb

            payload["eval/rollout"] = wandb.Video(video_path, format="mp4")
        except Exception as exc:  # noqa: BLE001 —— 见下
            # 传视频失败**不许影响训练**：它是观测手段，不是训练的一部分。
            # 但也不能静默吞掉，否则"wandb 里一直没有视频"会查不出原因。
            logger.warning("eval 视频没能进 wandb（训练不受影响）：{}", exc)
        return payload


@dataclass
class So101Exp(_DW05Exp):
    trainer_cls = So101Trainer
    # 模型配置原样沿用上游：动作与本体宽度都是 32，与变换链末尾的 PadState/PadAction
    # 对得上，见上面「关于宽度」那段。
    model_config: _DW05ModelConfig = field(default_factory=_DW05ModelConfig)
    trainer_config: So101TrainerConfig = field(default_factory=So101TrainerConfig)
    data_config: So101DataConfig = field(default_factory=So101DataConfig)
    inference_config: So101InferenceConfig = field(default_factory=So101InferenceConfig)
    norm_stats_config: So101NormStatsConfig = field(default_factory=So101NormStatsConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = So101Exp()
    apply_dotlist_overrides(exp, args.overrides)
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
    elif args.task == "smoke":
        exp.smoke()
