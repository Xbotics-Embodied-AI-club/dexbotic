"""在 SO101 数据上 LoRA 微调 DM0.5，并提供同一份配置的推理服务。

上游 `dexbotic/data/dataset_dm05/so101.py` 已注册 `so101_pick_cube`：`RobotType.SO101` 的
状态是「5 个关节 + 1 个夹爪」= 6 维，图像键 `images_1` / `images_2`，提示词
`["Head", "Left wrist"]`。这与本数据的口径一致（LeRobot v3、6 维动作与状态、top + wrist
两路 480×640、30 fps），所以不新建注册，只把数据目录指到 `convert_all.sh` 转出的
dexdata —— `DM05DataConfig._dataset_info` 会用 `jsonl_dir` / `image_dir` 覆盖注册里的默认值。

仿真三个场景与真机九个任务在同一份 dexdata 里，共享一份归一化统计；任务靠每帧的
`prompt` 区分。

动作口径沿用上游 SO101 配方的 `ActionMode.RELATIVE`：dexdata 里存绝对关节角，
`BuildAction` 按当前 state 现算增量；推理服务的输出变换再把状态加回去，返回绝对角。
所以评测侧直接当绝对角用（`rollout_so101.py --action-mode absolute`），再加一次会静默翻倍。

用法：
    CUDA_VISIBLE_DEVICES=0,1 bash script/so101/train_dm05_so101.sh     # 训练（仓根）
    python -m dexbotic.so101.dm05_exp --task inference --model-config.model-name-or-path <权重目录>
"""

from dataclasses import dataclass, field
from typing import Literal

import tyro

from dexbotic.constants_dm05.robot import ActionMode
from dexbotic.data.augmentations import NoAugmentationPipeline
from dexbotic.data.collator import TrainingCollator
from dexbotic.data.dataset_jsonl import JsonlDataset
from dexbotic.data.transforms import (
    ChatTokenization,
    LoadImages,
    Normalize,
    PadAction,
    Pipeline,
    PixelTransform,
)
from dexbotic.exp.dm05_exp import DM05DataConfig as _DM05DataConfig
from dexbotic.exp.dm05_exp import DM05Exp as _DM05Exp
from dexbotic.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from dexbotic.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig
from dexbotic.exp.dm05_exp import DM05OptimizerConfig as _DM05OptimizerConfig
from dexbotic.exp.dm05_exp import DM05TrainerConfig as _DM05TrainerConfig
from dexbotic.so101.layout import DATASETS_DIR, DEXDATA_DIR, RUNS_ROOT, WEIGHTS_DIR

DM05_RUN_DIR = f"{RUNS_ROOT}/dm05_so101"
#: SO101 单臂 5 关节 + 1 夹爪。与 `RobotType.SO101` 的状态描述一致。
SO101_ACTION_DIM = 6


@dataclass
class DM05DataConfig(_DM05DataConfig):
    """指向 `convert_all.sh` 转出的 dexdata，其余沿用上游 SO101 注册。"""

    dataset_name: str = field(default="so101_pick_cube")
    jsonl_dir: str | None = field(default=f"{DEXDATA_DIR}/jsonl")
    # dexdata 里视频的 `url` 是相对这个根的，与转换时的 `--image-root` 必须同值。
    image_dir: str | None = field(default=DATASETS_DIR)
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    add_state: bool = field(default=True)
    norm_stats_root: str = field(default=f"{DM05_RUN_DIR}/norm_stats")

    def build_dataset(
        self, processor, action_horizon: int, tokenizer_max_length: int = 1024
    ) -> tuple:
        """建数据集与 collator。

        变换链与上游 SO101 配方一致，只换数据目录：动作换算 → 读图 →
        像素变换（不做增强）→ 分位数归一化 → 对话 tokenize → 动作补齐到 32。

        Args:
            processor: DM0.5 的处理器，供 tokenize 用。
            action_horizon: 动作块长度。
            tokenizer_max_length: token 上限。

        Returns:
            `(dataset, collator)`。
        """
        dataset_info = self._dataset_info()
        pipeline = Pipeline(
            [
                self._action_transform(action_horizon),
                LoadImages(
                    image_keys=dataset_info["image_keys"], image_dir=dataset_info["image_dir"]
                ),
                PixelTransform(transform_pipeline=NoAugmentationPipeline()),
                Normalize(
                    norm_stats_path=str(self.norm_stats_path(action_horizon)),
                    norm_keys=["state", "action"],
                    use_quantiles=True,
                ),
                ChatTokenization(
                    processor=processor,
                    n_bins=self.n_bins,
                    max_length=tokenizer_max_length,
                    image_prompts=dataset_info["image_prompts"],
                    add_state=self.add_state,
                ),
                PadAction(32),
            ]
        )
        dataset = JsonlDataset(
            jsonl_dir=dataset_info["jsonl_dir"],
            transforms=pipeline,
            dataset_name=self.dataset_name,
            dataset_meta=self._dataset_meta(dataset_info),
        )
        collator = TrainingCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            max_length=tokenizer_max_length,
        )
        return dataset, collator


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    """底座是 dexbotic 布局的 `Dexmal/DM05`。

    **不是 `DM05-Lerobot`** —— 那份是 lerobot 布局（`dm05_processor/` 加 policy 前后处理器），
    dexbotic 的加载器读不了。两者同源，差别只在打包方式。
    """

    model_name_or_path: str | None = field(default=f"{WEIGHTS_DIR}/DM05")
    chunk_size: int = field(default=50)
    # flex_attention 在 sm_80 这代卡上不稳；上游 SO101 配方也退到 eager。
    llm_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="eager"
    )
    vision_attn_implementation: Literal["auto", "eager", "sdpa", "flash_attention_2"] = field(
        default="sdpa"
    )
    action_attn_implementation: Literal["auto", "eager", "sdpa", "flex_attention"] = field(
        default="sdpa"
    )
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=1e-4)
    warmup_steps: int = field(default=500)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    """训练预算：等效 batch 48、5000 步。

    `等效 batch = 卡数 × per_device_train_batch_size × gradient_accumulation_steps`。
    `train_dm05_so101.sh` 按卡数现算累积步数并覆盖这里的两项，所以任何卡数都复现同一个
    配方；这里的默认值对应两卡（2 × 8 × 3）。
    """

    output_dir: str = field(default=DM05_RUN_DIR)
    wandb_project: str = field(default="xbotics_dm05_so101")
    fsdp1: bool | None = field(default=False)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=3)
    save_steps: int = field(default=1000)
    num_train_steps: int = field(default=5000)
    save_total_limit: int = field(default=5)
    save_only_model: bool = field(default=True)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=SO101_ACTION_DIM)
    image_prompts: list[str] = field(default_factory=lambda: ["Head", "Left wrist"])


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=True)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


def main() -> None:
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")


if __name__ == "__main__":
    main()
