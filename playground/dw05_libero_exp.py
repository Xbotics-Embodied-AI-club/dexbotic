"""在 LIBERO 数据上跑 DW05 世界模型的实验入口。

这个文件放在本仓库自己的 playground 下，不在合并树里——合并树由 merge_dw05.py
从上游重新生成，放进去会被覆盖。

所有配置都走 DW05DataConfig 已有的扩展点，上游代码一行未改：
给出 annotations 路径后，dw05_exp.py 的 _build_recipe_entries 会自动拼一条内联
recipe，并把 dataset_meta_overrides 合并进 ROBOTWIN_META 的副本。
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexbotic.exp.base_dw_exp import apply_dotlist_overrides
from dexbotic.exp.dw05_exp import (
    DW05DataConfig as _DW05DataConfig,
    DW05Exp as _DW05Exp,
    DW05InferenceConfig as _DW05InferenceConfig,
    DW05NormStatsConfig as _DW05NormStatsConfig,
    DW05TrainerConfig as _DW05TrainerConfig,
)
from dexbotic.model.dw05 import DW05ModelConfig as _DW05ModelConfig

# 数据根由环境给出，两台机器上取值不同（一台挂 NAS、一台是本地盘）。
LIBERO_DW05_ROOT = f"{os.environ['DATASETS_ROOT']}/datasets/private/dexmal/libero-dw05"

# LIBERO 是单臂，八维状态是"末端位姿六维 + 两根手指"。
# RobotWin 的排布是每臂八槽：六个位姿维 + 一个占位 + 一个夹爪，两臂共十六槽。
# 这里只填第一块、第二块整体留空，保留预训练权重见过的维度语义；
# 两根手指取第一根（另一根是镜像），落在夹爪槽 7 上。
LIBERO_STATE_ARRANGEMENT = [0, 1, 2, 3, 4, 5, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="smoke",
        choices=["train", "inference", "compute_norm_stats", "smoke"],
    )
    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    return args


@dataclass
class LiberoDataConfig(_DW05DataConfig):
    # recipe 只是个名字，真正的数据来源走内联条目：给出 annotations 后，
    # dw05_exp.py 的 _build_recipe_entries 会拼出一条名为 dw05_local 的条目。
    # 但 base_dw_exp.build_data 仍要求 recipe 非空，所以这里对齐成同一个名字。
    recipe: str = field(default="dw05_local")
    annotations: str = field(default=f"{LIBERO_DW05_ROOT}/annotations")
    data_path_prefix: str = field(default=LIBERO_DW05_ROOT)
    index_path_prefix: str = field(default=f"{LIBERO_DW05_ROOT}/annotations")
    # 相机只喂第三人称：concat_views 在相机少于三路时只用第一路，凑数没有意义。
    images_keys: list[str] = field(default_factory=lambda: ["images_1"])
    # 没有预先缓存文本嵌入，缺失时补零。目录本身仍是必填项（校验器要求非空），
    # 建一个空目录即可，实际取不到就走 zero 兜底。
    text_embedding_cache_dir: str = field(default=f"{LIBERO_DW05_ROOT}/text_embeddings")
    missing_text_embedding: str = field(default="zero")
    # 由 --task compute_norm_stats 先生成；维度必须等于 len(state_arrangement)+1 = 17。
    norm_stats_path: str = field(default="./runs/dw05_libero/norm_stats/norm_stats.json")
    dataset_meta_overrides: dict | None = field(
        default_factory=lambda: {
            "state_arrangement": LIBERO_STATE_ARRANGEMENT,
            # 夹爪那一维不做 delta 编码，与 RobotWin 对夹爪槽的处理一致。
            "non_delta_mask": [7],
        }
    )


@dataclass
class LiberoTrainerConfig(_DW05TrainerConfig):
    output_dir: str = field(default="./runs/dw05_libero")
    wandb_project: str = field(default="dexbotic_dw05_libero")
    wandb_name: str = field(default="dw05-libero")
    batch_size: int = field(default=1)
    num_workers: int = field(default=2)


@dataclass
class LiberoNormStatsConfig(_DW05NormStatsConfig):
    norm_save_path: str = field(default="./runs/dw05_libero/norm_stats")
    batch_size: int = field(default=8)
    num_workers: int = field(default=2)
    max_batches: int | None = field(default=200)


@dataclass
class LiberoInferenceConfig(_DW05InferenceConfig):
    output_mp4: str = field(default="./runs/dw05_libero/rollout.mp4")


@dataclass
class LiberoExp(_DW05Exp):
    model_config: _DW05ModelConfig = field(default_factory=_DW05ModelConfig)
    trainer_config: LiberoTrainerConfig = field(default_factory=LiberoTrainerConfig)
    data_config: LiberoDataConfig = field(default_factory=LiberoDataConfig)
    inference_config: LiberoInferenceConfig = field(default_factory=LiberoInferenceConfig)
    norm_stats_config: LiberoNormStatsConfig = field(default_factory=LiberoNormStatsConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = LiberoExp()
    apply_dotlist_overrides(exp, args.overrides)
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
    elif args.task == "smoke":
        exp.smoke()
