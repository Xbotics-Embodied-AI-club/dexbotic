"""把上游 DW0.5 policy（按 RobotWin 双臂写的）换成 SO101 的训练口径，供推演自检与推理服务共用。

上游 `dexbotic.policy.dw05_policy` 把状态排布、「不做 delta 的维」写成模块常量，状态还按 RobotWin 的习惯做
分位数归一化。SO101 的训练口径不同（见 `dexbotic.so101.dw05_exp`），这里在构造 policy 之前把它们换掉，
上游代码不改。换错不报错，只会把关节喂进错的槽位 —— 所以直接从训练配置导入，不抄一份。
"""

import os

import numpy as np

from dexbotic.so101.dw05_exp import SO101_NON_DELTA_MASK, SO101_STATE_ARRANGEMENT

#: 一轮推演的动作步数与出帧数：32 步动作出 9 帧（相邻两帧隔 4 步），与训练一致。
ACTION_HORIZON = 32
FPS_STRIDE = 4
NUM_VIDEO_FRAMES = 9
#: 推理时的去噪步数与随机种子。
NUM_INFERENCE_STEPS = 10
SEED = 1234
#: 训练时的三路视图：top + wrist + wrist（SO101 只有一路腕部相机，放进两个腕部位置）。
VIEWS = ("top", "wrist", "wrist")


def patch_policy_for_so101():
    """把上游 policy 的 RobotWin 常量换成 SO101 的训练口径。

    Returns:
        打过补丁的 `dexbotic.policy.dw05_policy` 模块。
    """
    from dexbotic.policy import dw05_policy

    # 训练时 70% 的样本 prompt 就是原样的任务指令；不套 RobotWin 的模板。
    os.environ["DEPLOY_USE_DEFAULT_PROMPT"] = "0"

    dw05_policy.ROBOTWIN_STATE_ARRANGEMENT = list(SO101_STATE_ARRANGEMENT)
    # NON_DELTA_MASK 给的是排布后的槽位；policy 在原始维上判，换回原始维号。
    dw05_policy.ROBOTWIN_NON_DELTA_DIMS = [
        SO101_STATE_ARRANGEMENT[slot] for slot in SO101_NON_DELTA_MASK
    ]
    dw05_policy.ROBOTWIN_VALID_ARRANGED_DIMS = [
        i for i, src in enumerate(SO101_STATE_ARRANGEMENT) if src >= 0
    ]

    # 本体状态（proprio）在训练时**没有归一化**：`ActionNormMultiDataset` 只归一化 norm_stats
    # 里有的键，而 compute_norm_stats 只产了 `action`；proprio 由 `AddProprioTrajectory`
    # 直接从排布后的原始状态（度 + 终止位）切窗口。上游 policy 却按 RobotWin 的习惯把 state
    # 做分位数归一化 —— 照搬就是喂给模型一个训练时从没见过的量纲，不报错。
    # ⇒ 推理这一侧也不归一化：排布 → 补终止位 0 → 取模型宽度，与训练逐步一致。
    import torch

    def normalize_state_like_training(self, raw_state):
        raw_state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        with_term = np.concatenate(
            [dw05_policy._arrange_raw_state(raw_state), np.zeros(1, dtype=np.float32)]
        )
        value = dw05_policy._select_model_dims(with_term, self.proprio_dim)
        return (
            torch.from_numpy(value)
            .unsqueeze(0)
            .to(device=self.model.device, dtype=self.model.torch_dtype)
        )

    dw05_policy.DW05RobotWinPolicy.normalize_state = normalize_state_like_training
    # policy 构造时硬要求 stats 里有 `state`；它在上面已被绕开、不会被读，给一个占位让构造通过。
    original_load = dw05_policy._load_norm_stats

    def load_with_state_placeholder(path):
        stats = original_load(path)
        stats.setdefault("state", stats["action"])
        return stats

    dw05_policy._load_norm_stats = load_with_state_placeholder
    return dw05_policy
