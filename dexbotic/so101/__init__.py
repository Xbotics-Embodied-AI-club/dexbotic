"""SO-101（单臂 5 关节 + 夹爪）上的 DM0.5 / DW0.5：实验配置、数据转换、评测与自检。

目录约定见 `layout`；与推理服务、录像文件打交道的小函数见 `client`。
单卡上的一套入口（`python -m dexbotic.so101.<名字>`）：download · serve · rollout · label_videos ·
imagine · prepare_data · train_lora。rollout 经 LeRobot 的机器人接口驱动仿真或真机。
"""
