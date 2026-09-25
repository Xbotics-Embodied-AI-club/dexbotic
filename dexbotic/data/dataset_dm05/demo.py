"""Demo dataset registration."""

from dexbotic.constants_dm05.robot import RobotStateDesc
from dexbotic.data.dataset_dm05.register import register_dataset

ALOHA_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
)

register_dataset(
    {
        "demo": {
            "jsonl_dir": "./assets/demo/",
            "image_dir": "./assets/demo/",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "state_desc": ALOHA_STATE_DESC,
        },
    },
)
