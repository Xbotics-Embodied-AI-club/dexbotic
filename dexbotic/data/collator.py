from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from dexbotic.constants import IGNORE_INDEX


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        mapping_keys = {
            'image': 'images',
            'actions': 'actions',
            'action': 'actions',
            'state': 'states',
            'reward': 'reward',
            'image_masks': 'image_masks',
            'has_action': 'has_action',
            'has_text': 'has_text',
        }
        for key in mapping_keys:
            if key in instances[0]:
                values = [instance[key] for instance in instances]
                if all(x is not None and x.shape == values[0].shape for x in values):
                    batch[mapping_keys[key]] = torch.stack(values)
                else:
                    batch[mapping_keys[key]] = values

        return batch

# ==========================================================================
# 以下内容来自 dexmal/opendm（DM0.5），内部 import 已按并入后的位置重写。
# 两侧顶层符号零重名：dexbotic 侧 DataCollatorForSupervisedDataset，
# DM0.5 侧 TrainingCollator / NormStatsCollator，因此取并集即可。
# ==========================================================================

from collections.abc import Sequence
from enum import Enum

import numpy as np
import torch
from loguru import logger


class TrainingCollator:
    def __init__(
        self,
        pad_token_id: int,
        max_length: int | None = 1024,
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        batch = {}

        input_ids_list = [inst["input_ids"] for inst in instances]
        attention_mask_list = [inst["attention_mask"] for inst in instances]
        token_type_ids_list = [inst["token_type_ids"] for inst in instances]

        padded_input_ids = []
        padded_attention_mask = []
        padded_token_type_ids = []

        max_len = self.max_length
        for input_ids, attention_mask, token_type_ids in zip(
            input_ids_list, attention_mask_list, token_type_ids_list, strict=True
        ):
            seq_len = input_ids.shape[1]
            if seq_len > max_len:
                logger.warning(
                    "Input sequence length {} exceeds max_length {}; truncating.",
                    seq_len,
                    max_len,
                )
                input_ids = input_ids[:, :max_len]
                attention_mask = attention_mask[:, :max_len]
                token_type_ids = token_type_ids[:, :max_len]
                seq_len = max_len

            pad_len = max_len - seq_len
            if pad_len > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (
                                1,
                                pad_len,
                            ),
                            self.pad_token_id,
                            dtype=input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros((1, pad_len), dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )
                token_type_ids = torch.cat(
                    [
                        token_type_ids,
                        torch.zeros((1, pad_len), dtype=token_type_ids.dtype),
                    ],
                    dim=1,
                )
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_token_type_ids.append(token_type_ids)

        batch["input_ids"] = torch.cat(padded_input_ids, dim=0)
        batch["attention_mask"] = torch.cat(padded_attention_mask, dim=0)
        batch["token_type_ids"] = torch.cat(padded_token_type_ids, dim=0)

        pixel_values_list = [inst["pixel_values"] for inst in instances]
        action_list = [inst["action"] for inst in instances]
        action_mask_list = [inst["action_mask"] for inst in instances]

        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["action"] = torch.cat(action_list, dim=0)
        batch["action_mask"] = torch.cat(action_mask_list, dim=0)

        return batch


class NormStatsCollator:
    def __call__(self, instances: Sequence[dict]) -> dict:
        grouped_instances: dict[str | None, list[dict]] = {}
        for instance in instances:
            robot_type = instance.get("meta_data", {}).get("robot_type")
            if isinstance(robot_type, Enum):
                robot_type = str(robot_type.value)
            elif robot_type is not None:
                robot_type = str(robot_type)
            grouped_instances.setdefault(robot_type, []).append(instance)

        if None in grouped_instances and len(grouped_instances) > 1:
            raise ValueError(
                "Cannot compute norm stats from a batch containing both typed "
                "and untyped robot samples"
            )

        robot_batches = {}
        for robot_type, robot_instances in grouped_instances.items():
            robot_batch = {}
            for key in ("state", "action"):
                presence = [key in instance for instance in robot_instances]
                if any(presence) and not all(presence):
                    raise ValueError(
                        f"Inconsistent {key!r} presence for robot_type {robot_type!r}"
                    )
                if not any(presence):
                    continue
                values = [instance[key] for instance in robot_instances]
                robot_batch[key] = (
                    np.stack(values, axis=0)
                    if key == "state"
                    else np.concatenate(values, axis=0)
                )
            if "action" not in robot_batch:
                raise ValueError(
                    f"Cannot compute norm stats without action for robot_type "
                    f"{robot_type!r}"
                )
            robot_batches[robot_type] = robot_batch
        return {"robot_batches": robot_batches}
