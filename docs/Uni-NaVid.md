# Uni-NaVid in Dexbotic

## Overview

[Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) is an open-source embodied
vision-language navigation project. Dexbotic now supports Uni-NaVid-style SFT
and inference testing while keeping the training, inference, and evaluation
entrypoints inside the Dexbotic repository.

The integration includes three main pieces:

- A Dexbotic dataset loader for navigation episodes stored as JSONL + video.
- A supervised fine-tuning entrypoint.
- An inference server that can be evaluated with
  [dexbotic-benchmark](https://github.com/dexmal/dexbotic-benchmark).

### Resources

- **Example training data**:
  [Dexmal/uninavid-objnav-demo](https://huggingface.co/datasets/Dexmal/uninavid-objnav-demo)
  — 200 object-navigation training episodes in Dexbotic Uni-NaVid format,
  illustrating the JSONL + video layout below.
- **Pretrained weights**:
  [Dexmal/Dexbotic-UniNaVid](https://huggingface.co/Dexmal/Dexbotic-UniNaVid)
  — weights converted to the Dexbotic Uni-NaVid format for training
  initialization and inference.

## Data Conversion

Download the example dataset from **Resources** above and place the files under
the directory layout below (or update `annotations` and `data_path_prefix`
accordingly).

The default example dataset is registered as `uninavid_objnav` in
[`dexbotic/data/data_source/uninavid_official.py`](../dexbotic/data/data_source/uninavid_official.py):

```python
UNINAVID_DATASET = {
    "objnav": {
        "data_path_prefix": "./data/objnav/video",
        "annotations": "./data/objnav",
        "frequency": 1,
    },
}
```

This means Dexbotic reads annotation files from
`data/objnav` and the corresponding videos from `data/objnav/video` by default.

### Directory Layout

Convert Uni-NaVid or other navigation data into the following layout:

```text
data/objnav/
  00006-HkseAnWCgqk_22836_book_rack.jsonl
  00164-XfUxBGTFQQb_10993_table.jsonl
  ...
  video/
    00006-HkseAnWCgqk_22836_book_rack.mp4
    00164-XfUxBGTFQQb_10993_table.mp4
    ...
```

Each `*.jsonl` file represents one navigation episode. Each line in the file is
one frame or timestep from that episode. The video filename referenced by
`images_1.url` must exist under `data/objnav/video`.

If your data is stored somewhere else, update `annotations` and
`data_path_prefix` in `uninavid_official.py`.

### JSONL Format

Each line is a JSON object. Dexbotic currently uses the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `images_1` | object | Video reference for the current frame. |
| `images_1.type` | string | Should be `"video"`. |
| `images_1.url` | string | Video filename, for example `"00006-HkseAnWCgqk_22836_book_rack.mp4"`. Dexbotic takes the basename and joins it with `data_path_prefix`. |
| `images_1.frame_idx` | int | Frame index within the episode. Rows are sorted by this field during loading. |
| `state` | array | Navigation state for the current timestep. The current sample data uses 7 values. |
| `prompt` | string | Navigation instruction prompt. It should include the `<image>` placeholder expected by the Uni-NaVid prompt template. |
| `is_robot` | bool | Sample marker used by the Dexbotic data format. |
| `answer` | string | Supervised action label, written as space-separated discrete actions, for example `"left left left left"`. Valid actions are `forward`, `left`, `right`, and `stop`. |

During loading, `DexUniNaVidDataset` scans all `*.jsonl` files under
`annotations`, sorts rows by `images_1.frame_idx`, finds the matching video from
`data_path_prefix`, and samples historical frame prefixes for training. Extra
fields preserved from upstream conversion can remain in the JSONL files, but
they are not required by the current loader.

## Training

Use [`playground/example_uninavid_exp.py`](../playground/example_uninavid_exp.py)
as the launch script. Before training, update the paths in the script to point
to your local checkpoint and dataset. In the default config:

- `DataConfig.dataset_name` is `uninavid_objnav`.
- `DataConfig.video_fps` is `1`.
- `DataConfig.dex_use_nav_augment` enables navigation history augmentation.
- `ModelConfig.model_name_or_path` defaults to `./checkpoints/dexbotic-uninavid`
  (download pretrained weights from **Resources** above).
- `TrainerConfig.output_dir` defaults to `./user_checkpoints/dexbotic/uninavid/<date>`.

Start SFT with:

```bash
cd /path/to/dexbotic
torchrun --nproc_per_node=8 playground/example_uninavid_exp.py --task train
```

## Inference

Start the Uni-NaVid inference server from the Dexbotic repository:

```bash
CUDA_VISIBLE_DEVICES=0 python playground/example_uninavid_exp.py --task inference
```

The server listens on port `7892` by default and exposes:

```text
POST http://localhost:7892/process_frame
```

The endpoint accepts multipart form data with `text` and one or more `image`
fields. For online navigation, `episode_first_frame=true` resets navigation
memory at the beginning of an episode. `run_model=false` can be used to add image
frames to the history buffer without triggering model generation.

You can also run a quick single-image sanity check:

```bash
CUDA_VISIBLE_DEVICES=0 python playground/example_uninavid_exp.py --task inference_single \
  --image_path test_data/uninavid_test.png \
  --prompt "Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug."
```

## Evaluation Results

Uni-NaVid evaluation is run through
[dexbotic-benchmark](https://github.com/dexmal/dexbotic-benchmark). With the
current Dexbotic Uni-NaVid setup, the results on the R2R and RxR datasets in
dexbotic-benchmark are:

| Dataset | Metric | Result |
| --- | --- | --- |
| R2R | SR | 52.3 |
| RxR | SR | 55.1 |

The exact score may vary with checkpoint, inference configuration, benchmark
version, and environment setup.

## Related Files

| Purpose | Path |
| --- | --- |
| Dataset registration | [`dexbotic/data/data_source/uninavid_official.py`](../dexbotic/data/data_source/uninavid_official.py) |
| JSONL + video loader | [`dexbotic/data/dataset/dex_uninavid_dataset.py`](../dexbotic/data/dataset/dex_uninavid_dataset.py) |
| Uni-NaVid experiment implementation | [`dexbotic/exp/uninavid_exp.py`](../dexbotic/exp/uninavid_exp.py) |
| Example launch script | [`playground/example_uninavid_exp.py`](../playground/example_uninavid_exp.py) |
