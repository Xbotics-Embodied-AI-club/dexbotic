"""单卡 LoRA 微调 DM0.5：现场配方，一张 32 GB 的卡约 70 分钟，训完直接做开环自检。

    python -m dexbotic.so101.train_lora --gpu 0

配方：每次前向 8 个样本、梯度累积 6 次（一次参数更新用 48 个样本，与发布模型的完整训练相同），
训 300 步，前 30 步学习率从 0 升到 1e-4。累积次数是为了在单卡上凑出与完整训练相同的更新批量，别为求快去掉。

训完用第 300 步的检查点起推理服务，在三集训练数据上做开环自检，结果写在
`<输出目录>/openloop.json`：模型误差与「保持当前姿态不动」的误差并列给出。300 步只是入门，
两者通常处在同一水平（同一配方两次实测：7.1 与 9.7，对照 9.4）；开环自检每次推理起始噪声随机，
接近对照时一次结果说明不了太多。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

PER_DEVICE, GRAD_ACCUM, STEPS, WARMUP = 8, 6, 300, 30
#: 开环自检用的三集：两集仿真、一集真机，覆盖两种画面来源。
OPENLOOP_EPISODES = (
    "sim_cube40_ep00000.jsonl",
    "sim_cube20_ep00001.jsonl",
    "real_pick_up_a_cube_and_place_in_the_bin_ep00000.jsonl",
)


def wait_for_server(port: int, server: subprocess.Popen, log: pathlib.Path) -> None:
    """等推理服务能响应；服务进程提前退出就把日志尾巴打出来再停。"""
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
            return
        except urllib.error.HTTPError:
            return  # 服务已经在应答（根路径 404 也算起来了）
        except OSError:
            pass
        if server.poll() is not None:
            raise SystemExit(f"推理服务退出了，日志：\n{log.read_text()[-2000:]}")
        time.sleep(10)


def main() -> int:
    from dexbotic.so101.layout import DATASETS_DIR, ROOT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(ROOT) / "runs" / "lora_single_gpu")
    ap.add_argument("--port", type=int, default=7891, help="开环自检时推理服务用的端口")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl = pathlib.Path(DATASETS_DIR) / "so101-dexdata" / "jsonl"
    if not jsonl.is_dir():
        raise SystemExit(f"训练数据还没转换：{jsonl} 不存在；先运行 python -m dexbotic.so101.prepare_data")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "TOKENIZERS_PARALLELISM": "false"}
    env.setdefault("WANDB_MODE", "offline")
    train = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port",
        env.get("DM05_MASTER_PORT", "29500"),
        "-m",
        "dexbotic.so101.dm05_exp",
        "--task",
        "train",
        "--trainer-config.per-device-train-batch-size",
        str(PER_DEVICE),
        "--trainer-config.gradient-accumulation-steps",
        str(GRAD_ACCUM),
        "--trainer-config.num-train-steps",
        str(STEPS),
        "--trainer-config.save-steps",
        "100",
        "--optimizer-config.warmup-steps",
        str(WARMUP),
        "--trainer-config.output-dir",
        str(args.out),
        # 基座直接从 HF 缓存加载：推理服务第一次启动时已经下过同一份，不再另存一份到工作区。
        "--model-config.model-name-or-path",
        "Dexmal/DM05",
    ]
    print("训练：", " ".join(train), flush=True)
    with open(args.out / "train.log", "w") as log:
        subprocess.run(train, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)

    checkpoint = args.out / f"checkpoint-{STEPS}"
    server_log = args.out / "server.log"
    with open(server_log, "w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "dexbotic.so101.dm05_exp",
                "--task",
                "inference",
                "--model-config.model-name-or-path",
                str(checkpoint),
                "--inference-config.port",
                str(args.port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        wait_for_server(args.port, server, server_log)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dexbotic.so101.openloop_check",
                "--endpoint",
                f"http://127.0.0.1:{args.port}/v1/infer",
                "--jsonl",
                *[str(jsonl / name) for name in OPENLOOP_EPISODES],
                "--out",
                str(args.out / "openloop.json"),
            ],
            env=env,
            check=True,
        )
    finally:
        server.terminate()
        server.wait(timeout=60)
    print(
        f"开环自检结果：{args.out / 'openloop.json'}\n"
        "（模型误差与「保持不动」并列给出；300 步时两者通常处在同一水平）\n"
        f"部署自己的模型：python -m dexbotic.so101.serve --checkpoint {args.out / f'checkpoint-{STEPS}'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
