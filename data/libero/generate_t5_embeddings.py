#!/usr/bin/env python3
"""Generate one WAN UMT5 embedding per unique LIBERO task."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.lerobot.add_t5_cache_to_lerobot_dataset import _encode_t5, _init_wan_t5_encoder
from data.libero.libero_dataset import load_jsonl, suite_name_from_root


def generate(
    dataset_dir: Path,
    cache_dir: Path,
    wan_path: str,
    device: str,
    text_len: int,
    overwrite: bool,
) -> None:
    suite_roots = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and (path / "meta" / "info.json").exists()
    )
    pending = []
    for root in suite_roots:
        suite = suite_name_from_root(root)
        for row in load_jsonl(root / "meta" / "tasks.jsonl"):
            output = cache_dir / "t5" / suite / f"task_{int(row['task_index']):02d}.pt"
            if overwrite or not output.exists():
                pending.append((suite, int(row["task_index"]), str(row["task"]), output))
    if not pending:
        print("All LIBERO T5 embeddings already exist")
        return

    encoder = _init_wan_t5_encoder(wan_path=wan_path, device=device, text_len=text_len)
    for index, (suite, task_index, instruction, output) in enumerate(pending, start=1):
        output.parent.mkdir(parents=True, exist_ok=True)
        embedding = _encode_t5(encoder, instruction, device=device)
        temporary = output.with_suffix(".pt.tmp")
        torch.save(embedding, temporary)
        temporary.replace(output)
        print(f"[{index}/{len(pending)}] {suite} task {task_index}: {tuple(embedding.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--wan-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(
        args.dataset_dir.resolve(),
        args.cache_dir.resolve(),
        args.wan_path,
        args.device,
        args.text_len,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
