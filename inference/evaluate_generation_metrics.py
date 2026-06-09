#!/usr/bin/env python3
"""Evaluate generated robot videos against RobotWin ground-truth videos.

Default ground truth layout:

    data/robotwin_dataset/clean/<task>/videos/<episode>.mp4

Prediction layouts are matched automatically. Supported examples:

    <pred_root>/clean/<task>/videos/<episode>.mp4
    <pred_root>/<task>/videos/<episode>.mp4
    <pred_root>/<task>/<episode>.mp4
    <pred_root>/<task>_<episode>.mp4

PSNR and SSIM only need numpy, imageio, PIL, scipy/skimage. LPIPS, FID, and
FVD need their optional PyTorch dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VideoPair:
    task: str
    episode: str
    gt_path: Path
    pred_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/LPIPS/FID/FVD for generated RobotWin videos."
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=REPO_ROOT / "data/robotwin_dataset",
        help="RobotWin dataset root. Default: data/robotwin_dataset",
    )
    parser.add_argument(
        "--split",
        default="clean",
        help="RobotWin split under gt-root. Default: clean",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=REPO_ROOT / "inference/robotwin_generated_videos",
        help="Generated video root. Default: inference/robotwin_generated_videos",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "inference/robotwin_generation_metrics.json",
        help="Where to write metrics JSON.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "lpips", "fid", "fvd"],
        choices=["psnr", "ssim", "lpips", "fid", "fvd"],
        help="Metrics to compute.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task-name filter.")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all matched videos.")
    parser.add_argument("--max-frames", type=int, default=32, help="Frames per video. 0 means all.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame before max-frames.")
    parser.add_argument(
        "--fid-max-frames",
        type=int,
        default=8,
        help="Frames per video used for FID. Keeps evaluation memory bounded.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Resize both GT and prediction frames before metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Feature batch size.")
    parser.add_argument("--device", default="cuda", help="PyTorch device for LPIPS/FID/FVD.")
    parser.add_argument(
        "--fvd-backbone",
        choices=["auto", "r3d18"],
        default="auto",
        help=(
            "auto uses torchmetrics FrechetVideoDistance when available; otherwise "
            "uses a torchvision r3d_18 Frechet-video-distance proxy."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def natural_sort_key(path: Path) -> Tuple[str, int | str]:
    stem = path.stem
    return (str(path.parent), int(stem) if stem.isdigit() else stem)


def find_gt_videos(gt_root: Path, split: str, tasks: Optional[Sequence[str]]) -> List[Tuple[str, str, Path]]:
    split_root = gt_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"GT split root not found: {split_root}")

    task_filter = set(tasks) if tasks else None
    videos: List[Tuple[str, str, Path]] = []
    for task_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        task = task_dir.name
        if task_filter is not None and task not in task_filter:
            continue
        video_dir = task_dir / "videos"
        if not video_dir.exists():
            continue
        for video_path in sorted((p for p in video_dir.iterdir() if is_video_file(p)), key=natural_sort_key):
            videos.append((task, video_path.stem, video_path))
    return videos


def index_pred_videos(pred_root: Path) -> Dict[str, Path]:
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    all_videos = sorted((p for p in pred_root.rglob("*") if is_video_file(p)), key=natural_sort_key)
    index: Dict[str, Path] = {}
    basename_counts: Dict[str, int] = {}

    for path in all_videos:
        rel_parts = path.relative_to(pred_root).parts
        stem = path.stem
        basename_counts[stem] = basename_counts.get(stem, 0) + 1

        keys = set()
        if len(rel_parts) >= 3 and rel_parts[-2] == "videos":
            task = rel_parts[-3]
            if task in {"clean", "randomized"} and len(rel_parts) >= 4:
                task = rel_parts[-4]
            keys.add(f"{task}/{stem}")
        if len(rel_parts) >= 2:
            keys.add(f"{rel_parts[-2]}/{stem}")
        if "_" in stem:
            task, episode = stem.rsplit("_", 1)
            keys.add(f"{task}/{episode}")

        for key in keys:
            index.setdefault(key, path)

    for path in all_videos:
        if basename_counts[path.stem] == 1:
            index.setdefault(path.stem, path)

    return index


def match_pairs(
    gt_videos: Sequence[Tuple[str, str, Path]],
    pred_index: Dict[str, Path],
    max_videos: int,
) -> Tuple[List[VideoPair], List[str]]:
    pairs: List[VideoPair] = []
    missing: List[str] = []
    for task, episode, gt_path in gt_videos:
        pred_path = pred_index.get(f"{task}/{episode}") or pred_index.get(episode)
        if pred_path is None:
            missing.append(f"{task}/{episode}")
            continue
        pairs.append(VideoPair(task=task, episode=episode, gt_path=gt_path, pred_path=pred_path))
        if max_videos > 0 and len(pairs) >= max_videos:
            break
    return pairs, missing


def resize_frame(frame: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    image = Image.fromarray(frame)
    return np.asarray(image.resize((w, h), Image.BILINEAR))


def read_video(
    path: Path,
    max_frames: int,
    frame_stride: int,
    resize_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    errors: List[str] = []
    for reader in (read_video_cv2, read_video_imageio, read_video_torchvision):
        try:
            return reader(path, max_frames, frame_stride, resize_hw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{reader.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Failed to decode video {path}. Install one of: opencv-python, imageio[ffmpeg], "
        f"or torchvision with video support. Decoder errors: {' | '.join(errors)}"
    )


def read_video_cv2(
    path: Path,
    max_frames: int,
    frame_stride: int,
    resize_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cv2.VideoCapture could not open file")
    frames: List[np.ndarray] = []
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_stride > 1 and idx % frame_stride != 0:
                idx += 1
                continue
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                frame = resize_frame(frame, resize_hw)
            frames.append(frame)
            idx += 1
            if max_frames > 0 and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError("no frames decoded")
    return np.stack(frames, axis=0)


def read_video_imageio(
    path: Path,
    max_frames: int,
    frame_stride: int,
    resize_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    import imageio.v3 as iio  # type: ignore

    frames: List[np.ndarray] = []
    for idx, frame in enumerate(iio.imiter(path)):
        if frame_stride > 1 and idx % frame_stride != 0:
            continue
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if resize_hw is not None:
            frame = resize_frame(frame, resize_hw)
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


def read_video_torchvision(
    path: Path,
    max_frames: int,
    frame_stride: int,
    resize_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    from torchvision.io import read_video  # type: ignore

    video, _, _ = read_video(str(path), pts_unit="sec", output_format="THWC")
    frames_np = video.cpu().numpy()
    if frame_stride > 1:
        frames_np = frames_np[::frame_stride]
    if max_frames > 0:
        frames_np = frames_np[:max_frames]
    if resize_hw is not None:
        frames_np = np.stack([resize_frame(frame, resize_hw) for frame in frames_np], axis=0)
    if frames_np.size == 0:
        raise ValueError("no frames decoded")
    return frames_np.astype(np.uint8, copy=False)


def align_videos(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    frame_count = min(gt.shape[0], pred.shape[0])
    gt = gt[:frame_count]
    pred = pred[:frame_count]
    if gt.shape[1:3] != pred.shape[1:3]:
        pred = np.stack([resize_frame(frame, gt.shape[1:3]) for frame in pred], axis=0)
    return gt, pred


def compute_pair_metrics(gt: np.ndarray, pred: np.ndarray, want_psnr: bool, want_ssim: bool) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if want_psnr:
        values = [compute_psnr(gt[i], pred[i]) for i in range(gt.shape[0])]
        out["psnr"] = float(np.mean(values))
    if want_ssim:
        values = [compute_ssim(gt[i], pred[i]) for i in range(gt.shape[0])]
        out["ssim"] = float(np.mean(values))
    return out


def compute_psnr(real: np.ndarray, fake: np.ndarray) -> float:
    real_f = real.astype(np.float64)
    fake_f = fake.astype(np.float64)
    mse = np.mean((real_f - fake_f) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def compute_ssim(real: np.ndarray, fake: np.ndarray) -> float:
    real_f = real.astype(np.float64)
    fake_f = fake.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    def smooth(x: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter  # type: ignore

            return uniform_filter(x, size=(11, 11), mode="reflect")
        except Exception:
            try:
                import cv2  # type: ignore

                return cv2.GaussianBlur(x, (11, 11), 1.5)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("SSIM needs scipy or opencv-python for local smoothing.") from exc

    scores: List[float] = []
    for channel in range(real_f.shape[2]):
        x = real_f[..., channel]
        y = fake_f[..., channel]
        mux = smooth(x)
        muy = smooth(y)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = smooth(x * x) - mux2
        sigy2 = smooth(y * y) - muy2
        sigxy = smooth(x * y) - muxy
        ssim_map = ((2 * muxy + c1) * (2 * sigxy + c2)) / (
            (mux2 + muy2 + c1) * (sigx2 + sigy2 + c2)
        )
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    try:
        from scipy import linalg  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("FID/FVD need scipy for matrix square root.") from exc

    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    mu1, mu2 = real_features.mean(axis=0), fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
        sigma2 = np.array([[sigma2]])
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + eps) @ (sigma2 + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyTorch is required for LPIPS/FID/FVD. Activate the Motus/RoboTwin env.") from exc
    return torch


def choose_device(device: str):
    torch = require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def frames_to_torch(frames: np.ndarray, device, size_hw: Optional[Tuple[int, int]] = None):
    torch = require_torch()
    x = torch.from_numpy(frames).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(0, 3, 1, 2)
    if size_hw is not None and tuple(x.shape[-2:]) != size_hw:
        import torch.nn.functional as F

        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return x


def compute_lpips(pairs: Sequence[Tuple[np.ndarray, np.ndarray]], device_name: str, batch_size: int) -> Optional[float]:
    try:
        import lpips  # type: ignore
    except Exception:
        print("Skipping LPIPS: install dependency with `pip install lpips`.", file=sys.stderr)
        return None

    torch = require_torch()
    device = choose_device(device_name)
    model = lpips.LPIPS(net="alex").to(device).eval()
    values: List[float] = []
    with torch.no_grad():
        for gt, pred in pairs:
            gt_tensor = frames_to_torch(gt, device) * 2 - 1
            pred_tensor = frames_to_torch(pred, device) * 2 - 1
            for start in range(0, gt_tensor.shape[0], batch_size):
                score = model(gt_tensor[start : start + batch_size], pred_tensor[start : start + batch_size])
                values.extend(score.flatten().detach().cpu().numpy().astype(float).tolist())
    return float(np.mean(values)) if values else None


def extract_inception_features(frames_list: Sequence[np.ndarray], device_name: str, batch_size: int) -> np.ndarray:
    torch = require_torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3  # type: ignore

    device = choose_device(device_name)
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
    model.fc = nn.Identity()
    model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    features: List[np.ndarray] = []
    with torch.no_grad():
        batch: List[np.ndarray] = []
        for frames in frames_list:
            for frame in frames:
                batch.append(frame)
                if len(batch) >= batch_size:
                    x = frames_to_torch(np.stack(batch), device, size_hw=(299, 299))
                    x = (x - mean) / std
                    feat = model(x)
                    features.append(feat.detach().cpu().numpy())
                    batch = []
        if batch:
            x = frames_to_torch(np.stack(batch), device, size_hw=(299, 299))
            x = (x - mean) / std
            feat = model(x)
            features.append(feat.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(
    gt_videos: Sequence[np.ndarray],
    pred_videos: Sequence[np.ndarray],
    device_name: str,
    batch_size: int,
    fid_max_frames: int,
) -> Optional[float]:
    try:
        require_torch()
        import torchvision  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping FID: {exc}", file=sys.stderr)
        return None

    def take_frames(video: np.ndarray) -> np.ndarray:
        if fid_max_frames <= 0 or video.shape[0] <= fid_max_frames:
            return video
        idx = np.linspace(0, video.shape[0] - 1, fid_max_frames).round().astype(int)
        return video[idx]

    real_features = extract_inception_features([take_frames(v) for v in gt_videos], device_name, batch_size)
    fake_features = extract_inception_features([take_frames(v) for v in pred_videos], device_name, batch_size)
    return frechet_distance(real_features, fake_features)


def compute_fvd_torchmetrics(
    gt_videos: Sequence[np.ndarray],
    pred_videos: Sequence[np.ndarray],
    device_name: str,
) -> Optional[float]:
    try:
        import torch  # type: ignore
        from torchmetrics.video.fvd import FrechetVideoDistance  # type: ignore
    except Exception:
        return None

    device = choose_device(device_name)
    metric = FrechetVideoDistance().to(device)
    with torch.no_grad():
        for gt, pred in zip(gt_videos, pred_videos):
            real = torch.from_numpy(gt[None]).to(device=device, dtype=torch.uint8)
            fake = torch.from_numpy(pred[None]).to(device=device, dtype=torch.uint8)
            metric.update(real, real=True)
            metric.update(fake, real=False)
    return float(metric.compute().detach().cpu().item())


def extract_r3d18_features(videos: Sequence[np.ndarray], device_name: str, batch_size: int) -> np.ndarray:
    torch = require_torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models.video import R3D_18_Weights, r3d_18  # type: ignore

    device = choose_device(device_name)
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = nn.Identity()
    model.to(device).eval()

    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.21699], device=device).view(1, 3, 1, 1, 1)
    features: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(videos), batch_size):
            batch = videos[start : start + batch_size]
            tensors = []
            for video in batch:
                x = frames_to_torch(video, device, size_hw=(112, 112))  # [T, C, H, W]
                x = x.permute(1, 0, 2, 3)  # [C, T, H, W]
                tensors.append(x)
            x = torch.stack(tensors, dim=0)
            x = F.interpolate(x, size=(16, 112, 112), mode="trilinear", align_corners=False)
            x = (x - mean) / std
            feat = model(x)
            features.append(feat.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fvd(
    gt_videos: Sequence[np.ndarray],
    pred_videos: Sequence[np.ndarray],
    device_name: str,
    batch_size: int,
    backbone: str,
) -> Tuple[Optional[float], str]:
    if backbone == "auto":
        value = compute_fvd_torchmetrics(gt_videos, pred_videos, device_name)
        if value is not None:
            return value, "torchmetrics"

    try:
        require_torch()
        import torchvision  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping FVD: {exc}", file=sys.stderr)
        return None, "unavailable"

    print(
        "Using r3d_18 Frechet video distance proxy for FVD. "
        "Install torchmetrics with FVD support for the canonical implementation.",
        file=sys.stderr,
    )
    real_features = extract_r3d18_features(gt_videos, device_name, batch_size)
    fake_features = extract_r3d18_features(pred_videos, device_name, batch_size)
    return frechet_distance(real_features, fake_features), "r3d18_proxy"


def mean_metric(per_pair: Sequence[Dict[str, float]], key: str) -> Optional[float]:
    values = [item[key] for item in per_pair if key in item]
    return float(np.mean(values)) if values else None


def json_safe(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    resize_hw = tuple(args.resize) if args.resize is not None else None

    gt_videos = find_gt_videos(args.gt_root, args.split, args.tasks)
    pred_index = index_pred_videos(args.pred_root)
    pairs, missing = match_pairs(gt_videos, pred_index, args.max_videos)
    if not pairs:
        raise RuntimeError(
            f"No matched videos. GT root={args.gt_root / args.split}, pred root={args.pred_root}"
        )

    print(f"GT videos found: {len(gt_videos)}")
    print(f"Prediction videos indexed: {len(set(pred_index.values()))}")
    print(f"Matched pairs: {len(pairs)}")
    if missing:
        print(f"Missing predictions for {len(missing)} GT videos; first 10: {missing[:10]}")

    loaded_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    per_pair_metrics: List[Dict[str, float | str]] = []
    want_psnr = "psnr" in args.metrics
    want_ssim = "ssim" in args.metrics

    for idx, pair in enumerate(pairs, start=1):
        gt = read_video(pair.gt_path, args.max_frames, args.frame_stride, resize_hw)
        pred = read_video(pair.pred_path, args.max_frames, args.frame_stride, resize_hw)
        gt, pred = align_videos(gt, pred)
        loaded_pairs.append((gt, pred))

        item: Dict[str, float | str] = {
            "task": pair.task,
            "episode": pair.episode,
            "gt_path": str(pair.gt_path),
            "pred_path": str(pair.pred_path),
            "frames": gt.shape[0],
        }
        item.update(compute_pair_metrics(gt, pred, want_psnr, want_ssim))
        per_pair_metrics.append(item)
        if args.verbose and idx % 10 == 0:
            print(f"Loaded/evaluated {idx}/{len(pairs)} pairs")

    gt_arrays = [gt for gt, _ in loaded_pairs]
    pred_arrays = [pred for _, pred in loaded_pairs]

    summary: Dict[str, object] = {
        "gt_root": str(args.gt_root),
        "split": args.split,
        "pred_root": str(args.pred_root),
        "num_gt_videos": len(gt_videos),
        "num_pred_videos_indexed": len(set(pred_index.values())),
        "num_pairs": len(pairs),
        "num_missing_predictions": len(missing),
        "max_frames": args.max_frames,
        "frame_stride": args.frame_stride,
        "resize": list(resize_hw) if resize_hw else None,
    }

    for key in ("psnr", "ssim"):
        value = mean_metric(per_pair_metrics, key)  # type: ignore[arg-type]
        if value is not None:
            summary[key] = value

    if "lpips" in args.metrics:
        summary["lpips"] = compute_lpips(loaded_pairs, args.device, args.batch_size)
    if "fid" in args.metrics:
        summary["fid"] = compute_fid(
            gt_arrays, pred_arrays, args.device, args.batch_size, args.fid_max_frames
        )
    if "fvd" in args.metrics:
        fvd, backend = compute_fvd(
            gt_arrays, pred_arrays, args.device, args.batch_size, args.fvd_backbone
        )
        summary["fvd"] = fvd
        summary["fvd_backend"] = backend

    result = {
        "summary": summary,
        "pairs": per_pair_metrics,
        "missing_predictions": missing,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")

    print("\n=== Generation Metrics ===")
    for key in ["fid", "fvd", "ssim", "lpips", "psnr"]:
        if key in summary:
            print(f"{key:>8}: {summary[key]}")
    if "fvd_backend" in summary:
        print(f"{'fvd_backend':>8}: {summary['fvd_backend']}")
    print(f"\nWrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
