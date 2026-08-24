"""Lazy local LeRobot dataset loaders.

These loaders keep the Motus-facing behavior of the specialized LeRobot
datasets, but avoid constructing LeRobotDataset/MultiLeRobotDataset at
initialization time. They build a lightweight episode manifest and read one
episode parquet on demand in __getitem__.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.video_utils import decode_video_frames

from data.utils.norm import load_normalization_stats
from utils.vlm_utils import preprocess_vlm_messages_lap

from .lerobot_agibot_dataset import LeRobotAgiBotDataset
from .lerobot_dataset import AutoProcessor, LeRobotMotusDataset, preprocess_vlm_messages, tensor_to_pil
from .lerobot_interndata_dataset import LeRobotInternDataDataset, build_interndata_canonical55
from .lerobot_robocoin_dataset import LeRobotRoboCOINDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LazyEpisode:
    task_name: str
    task_idx: int
    dataset_root: Path
    episode_index: int
    length: int
    data_path: Path
    episode_meta: Dict[str, Any]


class LazyLeRobotMeta:
    def __init__(self, root: Path, info: Dict[str, Any], episodes: Dict[int, Dict[str, Any]]):
        self.root = root
        self.info = info
        self.features = info.get("features", {}) or {}
        self.episodes = episodes
        self.video_keys = [
            key
            for key, feature in self.features.items()
            if isinstance(feature, Mapping) and feature.get("dtype") in {"video", "image"}
        ]

    def get_video_file_path(self, episode_index: int, video_key: str) -> str:
        chunks_size = int(self.info.get("chunks_size", 1000) or 1000)
        episode_chunk = int(episode_index) // chunks_size
        pattern = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        return str(pattern).format(
            episode_index=int(episode_index),
            episode_chunk=episode_chunk,
            video_key=video_key,
        )


class LazyLeRobotMedia:
    def __init__(
        self,
        root: Path,
        info: Dict[str, Any],
        episodes: Dict[int, Dict[str, Any]],
        video_backend: str,
    ):
        self.root = root
        self.info = info
        self.features = info.get("features", {}) or {}
        self.meta = LazyLeRobotMeta(root, info, episodes)
        self.video_backend = video_backend
        fps = int(info.get("fps", 30) or 30)
        self.tolerance_s = 1 / fps - 1e-4
        self.num_frames = int(info.get("total_frames", 0) or 0)
        self.num_episodes = int(info.get("total_episodes", len(episodes)) or len(episodes))


class LazyLeRobotMixin:
    lazy_episodes: List[LazyEpisode]
    lazy_media: List[LazyLeRobotMedia]

    def _init_lazy_common(
        self,
        dataset_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        root: Optional[str] = None,
        split: Optional[str] = None,
        global_downsample_rate: int = 1,
        video_action_freq_ratio: int = 5,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (736, 640),
        max_episodes: Optional[int] = 10000,
        max_episodes_per_task: Optional[int] = None,
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
        enable_t5_fallback: bool = False,
        t5_wan_path: Optional[str] = None,
        t5_text_len: int = 512,
        t5_folder_name: str = "t5_embedding",
        t5_device: Optional[str] = None,
        video_backend: Optional[str] = None,
        embodiment_type: str = "aloha_agilex_2",
        task_mode: str = "single",
        task_name: Optional[str | List[str]] = None,
        task_discovery: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if root is None and dataset_dir is not None and os.path.exists(str(dataset_dir)):
            root = str(dataset_dir)
            if repo_id is None:
                repo_id = Path(root).name
        if root is None:
            raise ValueError("Lazy LeRobot datasets require a local root path")
        if repo_id is None:
            repo_id = Path(root).name

        self.repo_id = str(repo_id)
        self.root = str(root)
        self.split = split
        self.global_downsample_rate = int(global_downsample_rate)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.num_video_frames = int(num_video_frames)
        self.video_size = tuple(video_size)
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        self.max_episodes = max_episodes
        self.max_episodes_per_task = max_episodes_per_task
        self.image_aug = image_aug
        self.task_mode = task_mode
        if isinstance(task_name, str) and task_name.strip().lower() in {"", "none", "null"}:
            task_name = None
        self.task_name = task_name
        self.task_discovery = task_discovery
        self.enable_t5_fallback = bool(enable_t5_fallback)
        self.t5_wan_path = t5_wan_path or os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT")
        self.t5_text_len = int(t5_text_len)
        self.t5_folder_name = str(t5_folder_name)
        self.t5_device = t5_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._t5_encoder = None
        self._episode_embedding_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._active_lazy_episode: Optional[LazyEpisode] = None
        self.lerobot_dataset = None

        self.vlm_processor = None
        if vlm_checkpoint_path:
            if AutoProcessor is None:
                logger.warning("transformers is not installed, cannot load VLM processor")
            else:
                try:
                    self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                    logger.info("Loaded VLM processor from %s", vlm_checkpoint_path)
                except Exception as exc:
                    logger.warning("Failed to load VLM processor: %s", exc)

        resolved_video_backend = video_backend if video_backend is not None else "pyav"
        task_roots = self._resolve_lazy_task_roots(Path(root), task_name, task_discovery, task_mode)
        self.lazy_media = []
        self.lazy_episodes = []

        metadata_cache: Dict[str, Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]] = {}
        if max_episodes is not None and int(max_episodes) > 0:
            task_roots, selected_episode_ids = self._select_first_lazy_episodes(
                task_roots,
                int(max_episodes),
                metadata_cache,
            )
        else:
            selected_episode_ids = self._read_episode_ids_by_task(task_roots, metadata_cache)
        self.repo_ids = [name for name, _ in task_roots]

        for task_idx, (task_name_value, task_root) in enumerate(task_roots):
            info, episodes = metadata_cache.get(task_name_value) or self._read_lazy_metadata(task_root)
            media = LazyLeRobotMedia(task_root, info, episodes, resolved_video_backend)
            self.lazy_media.append(media)
            for episode_index in selected_episode_ids.get(task_name_value, []):
                ep_meta = episodes.get(int(episode_index))
                if ep_meta is None:
                    continue
                data_path = self._episode_data_path(task_root, info, int(episode_index))
                length = int(ep_meta.get("length") or self._parquet_num_rows(data_path))
                if length <= 0:
                    continue
                self.lazy_episodes.append(
                    LazyEpisode(
                        task_name=task_name_value,
                        task_idx=task_idx,
                        dataset_root=task_root,
                        episode_index=int(episode_index),
                        length=length,
                        data_path=data_path,
                        episode_meta=dict(ep_meta),
                    )
                )

        if not self.lazy_episodes:
            raise ValueError(f"No lazy LeRobot episodes found under {root}")
        self.num_episodes = len(self.lazy_episodes)

        current_dir = Path(__file__).parent.parent
        stat_path = current_dir / "utils" / "stat.json"
        self.action_min, self.action_max = load_normalization_stats(str(stat_path), embodiment_type)
        logger.info(
            "Lazy LeRobot dataset initialized: repo_id=%s, root=%s, episodes=%d, repos=%d",
            self.repo_id,
            self.root,
            len(self.lazy_episodes),
            len(self.repo_ids),
        )

    def _resolve_lazy_task_roots(
        self,
        root_path: Path,
        task_name: Optional[str | List[str]],
        task_discovery: Optional[Dict[str, Any]],
        task_mode: str,
    ) -> List[Tuple[str, Path]]:
        if task_mode == "single":
            return [(self.repo_id, root_path)]
        if task_discovery and bool(task_discovery.get("enabled", False)):
            names = self._discover_lazy_task_names(root_path, task_discovery)
        elif task_name is None:
            names = sorted(path.name for path in root_path.iterdir() if path.is_dir())
        elif isinstance(task_name, list):
            names = [str(name) for name in task_name]
        else:
            names = [str(task_name)]
        roots = [(name, root_path / name) for name in names]
        missing = [str(path) for _, path in roots if not path.is_dir()]
        if missing:
            raise ValueError(f"Lazy LeRobot task roots not found: {missing[:5]}")
        return roots

    def _discover_lazy_task_names(self, root_path: Path, task_discovery: Dict[str, Any]) -> List[str]:
        suffix = str(task_discovery.get("suffix", "") or "")
        required = task_discovery.get("required", ["meta/info.json", "data", "videos"])
        if isinstance(required, str):
            required = [required]

        task_names: List[str] = []
        max_tasks = (
            int(self.max_episodes)
            if self.max_episodes_per_task is None and self.max_episodes is not None and int(self.max_episodes) > 0
            else 0
        )
        for dir_path, dir_names, _ in os.walk(root_path):
            dir_names.sort()
            path = Path(dir_path)
            if suffix and not path.name.endswith(suffix):
                continue
            if not all((path / str(rel_path)).exists() for rel_path in required):
                continue
            if not (path / "meta" / "episodes.jsonl").is_file():
                continue
            task_names.append(path.relative_to(root_path).as_posix())
            dir_names[:] = []
            if max_tasks > 0 and len(task_names) >= max_tasks:
                break

        task_names = sorted(task_names)
        if not task_names:
            raise ValueError(f"No lazy LeRobot datasets discovered under {root_path} with suffix {suffix!r}")
        return task_names

    def _read_lazy_metadata(self, task_root: Path) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
        info = json.loads((task_root / "meta" / "info.json").read_text(encoding="utf-8"))
        episodes: Dict[int, Dict[str, Any]] = {}
        with (task_root / "meta" / "episodes.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                episode = json.loads(line)
                episodes[int(episode["episode_index"])] = episode
        return info, episodes

    def _read_episode_ids_by_task(
        self,
        task_roots: List[Tuple[str, Path]],
        metadata_cache: Optional[Dict[str, Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]]] = None,
    ) -> Dict[str, List[int]]:
        result: Dict[str, List[int]] = {}
        for task_name, task_root in task_roots:
            info, episodes = self._read_lazy_metadata(task_root)
            if metadata_cache is not None:
                metadata_cache[task_name] = (info, episodes)
            result[task_name] = self._limit_episode_ids_for_task(sorted(episodes))
        return result

    def _select_first_lazy_episodes(
        self,
        task_roots: List[Tuple[str, Path]],
        max_episodes: int,
        metadata_cache: Dict[str, Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]],
    ) -> Tuple[List[Tuple[str, Path]], Dict[str, List[int]]]:
        selected_roots: List[Tuple[str, Path]] = []
        selected_episode_ids: Dict[str, List[int]] = {}
        remaining = max_episodes
        for task_name, task_root in task_roots:
            info, episodes = self._read_lazy_metadata(task_root)
            metadata_cache[task_name] = (info, episodes)
            episode_ids = self._limit_episode_ids_for_task(sorted(episodes))
            if not episode_ids:
                continue
            chosen = episode_ids[:remaining]
            selected_roots.append((task_name, task_root))
            selected_episode_ids[task_name] = chosen
            remaining -= len(chosen)
            if remaining <= 0:
                break
        return selected_roots, selected_episode_ids

    def _limit_episode_ids_for_task(self, episode_ids: List[int]) -> List[int]:
        if self.max_episodes_per_task is None:
            return episode_ids
        limit = int(self.max_episodes_per_task)
        if limit <= 0:
            return []
        return episode_ids[:limit]

    def _episode_data_path(self, task_root: Path, info: Mapping[str, Any], episode_index: int) -> Path:
        chunks_size = int(info.get("chunks_size", 1000) or 1000)
        episode_chunk = episode_index // chunks_size
        pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        return task_root / str(pattern).format(episode_index=episode_index, episode_chunk=episode_chunk)

    @staticmethod
    def _parquet_num_rows(path: Path) -> int:
        return int(pq.ParquetFile(path).metadata.num_rows)

    def __len__(self) -> int:
        return len(self.lazy_episodes) * 1000

    def _sample_lazy_episode(self) -> LazyEpisode:
        return random.choice(self.lazy_episodes)

    def _read_episode_columns(self, episode: LazyEpisode) -> Dict[str, List[Any]]:
        return pq.read_table(episode.data_path).to_pydict()

    def _read_sample_columns(self, episode: LazyEpisode) -> Optional[Dict[str, List[Any]]]:
        try:
            return self._read_episode_columns(episode)
        except (FileNotFoundError, OSError) as exc:
            logger.warning("Failed to read lazy episode parquet %s: %s", episode.data_path, exc)
            return None

    @staticmethod
    def _value_at(columns: Mapping[str, List[Any]], key: str, index: int) -> Any:
        return columns[key][min(max(int(index), 0), len(columns[key]) - 1)]

    @staticmethod
    def _values_at(columns: Mapping[str, List[Any]], key: str, indices: Sequence[int]) -> List[Any]:
        return [LazyLeRobotMixin._value_at(columns, key, index) for index in indices]

    @staticmethod
    def _timestamps_at(columns: Mapping[str, List[Any]], indices: Sequence[int]) -> List[float]:
        return [float(LazyLeRobotMixin._value_at(columns, "timestamp", index)) for index in indices]

    def _row_at(self, columns: Mapping[str, List[Any]], index: int, episode: LazyEpisode) -> Dict[str, Any]:
        row = {key: self._value_at(columns, key, index) for key in columns}
        row.setdefault("episode_index", episode.episode_index)
        row.setdefault("task", self._task_text(episode.episode_meta))
        row.setdefault("tasks", episode.episode_meta.get("tasks", []))
        return row

    @staticmethod
    def _task_text(episode_meta: Mapping[str, Any]) -> str:
        tasks = episode_meta.get("tasks", None)
        if isinstance(tasks, list) and tasks:
            return str(tasks[0])
        if isinstance(tasks, str):
            return tasks
        return ""

    def _lazy_media_for(self, episode: LazyEpisode) -> LazyLeRobotMedia:
        return self.lazy_media[episode.task_idx]

    def _episodes_jsonl_path(self) -> Path:
        if self._active_lazy_episode is None:
            raise RuntimeError("No active lazy episode for episodes.jsonl path")
        return self._active_lazy_episode.dataset_root / "meta" / "episodes.jsonl"

    def _t5_cache_file_path(self, episode_index: int) -> Path:
        if self._active_lazy_episode is None:
            raise RuntimeError("No active lazy episode for T5 cache path")
        return self._active_lazy_episode.dataset_root / self.t5_folder_name / f"episode_{episode_index:06d}.pt"

    def _t5_lock_file_path(self, episode_index: int) -> Path:
        return self._t5_cache_file_path(episode_index).with_suffix(".pt.lock")

    def _atomic_update_episodes_jsonl(self, episode_index: int, updates: Dict[str, Any]) -> None:
        path = self._episodes_jsonl_path()
        tmp = path.with_suffix(path.suffix + ".tmp")
        found = False
        with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if int(obj.get("episode_index", -1)) == int(episode_index):
                    obj.update(updates)
                    found = True
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if found:
            tmp.replace(path)
        else:
            tmp.unlink(missing_ok=True)

    def _ensure_t5_encoder(self):
        if self._t5_encoder is not None:
            return self._t5_encoder
        import multiprocessing

        current_process = multiprocessing.current_process()
        if current_process.name != "MainProcess":
            raise RuntimeError("T5 encoder initialization in DataLoader worker process is disabled")
        try:
            from bak.wan.modules.t5 import T5EncoderModel  # type: ignore
        except Exception:
            import sys

            bak_root = str((Path(__file__).resolve().parents[2] / "bak").resolve())
            if bak_root not in sys.path:
                sys.path.insert(0, bak_root)
            from wan.modules.t5 import T5EncoderModel  # type: ignore

        if not self.t5_wan_path:
            raise ValueError("enable_t5_fallback=True but t5_wan_path is not provided")
        ckpt = os.path.join(self.t5_wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
        tok = os.path.join(self.t5_wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
        dtype = torch.bfloat16 if self.t5_device.startswith("cuda") else torch.float32
        self._t5_encoder = T5EncoderModel(
            text_len=self.t5_text_len,
            dtype=dtype,
            device=self.t5_device,
            checkpoint_path=ckpt,
            tokenizer_path=tok,
        )
        return self._t5_encoder

    def _encode_and_cache_t5_embedding(self, episode_index: int, instruction: str) -> torch.Tensor:
        out_pt = self._t5_cache_file_path(episode_index)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        if out_pt.exists():
            emb = torch.load(out_pt, map_location="cpu", weights_only=True)
            return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)

        lock_path = self._t5_lock_file_path(episode_index)
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if out_pt.exists():
                    emb = torch.load(out_pt, map_location="cpu", weights_only=True)
                    return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                if time.time() - start > 600:
                    raise TimeoutError(f"Timeout waiting for T5 embedding lock: {lock_path}")
                time.sleep(0.2)
        try:
            encoder = self._ensure_t5_encoder()
            with torch.no_grad():
                t5_out = encoder([instruction], self.t5_device)
            emb = t5_out[0] if isinstance(t5_out, list) else t5_out
            if isinstance(emb, torch.Tensor) and emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            torch.save(emb.detach().cpu(), out_pt)
            self._atomic_update_episodes_jsonl(
                episode_index,
                {"t5_embedding_path": f"{self.t5_folder_name}/episode_{episode_index:06d}.pt"},
            )
            return emb
        finally:
            lock_path.unlink(missing_ok=True)

    def _load_language_embedding(self, item_cond: Mapping[str, Any], task_idx: int) -> torch.Tensor:
        all_embeddings = item_cond.get("language_embedding", None)
        if all_embeddings is None:
            all_embeddings = item_cond.get("observation.feature.language_embedding", None)
        if all_embeddings is None:
            if self._active_lazy_episode is None:
                raise RuntimeError("No active lazy episode for language embedding")
            ep_index = int(self._active_lazy_episode.episode_index)
            ep_meta = self._active_lazy_episode.episode_meta
            rel_path = ep_meta.get("t5_embedding_path", None)
            if rel_path is None:
                if not self.enable_t5_fallback:
                    raise KeyError("t5_embedding_path not found in lazy episode metadata")
                instr = item_cond.get("language_instruction", None) or item_cond.get("task", "")
                all_embeddings = self._encode_and_cache_t5_embedding(ep_index, str(instr))
            else:
                abs_path = Path(str(rel_path))
                if not abs_path.is_absolute():
                    abs_path = self._active_lazy_episode.dataset_root / abs_path
                all_embeddings = torch.load(abs_path, map_location="cpu", weights_only=True)
                if not isinstance(all_embeddings, torch.Tensor):
                    all_embeddings = torch.tensor(all_embeddings)
            if all_embeddings.ndim == 2:
                all_embeddings = all_embeddings.unsqueeze(0)
        if not isinstance(all_embeddings, torch.Tensor):
            all_embeddings = torch.tensor(all_embeddings)
        if all_embeddings.ndim == 2:
            all_embeddings = all_embeddings.unsqueeze(0)
        return all_embeddings[0].float()

    def _load_language_action_from_file(
        self,
        episode_index: int,
        task_idx: int,
        condition_frame_idx: int,
    ) -> Optional[str]:
        if self._active_lazy_episode is None:
            raise RuntimeError("No active lazy episode for language_action")
        ep_meta = self._active_lazy_episode.episode_meta
        declared_rel_path = ep_meta.get("language_action_path", None)
        rel_path = declared_rel_path
        if rel_path is None:
            rel_path = f"{self.language_action_dir_name}/episode_{episode_index:06d}.txt"
        path = Path(str(rel_path))
        if not path.is_absolute():
            path = self._active_lazy_episode.dataset_root / path
        if not path.exists() and declared_rel_path is None:
            return None
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"No language action lines in {path}")
        return lines[min(condition_frame_idx, len(lines) - 1)]

    def _column_names_namespace(self, columns: Mapping[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(column_names=list(columns.keys()))


class LazyLeRobotRoboCOINDataset(LazyLeRobotMixin, LeRobotRoboCOINDataset):
    def __init__(
        self,
        *args: Any,
        visual_keys: Optional[Sequence[str]] = None,
        language_action_dir_name: str = "language_action",
        normalize_actions: bool = False,
        use_language_action: bool = False,
        **kwargs: Any,
    ):
        if normalize_actions:
            raise ValueError("LazyLeRobotRoboCOINDataset does not support normalize_actions=True")
        self.use_language_action = bool(use_language_action)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self.visual_keys_override = tuple(visual_keys) if visual_keys is not None else None
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}
        self._visual_keys_cache: Dict[str, Tuple[str, ...]] = {}
        self._init_lazy_common(*args, **kwargs)

    def __getitem__(self, idx: int):
        episode = self._sample_lazy_episode()
        self._active_lazy_episode = episode
        columns = self._read_sample_columns(episode)
        if columns is None:
            return self.__getitem__(idx)
        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(episode.length)
        ds_media = self._lazy_media_for(episode)

        item_cond = self._row_at(columns, condition_frame_idx, episode)
        timestamps = self._timestamps_at(columns, [condition_frame_idx] + video_indices)
        first_frame, video_frames_sampled = self._load_visual_frames(ds_media, episode.episode_index, timestamps)

        if "observation.state" in item_cond:
            initial_state = torch.as_tensor(item_cond["observation.state"]).float()
            state_names = self._feature_names(ds_media, "observation.state")
        elif "action" in item_cond:
            initial_state = torch.as_tensor(item_cond["action"]).float()
            state_names = self._feature_names(ds_media, "action")
        elif "actions" in item_cond:
            initial_state = torch.as_tensor(item_cond["actions"]).float()
            state_names = self._feature_names(ds_media, "actions")
        else:
            raise KeyError("No state found in lazy RoboCOIN item")

        action_key = "action" if "action" in columns else "actions" if "actions" in columns else None
        if action_key is None:
            raise KeyError("No action column found in lazy RoboCOIN item")
        action_sequence = self._to_action_tensor(self._values_at(columns, action_key, action_indices))
        if action_sequence.ndim == 1:
            action_sequence = action_sequence.unsqueeze(0)
        action_names = self._feature_names(ds_media, action_key)

        language_embedding = self._load_language_embedding(item_cond, episode.task_idx)
        language_action = None
        if self.use_language_action:
            language_action = self._load_language_action(
                item_cond=item_cond,
                episode_index=episode.episode_index,
                task_idx=episode.task_idx,
                condition_frame_idx=condition_frame_idx,
            )

        vlm_tokens = None
        if self.vlm_processor:
            text_instr = item_cond.get("language_instruction", None) or item_cond.get("task", "")
            first_frame_pil = tensor_to_pil(first_frame)
            vlm_tokens = self._preprocess_vlm_inputs(
                text_instr,
                first_frame_pil,
                language_action,
            )

        return {
            "first_frame": first_frame,
            "video_frames": video_frames_sampled,
            "initial_state": initial_state,
            "state_names": state_names,
            "action_sequence": action_sequence.float(),
            "action_names": action_names,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_tokens,
        }


class LazyLeRobotInternDataDataset(LazyLeRobotMixin, LeRobotInternDataDataset):
    def __init__(
        self,
        *args: Any,
        language_action_dir_name: str = "language_action",
        use_language_action: bool = True,
        normalize_actions: bool = False,
        **kwargs: Any,
    ):
        if normalize_actions:
            raise ValueError("LazyLeRobotInternDataDataset does not support normalize_actions=True")
        self.use_language_action = bool(use_language_action)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}
        self._init_lazy_common(*args, **kwargs)

    def __getitem__(self, idx: int):
        episode = self._sample_lazy_episode()
        self._active_lazy_episode = episode
        columns = self._read_sample_columns(episode)
        if columns is None:
            return self.__getitem__(idx)
        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(episode.length)
        ds_media = self._lazy_media_for(episode)
        item_cond = self._row_at(columns, condition_frame_idx, episode)

        first_frame, video_frames_sampled = self._load_visual_frames_lazy(
            ds_media,
            columns,
            item_cond,
            [condition_frame_idx] + video_indices,
        )

        state_columns = {key: item_cond[key] for key in item_cond.keys()}
        initial_state, state_mask = build_interndata_canonical55(state_columns, prefix="states")
        if not state_mask.any():
            initial_state, state_mask = build_interndata_canonical55(state_columns, prefix="actions")
        initial_state = initial_state[0]
        state_mask = state_mask[0]

        action_batch = {key: self._values_at(columns, key, action_indices) for key in columns}
        action_sequence, action_mask = build_interndata_canonical55(action_batch, prefix="actions")
        if not action_mask.any():
            raise KeyError("No usable lazy InternData action columns found")

        language_embedding = self._load_language_embedding(item_cond, episode.task_idx)
        language_action = None
        if self.use_language_action:
            language_action = self._load_language_action_from_file(
                episode.episode_index,
                episode.task_idx,
                condition_frame_idx,
            )

        vlm_tokens = None
        if self.vlm_processor:
            text_instr = self._instruction_from_item(item_cond)
            first_frame_pil = tensor_to_pil(first_frame)
            if self.use_language_action:
                vlm_tokens = preprocess_vlm_messages_lap(
                    text_instr,
                    first_frame_pil,
                    self.vlm_processor,
                    language_action,
                    supervise_answer=True,
                )
            else:
                vlm_tokens = preprocess_vlm_messages(text_instr, first_frame_pil, self.vlm_processor)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames_sampled,
            "initial_state": initial_state,
            "state_mask": state_mask,
            "action_sequence": action_sequence,
            "action_mask": action_mask,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_tokens,
        }

    def _load_visual_frames_lazy(
        self,
        ds_media: LazyLeRobotMedia,
        columns: Mapping[str, List[Any]],
        item_cond: Mapping[str, Any],
        indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestamps = self._timestamps_at(columns, indices)
        visual_key = self._select_visual_key(ds_media, self._column_names_namespace(columns))
        video_path = Path(ds_media.root) / ds_media.meta.get_video_file_path(
            int(item_cond.get("episode_index", self._active_lazy_episode.episode_index)),
            visual_key,
        )
        frames = decode_video_frames(video_path, timestamps, ds_media.tolerance_s, ds_media.video_backend).squeeze(0)
        first_frame = self._resize_frame_chw(frames[0].float(), self.video_size)
        video_frames = torch.stack(
            [self._resize_frame_chw(frames[i].float(), self.video_size) for i in range(1, frames.shape[0])],
            dim=0,
        )
        return first_frame, video_frames


class LazyLeRobotAgiBotDataset(LazyLeRobotMixin, LeRobotAgiBotDataset):
    def __init__(
        self,
        *args: Any,
        visual_keys: Optional[Sequence[str]] = None,
        language_action_dir_name: str = "language_action",
        normalize_actions: bool = False,
        stats_path: Optional[str] = None,
        stats_key: str = "agibot",
        use_language_action: bool = False,
        **kwargs: Any,
    ):
        self._feature_fields_cache: Dict[Tuple[str, str], Dict[str, Tuple[int, ...]]] = {}
        self.normalize_actions = bool(normalize_actions)
        self.stats_key = str(stats_key)
        self.agibot_state_min = None
        self.agibot_state_max = None
        self.agibot_action_min = None
        self.agibot_action_max = None
        if self.normalize_actions:
            if stats_path is None:
                stats_path = str((Path(__file__).resolve().parent.parent / "utils" / "stat.json"))
            self.agibot_state_min, self.agibot_state_max = load_normalization_stats(stats_path, self.stats_key, "state")
            self.agibot_action_min, self.agibot_action_max = load_normalization_stats(stats_path, self.stats_key, "action")
        self.use_language_action = bool(use_language_action)
        self.language_action_dir_name = str(language_action_dir_name).strip() or "language_action"
        self._language_action_cache: Dict[Tuple[int, int], List[str]] = {}
        self.visual_keys = tuple(visual_keys) if visual_keys is not None else self.DEFAULT_VISUAL_KEYS
        self._init_lazy_common(*args, **kwargs)

        missing_by_repo = {
            repo_id: [key for key in self.visual_keys if key not in media.features]
            for repo_id, media in zip(self.repo_ids, self.lazy_media)
        }
        missing_by_repo = {repo_id: missing for repo_id, missing in missing_by_repo.items() if missing}
        if missing_by_repo:
            raise ValueError(f"AgiBot visual keys not found in lazy dataset features: {missing_by_repo}")

    def __getitem__(self, idx: int):
        episode = self._sample_lazy_episode()
        self._active_lazy_episode = episode
        columns = self._read_sample_columns(episode)
        if columns is None:
            return self.__getitem__(idx)
        condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(episode.length)
        ds_media = self._lazy_media_for(episode)

        item_cond = self._row_at(columns, condition_frame_idx, episode)
        timestamps = self._timestamps_at(columns, [condition_frame_idx] + video_indices)
        first_frame, video_frames_sampled = self._load_stitched_video_frames(
            ds_media,
            episode.episode_index,
            timestamps,
        )

        if "observation.state" not in item_cond:
            raise KeyError("observation.state not found in lazy AgiBot item")
        raw_state = torch.as_tensor(item_cond["observation.state"]).float()
        initial_state, state_mask = self._select_state(raw_state, ds_media)

        action_key = "action" if "action" in columns else "actions" if "actions" in columns else None
        if action_key is None:
            raise KeyError("No action column found in lazy AgiBot item")
        action_sequence, action_mask = self._select_action_sequence(
            self._values_at(columns, action_key, action_indices),
            item_cond[action_key],
            ds_media,
        )

        language_embedding = self._load_language_embedding(item_cond, episode.task_idx)
        language_action = None
        if self.use_language_action:
            language_action = self._load_language_action(
                item_cond=item_cond,
                episode_index=episode.episode_index,
                task_idx=episode.task_idx,
                condition_frame_idx=condition_frame_idx,
            )

        vlm_tokens = None
        if self.vlm_processor:
            text_instr = item_cond.get("language_instruction", None) or item_cond.get("task", "")
            first_frame_pil = tensor_to_pil(first_frame)
            if self.use_language_action:
                vlm_tokens = preprocess_vlm_messages_lap(
                    text_instr,
                    first_frame_pil,
                    self.vlm_processor,
                    language_action,
                    supervise_answer=True,
                )
            else:
                vlm_tokens = preprocess_vlm_messages(text_instr, first_frame_pil, self.vlm_processor)

        return {
            "first_frame": first_frame,
            "video_frames": video_frames_sampled,
            "initial_state": initial_state,
            "action_sequence": action_sequence,
            "state_mask": state_mask,
            "action_mask": action_mask,
            "language_embedding": language_embedding,
            "vlm_inputs": vlm_tokens,
        }
