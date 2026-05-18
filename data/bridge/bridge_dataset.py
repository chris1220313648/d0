# Bridge Dataset Loader for Motus
# Uses epos as action supervision and optionally language_action for LAP-style VLM supervision.

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
import torch.utils.data as data
from transformers import AutoProcessor

from utils.vlm_utils import (
    preprocess_vlm_messages,
    preprocess_vlm_messages_lap,
    append_setup_control_suffix,
)
from data.utils.image_utils import tensor_to_pil, load_video_frames, get_video_frame_count
from data.utils.norm import normalize_actions, load_normalization_stats


logger = logging.getLogger(__name__)


class BridgeDataset(data.Dataset):
    """
    Dataset for Bridge data with task-level organization.

    Expected data structure:
      <dataset_dir>/
        train|test/
          <task_name>/
            videos/*.mp4
            qpos/*.pt
            epos/*.pt
            umt5_wan/*.pt
            instructions/*.txt
            language_action/*.txt  (optional)
    """

    def __init__(
        self,
        dataset_dir: str = "/data/user/wsong890/user68/cjy/Motus/data/bridge/bridge_dataset",
        data_mode: str = "train",  # train, test, both
        task_mode: str = "multi",  # single, multi
        task_name: Optional[str] = None,
        global_downsample_rate: int = 1,
        video_action_freq_ratio: int = 5,
        num_video_frames: int = 3,
        video_size: Tuple[int, int] = (320, 384),
        max_episodes: Optional[int] = None,
        val: bool = False,
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
        use_language_action: bool = False,
        normalize_actions: bool = False,
        stats_path: Optional[str] = None,
        stats_key: Optional[str] = None,
        state_stats_signal: str = "qpos",
        action_stats_signal: str = "epos",
        enable_setup_control_suffix: bool = False,
        setup_text: str = "bimanual yam robotic arms in molmoact2",
        randomized_limit_per_task: Optional[int] = None,  # kept for config compatibility; unused for bridge
        **kwargs,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.data_mode = data_mode
        self.task_mode = task_mode
        self.task_name = task_name
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        self.action_chunk_size = num_video_frames * video_action_freq_ratio
        self.video_size = video_size
        self.max_episodes = max_episodes
        self.val = val
        self.image_aug = image_aug and not val
        self.use_language_action = use_language_action
        self.normalize_actions = bool(normalize_actions)
        self.stats_key = stats_key
        self.state_stats_signal = self._canonical_signal(state_stats_signal)
        self.action_stats_signal = self._canonical_signal(action_stats_signal)
        self.enable_setup_control_suffix = bool(enable_setup_control_suffix)
        self.setup_text = str(setup_text)

        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name parameter")
        if data_mode not in ["train", "test", "both"]:
            raise ValueError(f"data_mode must be train/test/both, got {data_mode}")
        if task_mode not in ["single", "multi"]:
            raise ValueError(f"task_mode must be single/multi, got {task_mode}")
        if self.action_stats_signal not in ["epos", "qpos"]:
            raise ValueError(
                f"action_stats_signal must be one of ['epos', 'qpos', 'pos'], got {action_stats_signal}"
            )

        # Optional action normalization
        self.action_min = None
        self.action_max = None
        self.state_min = None
        self.state_max = None
        if self.normalize_actions:
            if stats_path is None:
                stats_path = str((Path(__file__).resolve().parent.parent / "utils" / "stat.json"))
            dataset_key = self.stats_key or "bridge"

            self.state_min, self.state_max = load_normalization_stats(
                stats_path, dataset_key, self.state_stats_signal
            )
            self.action_min, self.action_max = load_normalization_stats(
                stats_path, dataset_key, self.action_stats_signal
            )
            if (
                self.state_min is None
                or self.state_max is None
                or self.action_min is None
                or self.action_max is None
            ):
                raise ValueError(
                    f"Failed to load normalization stats for dataset={dataset_key} from {stats_path}"
                )

        self.vlm_processor = None
        if vlm_checkpoint_path is not None:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"VLM processor loaded from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor from {vlm_checkpoint_path}: {e}")
                logger.warning("VLM processing will be disabled for this dataset instance")

        if task_mode == "single":
            self.episode_files: List[Dict[str, Any]] = []
        else:
            self.task_to_episodes: Dict[str, List[Dict[str, Any]]] = {}
            self.task_weights: Dict[str, float] = {}

        self.total_episodes = 0
        self._load_episodes()

        logger.info("Bridge dataset initialized:")
        logger.info(f"  Dataset dir: {self.dataset_dir}")
        logger.info(f"  Data mode: {self.data_mode}")
        logger.info(f"  Task mode: {self.task_mode}")
        if self.task_name:
            logger.info(f"  Task name: {self.task_name}")
        logger.info(f"  Action chunk size: {self.action_chunk_size}")
        logger.info(f"  Total episodes: {self.total_episodes}")
        logger.info(f"  Use language action: {self.use_language_action}")
        logger.info(f"  Normalize actions: {self.normalize_actions}")
        logger.info(f"  Enable setup/control suffix: {self.enable_setup_control_suffix}")
        if self.normalize_actions:
            logger.info(
                f"  Stats key/path/signals: {self.stats_key or 'bridge'} / {stats_path} / "
                f"state={self.state_stats_signal}, action={self.action_stats_signal}"
            )

    @staticmethod
    def _canonical_signal(signal: str) -> str:
        if signal is None:
            return "epos"
        sig = str(signal).strip().lower()
        if sig == "pos":
            return "qpos"
        return sig

    def _scan_task_folder(self, task_path: Path, split_name: str) -> List[Dict[str, Any]]:
        videos_dir = task_path / "videos"
        qpos_dir = task_path / "qpos"
        epos_dir = task_path / "epos"
        umt5_dir = task_path / "umt5_wan"
        instructions_dir = task_path / "instructions"
        lang_action_dir = task_path / "language_action"

        required = [videos_dir, qpos_dir, epos_dir, umt5_dir, instructions_dir]
        if not all(p.exists() for p in required):
            logger.warning(f"Missing required dirs in {task_path}, skip this task.")
            return []
        if self.use_language_action and not lang_action_dir.exists():
            logger.warning(f"Missing language_action dir in {task_path}, skip this task.")
            return []

        episodes: List[Dict[str, Any]] = []
        for epos_file in sorted(epos_dir.glob("*.pt"), key=lambda p: p.name):
            episode_name = epos_file.stem
            video_file = videos_dir / f"{episode_name}.mp4"
            qpos_file = qpos_dir / f"{episode_name}.pt"
            umt5_file = umt5_dir / f"{episode_name}.pt"
            instruction_file = instructions_dir / f"{episode_name}.txt"
            lang_action_file = lang_action_dir / f"{episode_name}.txt"

            if not (video_file.exists() and qpos_file.exists() and umt5_file.exists() and instruction_file.exists()):
                continue
            if self.use_language_action and not lang_action_file.exists():
                continue

            episode_data = {
                "episode_name": episode_name,
                "task_name": task_path.name,
                "split_name": split_name,
                "epos_path": str(epos_file),
                "qpos_path": str(qpos_file),
                "video_path": str(video_file),
                "lang_path": str(umt5_file),
                "instruction_path": str(instruction_file),
            }
            if self.use_language_action:
                episode_data["lang_action_path"] = str(lang_action_file)
            episodes.append(episode_data)

        # logger.info(f"Task {task_path.name} ({split_name}): found {len(episodes)} valid episodes")
        return episodes

    def _load_episodes(self) -> None:
        splits = ["train", "test"] if self.data_mode == "both" else [self.data_mode]
        all_episodes: List[Dict[str, Any]] = []

        for split_name in splits:
            split_dir = self.dataset_dir / split_name
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue

            if self.task_mode == "single":
                task_dir = split_dir / str(self.task_name)
                if task_dir.exists():
                    all_episodes.extend(self._scan_task_folder(task_dir, split_name))
                else:
                    logger.warning(f"Task directory not found: {task_dir}")
            else:
                task_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
                for task_dir in task_dirs:
                    episodes = self._scan_task_folder(task_dir, split_name)
                    tname = task_dir.name
                    if tname not in self.task_to_episodes:
                        self.task_to_episodes[tname] = []
                    self.task_to_episodes[tname].extend(episodes)

        if self.task_mode == "single":
            self.episode_files = all_episodes
            if self.max_episodes is not None:
                self.episode_files = self.episode_files[: self.max_episodes]
            self.total_episodes = len(self.episode_files)
            if self.total_episodes == 0:
                raise ValueError(f"No valid episodes found for task {self.task_name}")
        else:
            if not self.task_to_episodes:
                raise ValueError("No valid episodes found for any task")
            # if self.max_episodes is not None:
            #     total: List[Dict[str, Any]] = []
            #     for tname in list(self.task_to_episodes.keys()):
            #         total.extend(self.task_to_episodes[tname])
            #     total = total[: self.max_episodes]
            #     self.task_to_episodes = {}
            #     for ep in total:
            #         tname = ep["task_name"]
            #         if tname not in self.task_to_episodes:
            #             self.task_to_episodes[tname] = []
            #         self.task_to_episodes[tname].append(ep)

            num_tasks = len(self.task_to_episodes)
            #计算一共有多少个episode
            # total_episodes = sum(len(v) for v in self.task_to_episodes.values())
            self.total_episodes = sum(len(v) for v in self.task_to_episodes.values())
            # for tname in self.task_to_episodes:
            #     self.task_weights[tname] = 1.0 / num_tasks
            for tname,v in self.task_to_episodes.items():
                self.task_weights[tname] =len(v)/self.total_episodes
            logger.info(f"Multi-task Bridge dataset with {num_tasks} tasks.")
            for tname, episodes in self.task_to_episodes.items():
                logger.info(f"  {tname}: {len(episodes)} episodes")

    def _load_language_embedding(self, lang_path: str) -> Tuple[torch.Tensor, int]:
        embedding_data = torch.load(lang_path, map_location="cpu")
        if isinstance(embedding_data, list):
            selected_idx = random.randint(0, len(embedding_data) - 1)
            embeddings = embedding_data[selected_idx]
        elif isinstance(embedding_data, torch.Tensor):
            selected_idx = 0
            embeddings = embedding_data
        else:
            raise TypeError(f"Unsupported embedding format at {lang_path}: {type(embedding_data)}")

        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        return embeddings, selected_idx

    def _load_text_instruction(self, instruction_path: str) -> str:
        with open(instruction_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            raise ValueError(f"No instruction text found in {instruction_path}")
        return lines[0]

    def _load_language_action(self, lang_action_path: str, condition_frame_idx: int) -> str:
        with open(lang_action_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            raise ValueError(f"No language action lines in {lang_action_path}")
        if condition_frame_idx >= len(lines):
            return lines[-1]
        return lines[condition_frame_idx]

    def _calculate_sampling_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        max_condition_idx = total_frames - physical_chunk_size - 1
        if max_condition_idx < 0:
            condition_frame_idx = 0
        else:
            condition_frame_idx = random.randint(0, max_condition_idx)

        action_indices: List[int] = []
        for i in range(self.action_chunk_size):
            idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(idx, total_frames - 1))

        video_indices: List[int] = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])
        return condition_frame_idx, video_indices, action_indices

    def _load_initial_state_and_actions(
        self,
        qpos_path: str,
        epos_path: str,
        condition_frame_idx: int,
        action_indices: List[int],
        action_stats_signal: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qpos_data = torch.load(qpos_path, map_location="cpu").float()
        epos_data = torch.load(epos_path, map_location="cpu").float()

        if qpos_data.ndim != 2:
            raise ValueError(f"qpos tensor must be [T,D], got {tuple(qpos_data.shape)} at {qpos_path}")
        if epos_data.ndim != 2:
            raise ValueError(f"epos tensor must be [T,D], got {tuple(epos_data.shape)} at {epos_path}")

        # Initial state uses absolute qpos.
        # action_sequence source is controlled by action_stats_signal:
        #   epos -> delta action
        #   qpos/pos -> absolute joint position
        if condition_frame_idx >= len(qpos_data):
            condition_frame_idx = len(qpos_data) - 1
        initial_state = qpos_data[condition_frame_idx].float()

        action_signal = self._canonical_signal(action_stats_signal or self.action_stats_signal)
        if action_signal == "epos":
            action_source = epos_data
        elif action_signal == "qpos":
            action_source = qpos_data
        else:
            raise ValueError(f"Unsupported action signal: {action_signal}")

        actions: List[torch.Tensor] = []
        for idx in action_indices:
            if idx >= len(action_source):
                idx = len(action_source) - 1
            actions.append(action_source[idx])
        action_sequence = torch.stack(actions).float()

        if self.normalize_actions:
            action_sequence = normalize_actions(action_sequence, self.action_min, self.action_max)
            initial_state = normalize_actions(initial_state.unsqueeze(0), self.state_min, self.state_max).squeeze(0)

        return initial_state, action_sequence

    def __len__(self) -> int:
        return self.total_episodes * 100

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        max_attempts = 8
        for _ in range(max_attempts):
            if self.task_mode == "single":
                if not self.episode_files:
                    continue
                episode_data = random.choice(self.episode_files)
            else:
                if not self.task_to_episodes:
                    continue
                task_name = random.choices(
                    list(self.task_weights.keys()),
                    weights=list(self.task_weights.values()),
                    k=1,
                )[0]
                task_episodes = self.task_to_episodes.get(task_name, [])
                if not task_episodes:
                    continue
                episode_data = random.choice(task_episodes)

            try:
                total_frames = get_video_frame_count(episode_data["video_path"])
                if total_frames < 2:
                    continue

                condition_frame_idx, video_indices, action_indices = self._calculate_sampling_indices(total_frames)
                first_frame = load_video_frames(episode_data["video_path"], [condition_frame_idx], self.video_size)
                video_frames = load_video_frames(episode_data["video_path"], video_indices, self.video_size)
                initial_state, action_sequence = self._load_initial_state_and_actions(
                    qpos_path=episode_data["qpos_path"],
                    epos_path=episode_data["epos_path"],
                    condition_frame_idx=condition_frame_idx,
                    action_indices=action_indices,
                    action_stats_signal=self.action_stats_signal,
                )
                language_embedding, _ = self._load_language_embedding(episode_data["lang_path"])
                text_instruction = self._load_text_instruction(episode_data["instruction_path"])
                final_instruction = append_setup_control_suffix(
                    text_instruction=text_instruction,
                    enable_setup_control_suffix=self.enable_setup_control_suffix,
                    setup_text=self.setup_text,
                    action_signal=self.action_stats_signal,
                )

                if self.use_language_action:
                    language_action = self._load_language_action(episode_data["lang_action_path"], condition_frame_idx)
                else:
                    language_action = None

                vlm_inputs = None
                if self.vlm_processor is not None:
                    first_frame_pil = tensor_to_pil(first_frame.squeeze(0))
                    if self.use_language_action:
                        vlm_inputs = preprocess_vlm_messages_lap(
                            final_instruction,
                            first_frame_pil,
                            self.vlm_processor,
                            language_action,
                            supervise_answer=True,
                        )
                    else:
                        vlm_inputs = preprocess_vlm_messages(final_instruction, first_frame_pil, self.vlm_processor)

                return {
                    "first_frame": first_frame.squeeze(0),
                    "video_frames": video_frames,
                    "initial_state": initial_state,
                    "action_sequence": action_sequence,
                    "language_embedding": language_embedding,
                    "vlm_inputs": vlm_inputs,
                }
            except Exception as e:
                logger.warning(f"Retry due to sample error ({episode_data.get('episode_name', '?')}): {e}")
                continue

        return None
