# VLM Utilities
# Simple functions for VLM data preprocessing

from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Tuple
import logging
import random
import torch

logger = logging.getLogger(__name__)

def append_setup_control_suffix(
    text_instruction: str,
    enable_setup_control_suffix: bool = False,
    setup_text: str = "single-arm robot with gripper",
    action_signal: str = "epos",
) -> str:
    """
    Append setup/control strings to the end of instruction text for VLM input only.
    """
    base_text = "" if text_instruction is None else str(text_instruction)
    if not enable_setup_control_suffix:
        return base_text

    signal = str(action_signal or "epos").strip().lower()
    if signal == "pos":
        signal = "qpos"

    if signal == "epos":
        control_text = "delta end-effector pose control"
    elif signal == "qpos":
        control_text = "absolute joint angle control"
    else:
        control_text = "absolute joint angle control"
        logger.warning(
            "Unknown action_signal '%s' for setup/control suffix, fallback to '%s'",
            action_signal,
            control_text,
        )

    setup_clean = str(setup_text or "single-arm robot with gripper").strip()
    suffix = (
        f"<setup_start>{setup_clean}<setup_end>, and "
        f"<control_start>{control_text}<control_end>."
    )
    sep = "" if not base_text or base_text.endswith((" ", "\n", "\t")) else " "
    return f"{base_text}{sep}{suffix}"


def preprocess_vlm_messages(text_instruction: str, image_pil, processor):
    """
    Complete VLM preprocessing - create messages, process vision, and get final inputs.
    
    Args:
        text_instruction: Robot task instruction
        image_pil: PIL Image object
        processor: VLM processor (AutoProcessor)
        
    Returns:
        VLM inputs ready for model forward
    """
    # Create VLM messages format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": text_instruction}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Get final processor inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs




def preprocess_vlm_messages_lap(
    text_instruction: str,
    image_pil,
    processor,
    language_action: str | None = None,
    supervise_answer: bool = False,
    enable_ik_language_action_sampling: bool = False,
    ik_language_action_sampling_rate: float = 0.2,
    final_frame_pil=None,
):
    # 1) 问题文本（你可以自定义）

    # 推理模式：只有问题
    if not supervise_answer or language_action is None:
        user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": text_instruction},
        ],
        }
        messages = [user_msg]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    # 训练模式：问题 + 答案（assistant）
    use_ik_question = (
        enable_ik_language_action_sampling
        and ik_language_action_sampling_rate > 0.0
        and random.random() < ik_language_action_sampling_rate
    )
    if use_ik_question:
        question = "predict the robot's action between two images in the prediction"
    else:
        question = f"Task：{text_instruction}\nPlease provide a language description of the next action."
    user_content = [{"type": "image", "image": image_pil}]
    if use_ik_question and final_frame_pil is not None:
        # print("use_ik_question")
        # IK-style sample: provide both current frame and action-horizon final frame.
        user_content.append({"type": "image", "image": final_frame_pil})
    user_content.append({"type": "text", "text": question})
    user_msg = {
        "role": "user",
        "content": user_content,
        }
    assistant_msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": language_action}],
    }
    full_messages = [user_msg, assistant_msg]

    # 全序列（含答案）
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    # 前缀（不含答案，用于确定答案起点）
    prompt_text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info([user_msg])

    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 2) 只监督答案部分（CE labels）
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100

    full_inputs["labels"] = labels
    full_inputs["answer_start"] = torch.tensor([prompt_len], dtype=torch.long)
    # print("full_inputs.input_ids", full_inputs["input_ids"])
    # print("prompt_inputs.input_ids", prompt_inputs["input_ids"])
    # print("full_inputs.attention_mask", full_inputs["attention_mask"])
    # print("prompt_inputs.attention_mask", prompt_inputs["attention_mask"])
    # print("full_inputs.labels", full_inputs["labels"])
    # print("prompt_inputs.labels", prompt_inputs["labels"])
    # print("full_inputs.answer_start", full_inputs["answer_start"])
    # print("prompt_inputs.answer_start", prompt_inputs["answer_start"])
    return full_inputs


def preprocess_vlm_messages_sft(
    question: str,
    answer: str,
    image_pils,
    processor,
    max_length: int | None = None,
):
    """Build supervised Qwen-VL inputs for image/text SFT.

    User image(s) and question are prompt tokens. Only assistant answer tokens
    receive CE labels.
    """
    images = list(image_pils if isinstance(image_pils, (list, tuple)) else [image_pils])
    if not images:
        raise ValueError("preprocess_vlm_messages_sft requires at least one image")

    clean_question = (question or "").replace("<image>", "").strip()
    clean_answer = (answer or "").strip()
    if not clean_question:
        raise ValueError("empty SFT question")
    if not clean_answer:
        raise ValueError("empty SFT answer")

    user_content = [{"type": "image", "image": image} for image in images]
    user_content.append({"type": "text", "text": clean_question})
    user_msg = {"role": "user", "content": user_content}
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": clean_answer}]}

    full_messages = [user_msg, assistant_msg]
    prompt_messages = [user_msg]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt_messages)

    processor_kwargs = {
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
    }
    if max_length is not None and int(max_length) > 0:
        processor_kwargs["max_length"] = int(max_length)
        processor_kwargs["truncation"] = True

    full_inputs = processor(text=[full_text], **processor_kwargs)
    prompt_inputs = processor(text=[prompt_text], **processor_kwargs)

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    full_inputs["answer_start"] = torch.tensor([prompt_len], dtype=torch.long)
    return full_inputs
