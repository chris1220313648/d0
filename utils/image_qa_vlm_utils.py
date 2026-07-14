"""Utilities for preserving Image-QA supervision in Qwen-VL inputs."""

from __future__ import annotations

from typing import Iterable

from utils.vlm_utils import preprocess_vlm_messages_sft


def preprocess_image_qa_messages(
    question: str,
    answer: str,
    image_pils: Iterable,
    processor,
    max_length: int | None = None,
):
    """Build supervised Qwen-VL inputs for one image-QA sample."""
    return preprocess_vlm_messages_sft(
        question=question,
        answer=answer,
        image_pils=list(image_pils),
        processor=processor,
        max_length=max_length,
    )
