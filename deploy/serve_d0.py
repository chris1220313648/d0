#!/usr/bin/env python3
"""Serve the Piper D0 policy over the OpenPI-compatible WebSocket protocol."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
import yaml

from deploy import d0_protocol
from deploy.d0_policy import D0Policy
from deploy.d0_preprocessing import task_metadata


logger = logging.getLogger(__name__)


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_deployment(
    checkpoint: str,
    *,
    config_override: str | None = None,
    stats_override: str | None = None,
    steps_override: int | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint).expanduser().resolve()
    manifest_path = checkpoint_dir / "deployment.yaml"
    if not manifest_path.is_file():
        if config_override is None or stats_override is None:
            raise FileNotFoundError(
                f"Missing {manifest_path}; legacy checkpoints require both --config and --stats"
            )
        manifest: dict[str, Any] = {}
        base = Path.cwd()
    else:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        if int(manifest.get("version", 0)) != 1:
            raise ValueError(f"Unsupported D0 deployment manifest version: {manifest.get('version')}")
        base = manifest_path.parent
    config_value = config_override or manifest.get("config")
    stats_value = stats_override or manifest.get("stats")
    if not config_value or not stats_value:
        raise ValueError("D0 deployment requires config and stats paths")
    return {
        **manifest,
        "checkpoint": str(checkpoint_dir),
        "config": _resolve_path(str(config_value), base),
        "stats": _resolve_path(str(stats_value), base),
        "num_inference_steps": int(
            steps_override if steps_override is not None else manifest.get("num_inference_steps", 4)
        ),
    }


class D0WebSocketServer:
    def __init__(self, policy: D0Policy):
        self.policy = policy
        common = policy.config["common"]
        horizon = int(common["num_video_frames"]) * int(common["video_action_freq_ratio"])
        flow_mode = str(policy.config.get("model", {}).get("flow_source", {}).get("mode", "gaussian"))
        self.metadata = {
            "protocol_version": 2,
            "model_name": policy.model_name,
            "state_schema": "bimanual_joint",
            "state_dim": int(common["state_dim"]),
            "action_schema": "bimanual_delta_ee_rpy",
            "action_dim": int(common["action_dim"]),
            "action_horizon": horizon,
            "history_action_required": flow_mode == "history",
            "history_action_horizon": horizon if flow_mode == "history" else 0,
            "control_hz": 30,
            "rotation_unit": "rad",
            "gripper_range": [0.0, 1.0],
            "rtc": {"supported": False},
            "supported_tasks": task_metadata(policy.task_specs),
        }

    async def healthz(self, request: web.Request) -> web.Response:
        return web.Response(text="OK\n")

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(compress=False, max_msg_size=0)
        await ws.prepare(request)
        peer = request.remote
        logger.info("Connection from %s opened", peer)
        packer = d0_protocol.Packer()
        await ws.send_bytes(packer.pack(self.metadata))
        previous_total_ms = None
        async for message in ws:
            if message.type != WSMsgType.BINARY:
                if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break
                await ws.send_str("D0 server expects binary MessagePack requests")
                continue
            started = time.perf_counter()
            try:
                observation = d0_protocol.unpackb(message.data)
                response = await asyncio.to_thread(self.policy.infer, observation)
                response["server_timing"] = {
                    "infer_ms": response.get("policy_timing", {}).get("infer_ms", 0.0),
                }
                if previous_total_ms is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_ms
                await ws.send_bytes(packer.pack(response))
                previous_total_ms = (time.perf_counter() - started) * 1000.0
            except Exception:
                logger.exception("D0 inference failed for %s", peer)
                await ws.send_str(traceback.format_exc())
                await ws.close(code=1011, message=b"Internal D0 inference error")
                break
        logger.info("Connection from %s closed", peer)
        return ws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    deployment = load_deployment(
        args.checkpoint,
        config_override=args.config,
        stats_override=args.stats,
        steps_override=args.num_inference_steps,
    )
    policy = D0Policy(
        checkpoint_dir=deployment["checkpoint"],
        config_path=deployment["config"],
        stats_path=deployment["stats"],
        device=args.device,
        num_inference_steps=deployment["num_inference_steps"],
        tasks=deployment.get("tasks"),
        model_name=deployment.get("model_name"),
    )
    warmup = policy.warmup(args.warmup_runs)
    if warmup["runs"]:
        logger.info(
            "D0 warmup complete: task_id=%s timings_ms=%s",
            warmup["task_id"],
            [round(value, 2) for value in warmup["timings_ms"]],
        )
    server = D0WebSocketServer(policy)
    app = web.Application(client_max_size=0)
    app.router.add_get("/healthz", server.healthz)
    app.router.add_get("/", server.websocket)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
