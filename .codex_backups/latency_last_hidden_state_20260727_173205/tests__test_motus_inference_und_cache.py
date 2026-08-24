from types import SimpleNamespace

import torch

from models.motus import Motus


class _FakeVideoModel:
    precision = torch.float32

    def encode_video(self, first_frame):
        batch = first_frame.shape[0]
        return torch.zeros((batch, 2, 1, 2, 2), dtype=first_frame.dtype)

    def decode_video(self, video_latent):
        return video_latent


class _FakeUndModule:
    def __init__(self):
        self.extract_calls = 0

    def extract_und_features(self, vlm_inputs):
        self.extract_calls += 1
        return torch.zeros((1, 3, 4)), None

    def process_ffn(self, und_tokens, layer_idx):
        return und_tokens + 10


class _FakeVideoModule:
    def __init__(self):
        self.und_inputs = []

    def preprocess_t5_embeddings(self, language_embeddings):
        return language_embeddings

    def prepare_input(self, video_latent):
        return video_latent

    def get_time_embedding(self, timestep, seq_len):
        return timestep, None

    def compute_adaln_modulation(self, params, layer_idx):
        return None

    def process_joint_attention(
        self,
        video_tokens,
        action_tokens,
        video_modulation,
        action_modulation,
        layer_idx,
        action_block,
        und_tokens,
        und_block,
        und_k_lens=None,
    ):
        self.und_inputs.append(und_tokens.clone())
        return video_tokens, action_tokens, und_tokens + 1

    def process_cross_attention(self, video_tokens, params, layer_idx, context):
        return video_tokens

    def process_ffn(self, video_tokens, modulation, layer_idx):
        return video_tokens

    def apply_output_head(self, video_tokens, time_embedding):
        return torch.zeros_like(video_tokens)


class _FakeActionExpert:
    def __init__(self):
        self.config = SimpleNamespace(num_registers=1)
        self.registers = torch.zeros((1, 1, 14))
        self.blocks = [object()]

    def input_encoder(self, state_tokens, action_latent, registers):
        batch = action_latent.shape[0]
        return torch.zeros((batch, 50, 14), dtype=action_latent.dtype)

    def decoder(self, action_tokens, time_embedding):
        return torch.zeros_like(action_tokens)


class _FakeActionModule:
    def get_time_embedding(self, timestep, seq_len):
        return timestep, None

    def compute_adaln_modulation(self, params, layer_idx):
        return None

    def process_ffn(self, action_tokens, modulation, layer_idx):
        return action_tokens


def test_euler_inference_extracts_vlm_once_and_resets_tokens_each_step():
    model = object.__new__(Motus)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.config = SimpleNamespace(
        num_video_frames=8,
        flow_source_video_mode="gaussian",
        action_chunk_size=48,
        action_dim=14,
        flow_source_mode="history",
        flow_source_action_noise_std=0.0,
        num_layers=1,
    )
    model.video_model = _FakeVideoModel()
    model.und_module = _FakeUndModule()
    model.video_module = _FakeVideoModule()
    model.action_expert = _FakeActionExpert()
    model.action_module = _FakeActionModule()
    model.und_expert = SimpleNamespace(blocks=[object()])

    _, actions = model.inference_step(
        first_frame=torch.zeros((1, 3, 4, 4)),
        state=torch.zeros((1, 14)),
        num_inference_steps=4,
        language_embeddings=[torch.zeros((2, 4))],
        vlm_inputs=[{}],
        action_source=torch.zeros((1, 48, 14)),
    )

    assert model.und_module.extract_calls == 1
    assert len(model.video_module.und_inputs) == 4
    for und_input in model.video_module.und_inputs:
        torch.testing.assert_close(und_input, torch.zeros_like(und_input))
    assert actions.shape == (1, 48, 14)
