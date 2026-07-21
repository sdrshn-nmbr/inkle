# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate==1.14.0",
#   "httpx==0.28.1",
#   "huggingface-hub>=1.5,<2",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
#   "scipy==1.17.0",
#   "tokenizers==0.22.2",
#   "torch==2.12.0",
#   "transformers==5.14.0",
# ]
# ///

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from accelerate import init_empty_weights
from checkpoint_io import NVFP4_REPOSITORY, NVFP4_REVISION, HuggingFaceSafetensorsRepository
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoConfig
from transformers.models.inkling.modeling_inkling import InklingDecoderLayer, InklingExperts

HIDDEN_SIZE = 6144
ROUTED_EXPERTS = 256
LOGITS_MUP_WIDTH_MULTIPLIER = 24.0
UNPADDED_VOCABULARY_SIZE = 200058
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def to_torch(array: np.ndarray, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32)).to(dtype=dtype)


def deinterleave_gate_up(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-2] % 2:
        raise ValueError(f"INKLING_INVALID_GATE_UP_SHAPE shape={tuple(weight.shape)}")
    return torch.cat((weight[..., 0::2, :], weight[..., 1::2, :]), dim=-2)


def deinterleave_gate_up_numpy(weight: np.ndarray) -> np.ndarray:
    if weight.shape[-2] % 2:
        raise ValueError(f"INKLING_INVALID_GATE_UP_SHAPE shape={weight.shape}")
    return np.concatenate((weight[..., 0::2, :], weight[..., 1::2, :]), axis=-2)


def decode_nvfp4(
    packed_weight: np.ndarray,
    block_scale: np.ndarray,
    global_scale: np.ndarray,
) -> torch.Tensor:
    packed = torch.from_numpy(packed_weight)
    codes = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    values = E2M1_VALUES[codes.long()]
    blocked = values.reshape(*values.shape[:-1], -1, 16)
    scales = torch.from_numpy(block_scale.astype(np.float32))[..., None]
    decoded = blocked * scales * float(np.asarray(global_scale, dtype=np.float32))
    return decoded.reshape(values.shape).to(torch.bfloat16)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float32).ravel()
    candidate = candidate.astype(np.float32).ravel()
    difference = candidate - reference
    return {
        "cosine": float(np.vdot(reference, candidate) / (np.linalg.norm(reference) * np.linalg.norm(candidate))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "normalized_root_mean_square_error": float(
            np.sqrt(np.mean(np.square(difference))) / np.sqrt(np.mean(np.square(reference)))
        ),
    }


class StreamingExperts(InklingExperts):
    def __init__(self, repository: HuggingFaceSafetensorsRepository, layer_id: int) -> None:
        torch.nn.Module.__init__(self)
        self.repository = repository
        self.layer_id = layer_id
        self.last_routes: np.ndarray | None = None

    def load_expert(self, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"model.llm.layers.{self.layer_id}.mlp.experts"
        if self.layer_id == 2:
            w13 = deinterleave_gate_up(to_torch(self.repository.read_first_axis(f"{prefix}.w13_weight", expert_id)))
            w2 = to_torch(self.repository.read_first_axis(f"{prefix}.w2_weight", expert_id))
            return w13, w2
        packed_w13 = torch.from_numpy(self.repository.read_first_axis(f"{prefix}.w13_weight", expert_id))
        w13_scale = torch.from_numpy(
            self.repository.read_first_axis(f"{prefix}.w13_weight.scale", expert_id).astype(np.float32)
        )
        packed_w13 = deinterleave_gate_up(packed_w13)
        w13_scale = deinterleave_gate_up(w13_scale)
        w13 = decode_nvfp4(
            packed_w13.numpy(),
            w13_scale.numpy(),
            self.repository.read_first_axis(f"{prefix}.w13_weight.scale2", expert_id),
        )
        w2 = decode_nvfp4(
            self.repository.read_first_axis(f"{prefix}.w2_weight", expert_id),
            self.repository.read_first_axis(f"{prefix}.w2_weight.scale", expert_id),
            self.repository.read_first_axis(f"{prefix}.w2_weight.scale2", expert_id),
        )
        return w13, w2

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.last_routes = top_k_index.detach().cpu().numpy()
        output = torch.zeros_like(hidden_states)
        for expert_id in torch.unique(top_k_index, sorted=True).tolist():
            token_indices, route_slots = torch.where(top_k_index == expert_id)
            w13, w2 = self.load_expert(expert_id)
            gate_up = functional.linear(hidden_states[token_indices], w13)
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = functional.linear(functional.silu(gate) * up, w2)
            contribution = expert_output * top_k_weights[token_indices, route_slots, None]
            output.index_add_(0, token_indices, contribution)
            del w13, w2, gate_up, gate, up, expert_output, contribution
        return output


class ReferenceCheckpoint:
    def __init__(self) -> None:
        self.repository = HuggingFaceSafetensorsRepository(NVFP4_REPOSITORY, NVFP4_REVISION)

    def close(self) -> None:
        self.repository.close()

    def read(self, name: str) -> np.ndarray:
        return self.repository.read_tensor(name)

    def load_layer(self, config: object, layer_id: int) -> InklingDecoderLayer:
        with init_empty_weights(include_buffers=True):
            layer = InklingDecoderLayer(config, layer_id)
            if layer_id >= 2:
                layer.mlp.experts = StreamingExperts(self.repository, layer_id)
        layer.to_empty(device="cpu")
        layer.to(dtype=torch.bfloat16)
        prefix = f"model.llm.layers.{layer_id}"
        state: dict[str, np.ndarray] = {
            "self_attn.q_proj.weight": self.read(f"{prefix}.attn.wq_du.weight"),
            "self_attn.k_proj.weight": self.read(f"{prefix}.attn.wk_dv.weight"),
            "self_attn.v_proj.weight": self.read(f"{prefix}.attn.wv_dv.weight"),
            "self_attn.r_proj.weight": self.read(f"{prefix}.attn.wr_du.weight"),
            "self_attn.o_proj.weight": self.read(f"{prefix}.attn.wo_ud.weight"),
            "self_attn.k_sconv.conv1d.weight": self.read(f"{prefix}.attn.k_sconv.weight"),
            "self_attn.v_sconv.conv1d.weight": self.read(f"{prefix}.attn.v_sconv.weight"),
            "self_attn.q_norm.weight": self.read(f"{prefix}.attn.q_norm.weight"),
            "self_attn.k_norm.weight": self.read(f"{prefix}.attn.k_norm.weight"),
            "self_attn.rel_logits_proj.proj": self.read(f"{prefix}.attn.rel_logits_proj.proj"),
            "input_layernorm.weight": self.read(f"{prefix}.attn_norm.weight"),
            "post_attention_layernorm.weight": self.read(f"{prefix}.mlp_norm.weight"),
            "attn_sconv.conv1d.weight": self.read(f"{prefix}.attn_sconv.weight"),
            "mlp_sconv.conv1d.weight": self.read(f"{prefix}.mlp_sconv.weight"),
        }
        if layer_id < 2:
            w13 = deinterleave_gate_up_numpy(self.read(f"{prefix}.mlp.w13_dn.weight"))
            state["mlp.gate_proj.weight"], state["mlp.up_proj.weight"] = np.split(w13, 2, axis=-2)
            state["mlp.down_proj.weight"] = self.read(f"{prefix}.mlp.w2_md.weight")
            state["mlp.global_scale"] = self.read(f"{prefix}.mlp.global_scale")
        else:
            shared_w13 = deinterleave_gate_up_numpy(self.read(f"{prefix}.mlp.shared_experts.shared_w13_weight"))
            state["mlp.shared_experts.gate_proj"], state["mlp.shared_experts.up_proj"] = np.split(
                shared_w13, 2, axis=-2
            )
            state["mlp.shared_experts.down_proj"] = self.read(f"{prefix}.mlp.shared_experts.shared_w2_weight")
            state["mlp.gate.weight"] = self.read(f"{prefix}.mlp.gate.weight")
            state["mlp.gate.e_score_correction_bias"] = self.read(f"{prefix}.mlp.gate.bias")
            state["mlp.gate.global_scale"] = self.read(f"{prefix}.mlp.gate.global_scale")

        parameters = dict(layer.named_parameters())
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        if missing or unexpected:
            raise RuntimeError(
                f"INKLING_LAYER_STATE_MISMATCH layer={layer_id} missing={missing} unexpected={unexpected}"
            )
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(to_torch(value, parameters[name].dtype))
        return layer.eval()


def tokenize_prompt(prompt: str) -> np.ndarray:
    tokenizer_path = hf_hub_download(NVFP4_REPOSITORY, "tokenizer.json", revision=NVFP4_REVISION)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return np.asarray([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=np.int64)


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    normalized = hidden_states.float() * torch.rsqrt(hidden_states.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return (normalized * weight).to(hidden_states.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--validation-directory", type=Path)
    parser.add_argument("--save-hidden-directory", type=Path)
    parser.add_argument("--stop-after-layer", type=int, default=65)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(NVFP4_REPOSITORY, revision=NVFP4_REVISION).text_config
    config._attn_implementation = "eager"
    checkpoint = ReferenceCheckpoint()
    input_ids = tokenize_prompt(args.prompt)
    embedding_rows = np.stack(
        [checkpoint.repository.read_first_axis("model.llm.embed.weight", int(token_id)) for token_id in input_ids[0]]
    )
    embedding = to_torch(embedding_rows).unsqueeze(0)
    embed_norm = to_torch(checkpoint.read("model.llm.embed_norm.weight"))
    hidden_states = rms_norm(embedding, embed_norm)
    sequence_length = input_ids.shape[1]
    attention_mask = torch.full(
        (1, 1, sequence_length, sequence_length),
        torch.finfo(torch.bfloat16).min,
        dtype=torch.bfloat16,
    )
    attention_mask = torch.triu(attention_mask, diagonal=1)

    comparisons: dict[str, object] = {}
    routes: dict[str, object] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for layer_id in range(args.stop_after_layer + 1):
            layer = checkpoint.load_layer(config, layer_id)
            hidden_states = layer(hidden_states, attention_mask=attention_mask, conv_mask=None)
            reference_hidden = hidden_states.float().numpy(force=True)
            if args.save_hidden_directory is not None:
                args.save_hidden_directory.mkdir(parents=True, exist_ok=True)
                np.save(args.save_hidden_directory / f"layer_{layer_id:02d}_hidden.npy", reference_hidden)
            if args.validation_directory is not None:
                candidate_path = args.validation_directory / f"layer_{layer_id:02d}_hidden.npy"
                if candidate_path.exists():
                    comparisons[str(layer_id)] = compare(reference_hidden, np.load(candidate_path))
                if layer_id >= 2:
                    route_path = args.validation_directory / f"layer_{layer_id:02d}_routes.json"
                    if route_path.exists():
                        candidate_routes = np.asarray(json.loads(route_path.read_text()))
                        reference_routes = layer.mlp.experts.last_routes
                        routes[str(layer_id)] = {
                            "same_sets": bool(
                                np.array_equal(np.sort(reference_routes, axis=-1), np.sort(candidate_routes, axis=-1))
                            ),
                            "reference": reference_routes.tolist(),
                            "candidate": candidate_routes.tolist(),
                        }
            print(
                json.dumps(
                    {
                        "event": "cpu_layer_completed",
                        "layer": layer_id,
                        "hidden_sha256": hashlib.sha256(reference_hidden.tobytes()).hexdigest(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del layer
            gc.collect()

    result: dict[str, object] = {
        "checkpoint": {"repository": NVFP4_REPOSITORY, "revision": NVFP4_REVISION},
        "comparisons": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "input_ids": input_ids.tolist(),
        "prompt": args.prompt,
        "routes": routes,
        "stop_after_layer": args.stop_after_layer,
    }
    if args.stop_after_layer == 65:
        final_norm = to_torch(checkpoint.read("model.llm.norm.weight"))
        final_hidden = rms_norm(hidden_states, final_norm)[0, -1] / LOGITS_MUP_WIDTH_MULTIPLIER
        logits = np.empty(UNPADDED_VOCABULARY_SIZE, dtype=np.float32)
        for start in range(0, UNPADDED_VOCABULARY_SIZE, 4096):
            stop = min(start + 4096, UNPADDED_VOCABULARY_SIZE)
            unembed = to_torch(checkpoint.repository.read_first_axis_slice("model.llm.unembed.weight", start, stop))
            logits[start:stop] = functional.linear(final_hidden, unembed).float().numpy(force=True)
        top_indices = np.argsort(logits)[-10:][::-1]
        result["logits_sha256"] = hashlib.sha256(logits.tobytes()).hexdigest()
        result["top_10"] = [{"token_id": int(index), "logit": float(logits[index])} for index in top_indices]
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, logits=logits, metadata=json.dumps(result))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    checkpoint.close()


if __name__ == "__main__":
    main()
