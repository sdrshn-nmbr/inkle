# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx==0.28.1",
#   "huggingface-hub==1.3.3",
#   "jax[tpu]==0.10.2",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
#   "tokenizers==0.22.2",
# ]
# ///

import argparse
import collections
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from checkpoint_io import NVFP4_REPOSITORY, NVFP4_REVISION, HuggingFaceSafetensorsRepository
from huggingface_hub import hf_hub_download
from inkling_layout import decode_nvfp4_jax, deinterleave_gate_up_numpy
from jax import lax
from jax.experimental import multihost_utils
from jax.scipy.special import logsumexp
from tokenizers import Tokenizer

HIDDEN_SIZE = 6144
HEAD_DIM = 128
NUM_ATTENTION_HEADS = 64
ROUTED_EXPERTS = 256
EXPERTS_PER_TOKEN = 6
SHARED_EXPERTS = 2
SLIDING_WINDOW = 512
RMS_NORM_EPSILON = 1e-6
ROUTE_SCALE = 8.0
LOGITS_MUP_WIDTH_MULTIPLIER = 24.0
UNPADDED_VOCABULARY_SIZE = 200058
LOCAL_LAYER_IDS = frozenset(set(range(66)) - set(range(5, 66, 6)))
PROCESS_LAYER_PARTITIONS = (
    ((0, 5), (5, 9), (9, 13), (13, 17)),
    ((17, 21), (21, 25), (25, 29), (29, 33)),
    ((33, 37), (37, 41), (41, 45), (45, 49)),
    ((49, 53), (53, 57), (57, 61), (61, 66)),
)
SINGLE_PROCESS_LAYER_PARTITIONS = ((0, 17), (17, 33), (33, 49), (49, 66))

LayerParams = dict[str, jax.Array]
HostExpert = tuple[np.ndarray, ...]


def linear(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    return jnp.einsum("...h,oh->...o", hidden_states, weight)


def rms_norm(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    input_dtype = hidden_states.dtype
    as_float = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(as_float), axis=-1, keepdims=True)
    return (weight * as_float * lax.rsqrt(variance + RMS_NORM_EPSILON)).astype(input_dtype)


def short_convolution(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    residual = hidden_states
    channels_first = jnp.transpose(hidden_states.astype(jnp.float32), (0, 2, 1))
    padding = weight.shape[-1] - 1
    convolved = lax.conv_general_dilated(
        channels_first,
        weight.astype(jnp.float32),
        window_strides=(1,),
        padding=((padding, 0),),
        feature_group_count=channels_first.shape[1],
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return (jnp.transpose(convolved, (0, 2, 1)) + residual).astype(hidden_states.dtype)


def relative_logits(
    relative_states: jax.Array,
    projection: jax.Array,
    sequence_length: int,
) -> jax.Array:
    projected = jnp.einsum("bqhd,de->bhqe", relative_states, projection)
    positions = jnp.arange(sequence_length)
    distance = positions[:, None] - positions[None, :]
    gather_index = jnp.clip(distance, 0, projection.shape[-1] - 1)[None, None, :, :]
    position_bias = jnp.take_along_axis(projected, gather_index, axis=-1)
    valid = (distance >= 0) & (distance < projection.shape[-1])
    return jnp.where(valid[None, None, :, :], position_bias, 0.0)


@jax.jit
def attention_block(params: LayerParams, hidden_states: jax.Array) -> jax.Array:
    batch_size, sequence_length, _ = hidden_states.shape
    query = linear(hidden_states, params["q"])
    key = short_convolution(linear(hidden_states, params["k"]), params["k_sconv"])
    value = short_convolution(linear(hidden_states, params["v"]), params["v_sconv"])
    relative_states = linear(hidden_states, params["r"])

    key_value_heads = key.shape[-1] // HEAD_DIM
    query = query.reshape(batch_size, sequence_length, NUM_ATTENTION_HEADS, HEAD_DIM)
    key = key.reshape(batch_size, sequence_length, key_value_heads, HEAD_DIM)
    value = value.reshape(batch_size, sequence_length, key_value_heads, HEAD_DIM)
    query = jnp.transpose(rms_norm(query, params["q_norm"]), (0, 2, 1, 3))
    key = jnp.transpose(rms_norm(key, params["k_norm"]), (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))
    repeats = NUM_ATTENTION_HEADS // key_value_heads
    key = jnp.repeat(key, repeats, axis=1)
    value = jnp.repeat(value, repeats, axis=1)

    relative_states = relative_states.reshape(batch_size, sequence_length, NUM_ATTENTION_HEADS, -1)
    scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / HEAD_DIM
    scores += relative_logits(relative_states, params["rel_proj"], sequence_length)
    positions = jnp.arange(sequence_length)
    allowed = positions[None, :] <= positions[:, None]
    if params["rel_proj"].shape[-1] == SLIDING_WINDOW:
        allowed &= positions[None, :] > positions[:, None] - SLIDING_WINDOW
    scores = jnp.where(allowed[None, None, :, :], scores, -jnp.inf)
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
    attended = jnp.matmul(probabilities, value)
    attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch_size, sequence_length, -1)
    return linear(attended, params["o"])


@jax.jit
def attention_residual(params: LayerParams, hidden_states: jax.Array) -> jax.Array:
    normalized = rms_norm(hidden_states, params["attn_norm"])
    attended = attention_block(params, normalized)
    return hidden_states + short_convolution(attended, params["attn_sconv"])


@jax.jit
def dense_residual(params: LayerParams, hidden_states: jax.Array) -> jax.Array:
    normalized = rms_norm(hidden_states, params["mlp_norm"])
    gate = linear(normalized, params["dense_gate"])
    up = linear(normalized, params["dense_up"])
    transformed = linear(jax.nn.silu(gate) * up, params["dense_down"])
    transformed *= params["dense_scale"]
    return hidden_states + short_convolution(transformed, params["mlp_sconv"])


@jax.jit
def route_and_shared(
    params: LayerParams,
    hidden_states: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    normalized = rms_norm(hidden_states, params["mlp_norm"])
    flat = normalized.reshape(-1, HIDDEN_SIZE)
    router_logits = linear(flat, params["router_weight"])
    routed_logits = router_logits[:, :ROUTED_EXPERTS]
    scores_for_choice = jax.nn.sigmoid(routed_logits) + params["router_bias"]
    _, topk_indices = lax.top_k(scores_for_choice, EXPERTS_PER_TOKEN)
    chosen_logits = jnp.take_along_axis(routed_logits, topk_indices, axis=-1)
    shared_logits = router_logits[:, ROUTED_EXPERTS:]
    selected_logits = jnp.concatenate((chosen_logits, shared_logits), axis=-1)
    log_probabilities = jax.nn.log_sigmoid(selected_logits)
    weights = jnp.exp(log_probabilities - logsumexp(log_probabilities, axis=-1, keepdims=True))
    weights *= ROUTE_SCALE * params["router_scale"]
    routed_weights = weights[:, :EXPERTS_PER_TOKEN]
    shared_weights = weights[:, EXPERTS_PER_TOKEN:]

    shared_gate_up = jnp.einsum("th,eih->eti", flat, params["shared_w13"])
    shared_gate, shared_up = jnp.split(shared_gate_up, 2, axis=-1)
    shared_activated = jax.nn.silu(shared_gate) * shared_up
    shared_activated *= jnp.transpose(shared_weights, (1, 0))[:, :, None]
    shared = jnp.einsum("eti,ehi->eth", shared_activated, params["shared_w2"])
    shared = jnp.sum(shared.astype(jnp.float32), axis=0).astype(flat.dtype)
    return normalized, topk_indices, routed_weights, shared


@jax.jit
def bf16_expert_forward(x: jax.Array, w13: jax.Array, w2: jax.Array) -> jax.Array:
    gate_up = linear(x, w13)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return linear(jax.nn.silu(gate) * up, w2)


@jax.jit
def nvfp4_expert_forward(
    x: jax.Array,
    packed_w13: jax.Array,
    w13_scale: jax.Array,
    w13_scale2: jax.Array,
    packed_w2: jax.Array,
    w2_scale: jax.Array,
    w2_scale2: jax.Array,
) -> jax.Array:
    w13 = decode_nvfp4_jax(packed_w13, w13_scale, w13_scale2)
    w2 = decode_nvfp4_jax(packed_w2, w2_scale, w2_scale2)
    return bf16_expert_forward(x, w13, w2)


@jax.jit
def sparse_residual(
    params: LayerParams,
    hidden_states: jax.Array,
    routed: jax.Array,
    shared: jax.Array,
) -> jax.Array:
    transformed = (routed + shared).reshape(hidden_states.shape)
    return hidden_states + short_convolution(transformed, params["mlp_sconv"])


class ExpertCache:
    def __init__(self, maximum_bytes: int) -> None:
        self.maximum_bytes = maximum_bytes
        self.current_bytes = 0
        self.values: collections.OrderedDict[tuple[int, int], HostExpert] = collections.OrderedDict()

    def get(self, key: tuple[int, int]) -> HostExpert | None:
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
        return value

    def put(self, key: tuple[int, int], value: HostExpert) -> None:
        size = sum(array.nbytes for array in value)
        while self.values and self.current_bytes + size > self.maximum_bytes:
            _, evicted = self.values.popitem(last=False)
            self.current_bytes -= sum(array.nbytes for array in evicted)
        self.values[key] = value
        self.current_bytes += size


@dataclass
class RuntimeMetrics:
    checkpoint_read_seconds: float = 0.0
    expert_execute_seconds: float = 0.0
    expert_loads: int = 0


class InklingCheckpoint:
    def __init__(self, maximum_expert_cache_bytes: int, expert_disk_cache_directory: Path | None = None) -> None:
        self.repository = HuggingFaceSafetensorsRepository(NVFP4_REPOSITORY, NVFP4_REVISION)
        self.expert_cache = ExpertCache(maximum_expert_cache_bytes)
        self.expert_disk_cache_directory = expert_disk_cache_directory
        self.metrics = RuntimeMetrics()

    def close(self) -> None:
        self.repository.close()

    def read(self, name: str) -> np.ndarray:
        started = time.perf_counter()
        value = self.repository.read_tensor(name)
        self.metrics.checkpoint_read_seconds += time.perf_counter() - started
        return value

    def load_layer(self, layer_id: int, device: jax.Device) -> LayerParams:
        prefix = f"model.llm.layers.{layer_id}"
        mapping = {
            "q": f"{prefix}.attn.wq_du.weight",
            "k": f"{prefix}.attn.wk_dv.weight",
            "v": f"{prefix}.attn.wv_dv.weight",
            "r": f"{prefix}.attn.wr_du.weight",
            "o": f"{prefix}.attn.wo_ud.weight",
            "q_norm": f"{prefix}.attn.q_norm.weight",
            "k_norm": f"{prefix}.attn.k_norm.weight",
            "rel_proj": f"{prefix}.attn.rel_logits_proj.proj",
            "k_sconv": f"{prefix}.attn.k_sconv.weight",
            "v_sconv": f"{prefix}.attn.v_sconv.weight",
            "attn_norm": f"{prefix}.attn_norm.weight",
            "attn_sconv": f"{prefix}.attn_sconv.weight",
            "mlp_norm": f"{prefix}.mlp_norm.weight",
            "mlp_sconv": f"{prefix}.mlp_sconv.weight",
        }
        host_params = {key: self.read(name) for key, name in mapping.items()}
        if layer_id < 2:
            raw_w13 = deinterleave_gate_up_numpy(self.read(f"{prefix}.mlp.w13_dn.weight"))
            host_params["dense_gate"], host_params["dense_up"] = np.split(raw_w13, 2, axis=-2)
            host_params["dense_down"] = self.read(f"{prefix}.mlp.w2_md.weight")
            host_params["dense_scale"] = self.read(f"{prefix}.mlp.global_scale")
        else:
            raw_shared_w13 = deinterleave_gate_up_numpy(self.read(f"{prefix}.mlp.shared_experts.shared_w13_weight"))
            host_params["shared_w13"] = raw_shared_w13
            host_params["shared_w2"] = self.read(f"{prefix}.mlp.shared_experts.shared_w2_weight")
            host_params["router_weight"] = self.read(f"{prefix}.mlp.gate.weight")
            host_params["router_bias"] = self.read(f"{prefix}.mlp.gate.bias")
            host_params["router_scale"] = self.read(f"{prefix}.mlp.gate.global_scale")
        return {key: jax.device_put(value, device) for key, value in host_params.items()}

    def load_expert(self, layer_id: int, expert_id: int) -> HostExpert:
        key = (layer_id, expert_id)
        cached = self.expert_cache.get(key)
        if cached is not None:
            return cached
        disk_path = None
        if self.expert_disk_cache_directory is not None:
            disk_path = self.expert_disk_cache_directory / f"layer_{layer_id:02d}_expert_{expert_id:03d}.npz"
            if disk_path.exists():
                with np.load(disk_path) as bundle:
                    value = tuple(bundle[name].copy() for name in sorted(bundle.files))
                self.expert_cache.put(key, value)
                return value
        started = time.perf_counter()
        prefix = f"model.llm.layers.{layer_id}.mlp.experts"
        if layer_id == 2:
            value = (
                deinterleave_gate_up_numpy(self.repository.read_first_axis(f"{prefix}.w13_weight", expert_id)),
                self.repository.read_first_axis(f"{prefix}.w2_weight", expert_id),
            )
        else:
            value = (
                deinterleave_gate_up_numpy(self.repository.read_first_axis(f"{prefix}.w13_weight", expert_id)),
                deinterleave_gate_up_numpy(self.repository.read_first_axis(f"{prefix}.w13_weight.scale", expert_id)),
                self.repository.read_first_axis(f"{prefix}.w13_weight.scale2", expert_id),
                self.repository.read_first_axis(f"{prefix}.w2_weight", expert_id),
                self.repository.read_first_axis(f"{prefix}.w2_weight.scale", expert_id),
                self.repository.read_first_axis(f"{prefix}.w2_weight.scale2", expert_id),
            )
        self.metrics.checkpoint_read_seconds += time.perf_counter() - started
        self.metrics.expert_loads += 1
        if disk_path is not None:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = disk_path.with_suffix(".tmp")
            with temporary_path.open("wb") as temporary_file:
                np.savez(temporary_file, *value)
            temporary_path.replace(disk_path)
        self.expert_cache.put(key, value)
        return value


def run_sparse_layer(
    checkpoint: InklingCheckpoint,
    params: LayerParams,
    hidden_states: jax.Array,
    layer_id: int,
    device: jax.Device,
) -> tuple[jax.Array, np.ndarray]:
    normalized, topk_indices, routed_weights, shared = route_and_shared(params, hidden_states)
    host_indices = np.asarray(topk_indices)
    routed = jax.device_put(np.zeros((host_indices.shape[0], HIDDEN_SIZE), dtype=np.float32), device).astype(
        hidden_states.dtype
    )
    for expert_id in np.unique(host_indices):
        token_indices, route_slots = np.where(host_indices == expert_id)
        device_token_indices = jax.device_put(token_indices, device)
        device_route_slots = jax.device_put(route_slots, device)
        expert_input = normalized.reshape(-1, HIDDEN_SIZE)[device_token_indices]
        expert_weights = tuple(
            jax.device_put(array, device) for array in checkpoint.load_expert(layer_id, int(expert_id))
        )
        started = time.perf_counter()
        if layer_id == 2:
            expert_output = bf16_expert_forward(expert_input, *expert_weights)
        else:
            expert_output = nvfp4_expert_forward(expert_input, *expert_weights)
        contribution = (expert_output * routed_weights[device_token_indices, device_route_slots, None]).astype(
            hidden_states.dtype
        )
        routed = routed.at[device_token_indices].add(contribution)
        routed.block_until_ready()
        checkpoint.metrics.expert_execute_seconds += time.perf_counter() - started
        for array in expert_weights:
            array.delete()
    return sparse_residual(params, hidden_states, routed, shared), host_indices


def broadcast_hidden(hidden_states: np.ndarray, source_process: int) -> np.ndarray:
    return np.asarray(
        multihost_utils.broadcast_one_to_all(
            hidden_states,
            is_source=jax.process_index() == source_process,
        )
    )


def tokenize_prompt(prompt: str) -> np.ndarray:
    tokenizer_path = hf_hub_download(
        NVFP4_REPOSITORY,
        "tokenizer.json",
        revision=NVFP4_REVISION,
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return np.asarray([tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=np.int32)


def log_event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "process": jax.process_index(), **fields}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-directory", type=Path)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--input-hidden", type=Path)
    parser.add_argument("--expert-cache-gib", type=float, default=24.0)
    parser.add_argument("--expert-disk-cache-directory", type=Path, default=Path("/tmp/inkling-expert-cache"))
    parser.add_argument("--initialize-distributed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def initialize_distributed() -> None:
    jax.distributed.initialize()


def main() -> None:
    args = parse_args()
    if args.initialize_distributed:
        initialize_distributed()
    process_count = jax.process_count()
    if process_count not in (1, 4):
        raise RuntimeError(f"INKLING_UNSUPPORTED_PROCESS_COUNT expected=1_or_4 actual={process_count}")
    if process_count == 4 and len(jax.local_devices()) != 4:
        raise RuntimeError(f"INKLING_UNSUPPORTED_LOCAL_DEVICE_COUNT expected=4 actual={len(jax.local_devices())}")

    process_id = jax.process_index()
    effective_process_id = process_id if process_count == 4 else 0
    devices = jax.local_devices()
    if process_count == 1 and len(devices) < 4:
        raise RuntimeError(f"INKLING_REQUIRES_FOUR_LOCAL_DEVICES actual={len(devices)}")
    partitions = (
        PROCESS_LAYER_PARTITIONS[effective_process_id] if process_count == 4 else SINGLE_PROCESS_LAYER_PARTITIONS
    )
    layer_devices = {
        layer_id: devices[device_id]
        for device_id, (start, stop) in enumerate(partitions)
        for layer_id in range(start, stop)
    }
    input_ids = tokenize_prompt(args.prompt)
    if not 0 <= args.start_layer < 66:
        raise ValueError(f"INKLING_INVALID_START_LAYER layer={args.start_layer}")
    if args.start_layer > 0 and args.input_hidden is None:
        raise ValueError("INKLING_INPUT_HIDDEN_REQUIRED when start-layer is greater than 0")
    checkpoint = InklingCheckpoint(
        int(args.expert_cache_gib * 1024**3),
        expert_disk_cache_directory=args.expert_disk_cache_directory,
    )
    active_layer_devices = {
        layer_id: device for layer_id, device in layer_devices.items() if layer_id >= args.start_layer
    }
    log_event("load_started", layers=sorted(active_layer_devices))
    local_layers = {}
    for layer_id, device in active_layer_devices.items():
        local_layers[layer_id] = checkpoint.load_layer(layer_id, device)
        log_event("layer_loaded", layer=layer_id)
    log_event("load_completed", checkpoint_read_seconds=checkpoint.metrics.checkpoint_read_seconds)

    if args.start_layer > 0:
        hidden = np.load(args.input_hidden)
    elif effective_process_id == 0:
        embedding_device = devices[0]
        embedding = jax.device_put(checkpoint.read("model.llm.embed.weight"), embedding_device)
        embed_norm = jax.device_put(checkpoint.read("model.llm.embed_norm.weight"), embedding_device)
        hidden_device = rms_norm(embedding[jax.device_put(input_ids, embedding_device)], embed_norm)
        hidden = np.asarray(hidden_device)
        embedding.delete()
        embed_norm.delete()
    else:
        hidden = np.zeros((*input_ids.shape, HIDDEN_SIZE), dtype=ml_dtypes.bfloat16)
    if process_count == 4:
        hidden = broadcast_hidden(hidden, 0)

    routes: dict[str, list[list[int]]] = {}
    inference_started = time.perf_counter()
    for stage_id in range(process_count if process_count == 4 else 1):
        if effective_process_id == stage_id:
            for layer_id in sorted(local_layers):
                device = layer_devices[layer_id]
                hidden_device = jax.device_put(hidden, device)
                hidden_device = attention_residual(local_layers[layer_id], hidden_device)
                if layer_id < 2:
                    hidden_device = dense_residual(local_layers[layer_id], hidden_device)
                else:
                    hidden_device, layer_routes = run_sparse_layer(
                        checkpoint,
                        local_layers[layer_id],
                        hidden_device,
                        layer_id,
                        device,
                    )
                    routes[str(layer_id)] = layer_routes.tolist()
                hidden = np.asarray(hidden_device)
                if args.validation_directory is not None:
                    args.validation_directory.mkdir(parents=True, exist_ok=True)
                    np.save(args.validation_directory / f"layer_{layer_id:02d}_hidden.npy", hidden)
                    if layer_id >= 2:
                        (args.validation_directory / f"layer_{layer_id:02d}_routes.json").write_text(
                            json.dumps(routes[str(layer_id)])
                        )
                log_event("layer_completed", layer=layer_id, hidden_sha256=hashlib.sha256(hidden.tobytes()).hexdigest())
        if process_count == 4:
            hidden = broadcast_hidden(hidden, stage_id)

    if effective_process_id == (3 if process_count == 4 else 0):
        output_device = devices[-1]
        final_norm = jax.device_put(checkpoint.read("model.llm.norm.weight"), output_device)
        unembed = jax.device_put(checkpoint.read("model.llm.unembed.weight"), output_device)
        final_hidden = rms_norm(jax.device_put(hidden, output_device), final_norm)
        logits = linear(final_hidden / LOGITS_MUP_WIDTH_MULTIPLIER, unembed)
        logits = np.asarray(logits[..., :UNPADDED_VOCABULARY_SIZE], dtype=np.float32)
        last_logits = logits[0, -1]
        top_indices = np.argsort(last_logits)[-10:][::-1]
        result = {
            "checkpoint": {"repository": NVFP4_REPOSITORY, "revision": NVFP4_REVISION},
            "checkpoint_read_seconds": checkpoint.metrics.checkpoint_read_seconds,
            "expert_cache_bytes": checkpoint.expert_cache.current_bytes,
            "expert_execute_seconds": checkpoint.metrics.expert_execute_seconds,
            "expert_loads": checkpoint.metrics.expert_loads,
            "inference_seconds": time.perf_counter() - inference_started,
            "input_ids": input_ids.tolist(),
            "logits_sha256": hashlib.sha256(last_logits.tobytes()).hexdigest(),
            "prompt": args.prompt,
            "routes": routes,
            "top_10": [{"token_id": int(index), "logit": float(last_logits[index])} for index in top_indices],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output, logits=last_logits, metadata=json.dumps(result))
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    checkpoint.close()


if __name__ == "__main__":
    main()
