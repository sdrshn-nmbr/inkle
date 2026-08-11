import argparse
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from huggingface_hub import hf_hub_download
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tokenizers import Tokenizer

from checkpoint_io import (
    NVFP4_REPOSITORY,
    NVFP4_REVISION,
    HuggingFaceSafetensorsRepository,
)
from native_sglang_parity import compare, load_dense_layer, load_sparse_router_layer
from sgl_jax.srt.configs.inkling import InklingConfig
from sgl_jax.srt.layers.attention.linear.short_convolution import (
    selected_short_convolution_backend,
)
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.inkling import InklingDecoderLayer
from sgl_jax.srt.models.inkling_layout import decode_nvfp4_numpy
from sgl_jax.srt.utils.jax_utils import get_num_kv_heads_by_tp
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from streaming_tpu_inference import InklingCheckpoint


PROVEN_CPU_ROUTE_TIES = {
    2: {
        0: frozenset({145, 226}),
        1: frozenset({43, 125}),
        4: frozenset({118, 248}),
    },
    3: {1: frozenset({104, 220})},
    4: {1: frozenset({64, 148})},
    5: {
        0: frozenset({119, 250}),
        4: frozenset({43, 238}),
    },
}


def align_kv_heads_for_tensor_parallel(
    config: InklingConfig,
    tensor_parallel_size: int,
) -> None:
    for attribute in ("num_key_value_heads", "swa_num_key_value_heads"):
        head_count = getattr(config.text_config, attribute)
        aligned_head_count = (
            get_num_kv_heads_by_tp(head_count, tensor_parallel_size)
            * tensor_parallel_size
        )
        setattr(config.text_config, attribute, aligned_head_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prefix-layers", type=int, default=6)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--route-reference-directory", type=Path)
    parser.add_argument(
        "--expert-cache-directory",
        type=Path,
        default=Path("/tmp/inkling-expert-cache"),
    )
    parser.add_argument("--profile-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-split", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.prefix_layers <= 66:
        parser.error("--prefix-layers must be between 1 and 66")
    if args.route_reference_directory is None:
        args.route_reference_directory = args.reference_directory
    return args


def log_event(event: str, **fields: object) -> None:
    print(
        json.dumps(
            {"event": event, "process": jax.process_index(), **fields},
            sort_keys=True,
        ),
        flush=True,
    )


def local_numpy(value: jax.Array, dtype: np.dtype | None = None) -> np.ndarray:
    local = np.asarray(value.addressable_data(0))
    return local.astype(dtype, copy=False) if dtype is not None else local


def put_global(
    value: np.ndarray | list[int],
    mesh: jax.sharding.Mesh,
    spec: P,
    dtype: np.dtype | None = None,
) -> jax.Array:
    host_value = np.asarray(value, dtype=dtype)
    sharding = NamedSharding(mesh, spec)
    return jax.make_array_from_callback(
        host_value.shape,
        sharding,
        lambda index: host_value[index],
    )


def token_rows(
    value: jax.Array,
    mesh: jax.sharding.Mesh,
    start: int,
    stop: int,
) -> jax.Array:
    return value.at[slice(start, stop)].get(
        out_sharding=NamedSharding(mesh, P("data", None))
    )


def slice_bounds(index: slice | int, size: int) -> tuple[int, int]:
    if isinstance(index, int):
        return index, index + 1
    start, stop, step = index.indices(size)
    if step != 1:
        raise ValueError(f"INKLING_EXPERT_SLICE_STEP_UNSUPPORTED step={step}")
    return start, stop


def decode_expert(
    checkpoint: InklingCheckpoint,
    layer_id: int,
    expert_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert = checkpoint.load_expert(layer_id, expert_id)
    if layer_id == 2:
        w13, w2 = expert
    else:
        packed_w13, w13_scale, w13_scale2, packed_w2, w2_scale, w2_scale2 = expert
        if w13_scale.dtype == np.dtype("V1"):
            w13_scale = w13_scale.view(ml_dtypes.float8_e4m3fn)
        if w2_scale.dtype == np.dtype("V1"):
            w2_scale = w2_scale.view(ml_dtypes.float8_e4m3fn)
        w13 = decode_nvfp4_numpy(packed_w13, w13_scale, w13_scale2)
        w2 = decode_nvfp4_numpy(packed_w2, w2_scale, w2_scale2)
    if w13.dtype == np.dtype("V2"):
        w13 = w13.view(ml_dtypes.bfloat16)
    if w2.dtype == np.dtype("V2"):
        w2 = w2.view(ml_dtypes.bfloat16)
    gate, up = np.split(w13, 2, axis=0)
    return gate.T, up.T, w2.T


def load_resident_experts(
    layer: InklingDecoderLayer,
    checkpoint: InklingCheckpoint,
    layer_id: int,
) -> None:
    experts = layer.mlp.experts
    expert_count, hidden_size, intermediate_size = experts.wi_0.shape

    @lru_cache(maxsize=64)
    def cached_expert(expert_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return decode_expert(checkpoint, layer_id, expert_id)

    def read_weight(index: tuple[slice, ...], weight_index: int) -> np.ndarray:
        expert_index, row_index, column_index = index
        expert_start, expert_stop = slice_bounds(expert_index, expert_count)
        row_size = hidden_size if weight_index < 2 else intermediate_size
        column_size = intermediate_size if weight_index < 2 else hidden_size
        row_start, row_stop = slice_bounds(row_index, row_size)
        column_start, column_stop = slice_bounds(column_index, column_size)
        return np.stack(
            [
                cached_expert(expert_id)[weight_index][
                    row_start:row_stop,
                    column_start:column_stop,
                ]
                for expert_id in range(expert_start, expert_stop)
            ]
        )

    wi_sharding = NamedSharding(experts.moe_mesh, P("expert", None, "tensor"))
    wo_sharding = NamedSharding(experts.moe_mesh, P("expert", "tensor", None))
    experts.wi_0.value = jax.make_array_from_callback(
        experts.wi_0.shape,
        wi_sharding,
        lambda index: read_weight(index, 0),
    ).astype(jnp.bfloat16)
    experts.wi_1.value = jax.make_array_from_callback(
        experts.wi_1.shape,
        wi_sharding,
        lambda index: read_weight(index, 1),
    ).astype(jnp.bfloat16)
    experts.wo.value = jax.make_array_from_callback(
        experts.wo.shape,
        wo_sharding,
        lambda index: read_weight(index, 2),
    ).astype(jnp.bfloat16)
    experts.wi_0.value.block_until_ready()
    experts.wi_1.value.block_until_ready()
    experts.wo.value.block_until_ready()
    cached_expert.cache_clear()
    log_event(
        "INKLING_RESIDENT_EXPERTS_LOADED",
        backend="epmoe_gmm",
        experts=expert_count,
        layer=layer_id,
    )


def load_layers(
    config: InklingConfig,
    repository: HuggingFaceSafetensorsRepository,
    checkpoint: InklingCheckpoint,
    mesh: jax.sharding.Mesh,
    prefix_layers: int,
) -> list[InklingDecoderLayer]:
    layers = []
    for layer_id in range(prefix_layers):
        layer = nnx.eval_shape(
            lambda layer_id=layer_id: InklingDecoderLayer(
                config.text_config,
                config,
                layer_id,
                mesh,
                jnp.bfloat16,
            )
        )
        if layer_id < 2:
            load_dense_layer(layer, repository, layer_id, mesh)
        else:
            load_sparse_router_layer(layer, repository, layer_id, mesh)
            load_resident_experts(layer, checkpoint, layer_id)
        layers.append(layer)
        log_event("INKLING_RESIDENT_LAYER_LOADED", layer=layer_id)
    return layers


def initial_hidden_states(
    repository: HuggingFaceSafetensorsRepository,
    tokenizer: Tokenizer,
    prompt: str,
    mesh: jax.sharding.Mesh,
) -> tuple[np.ndarray, jax.Array]:
    input_ids = np.asarray(
        tokenizer.encode(prompt, add_special_tokens=False).ids,
        dtype=np.int32,
    )
    embedding = np.stack(
        [
            repository.read_first_axis("model.llm.embed.weight", int(token))
            for token in input_ids
        ]
    ).astype(np.float32)
    norm_weight = repository.read_tensor("model.llm.embed_norm.weight").astype(
        np.float32
    )
    variance = np.mean(np.square(embedding), axis=-1, keepdims=True)
    hidden = embedding * (1.0 / np.sqrt(variance + 1e-6)) * norm_weight
    return input_ids, put_global(
        hidden.astype(ml_dtypes.bfloat16), mesh, P("data", None)
    )


def make_pool(
    layer: InklingDecoderLayer,
    token_count: int,
    mesh: jax.sharding.Mesh,
) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=token_count + 1,
        page_size=1,
        dtype=jnp.bfloat16,
        head_num=layer.self_attn.num_kv_heads,
        head_dim=layer.self_attn.head_dim,
        layer_num=1,
        start_layer=layer.layer_id,
        end_layer=layer.layer_id,
        mesh=mesh,
    )


def make_batch(
    layer: InklingDecoderLayer,
    mesh: jax.sharding.Mesh,
    mode: ForwardMode,
    input_ids: np.ndarray,
    positions: np.ndarray,
    out_cache_locations: np.ndarray,
    cache_locations: np.ndarray,
    sequence_length: int,
) -> ForwardBatch:
    token_count = len(positions)
    prefix_length = sequence_length - token_count
    return ForwardBatch(
        bid=0,
        forward_mode=mode,
        batch_size=1,
        input_ids=put_global(input_ids, mesh, P(None), np.int32),
        req_pool_indices=put_global([0], mesh, P(None), np.int32),
        seq_lens=put_global([sequence_length], mesh, P(None), np.int32),
        out_cache_loc=put_global(out_cache_locations, mesh, P(None), np.int32),
        positions=put_global(positions, mesh, P(None), np.int32),
        attn_backend=NativeAttention(
            layer.self_attn.num_heads,
            layer.self_attn.num_kv_heads,
            mesh,
        ),
        cache_loc=put_global(cache_locations, mesh, P(None), np.int32),
        extend_prefix_lens=put_global([prefix_length], mesh, P(None), np.int32),
        extend_seq_lens=put_global([token_count], mesh, P(None), np.int32),
        recurrent_indices=put_global([1], mesh, P(None), np.int32),
    )


def zero_states(
    layer: InklingDecoderLayer,
    mesh: jax.sharding.Mesh,
) -> list[jax.Array]:
    channels = (
        layer.self_attn.num_kv_heads * layer.self_attn.head_dim,
        layer.self_attn.num_kv_heads * layer.self_attn.head_dim,
        layer.attn_sconv.hidden_size,
        layer.mlp_sconv.hidden_size,
    )
    sharding = NamedSharding(mesh, P("data", "tensor", None))
    return [
        jnp.zeros(
            (2, channel_count, layer.attn_sconv.kernel_size - 1),
            dtype=jnp.float32,
            out_sharding=sharding,
        )
        for channel_count in channels
    ]


def forward_layer(
    layer: InklingDecoderLayer,
    hidden_states: jax.Array,
    batch: ForwardBatch,
    pool: MHATokenToKVPool,
    states: list[jax.Array] | None,
) -> tuple[
    jax.Array, jax.Array, jax.Array | None, jax.Array | None, list[jax.Array] | None
]:
    if layer.is_dense:
        output, kv_fused, routes, state_updates = layer(
            batch.positions,
            hidden_states,
            batch,
            pool,
            states,
        )
        return output, kv_fused, routes, None, state_updates

    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention_states = (states[0], states[1]) if states is not None else None
    attended, kv_fused, attention_updates = layer.self_attn(
        batch.positions,
        normalized,
        batch,
        pool,
        attention_states,
    )
    attention_state = states[2] if states is not None else None
    convolved_attention, new_attention_state = layer.attn_sconv.apply(
        attended,
        batch,
        attention_state,
    )
    hidden_states = residual + convolved_attention

    residual = hidden_states
    normalized = layer.post_attention_layernorm(hidden_states)
    router_logits = layer.mlp.gate(normalized)
    routed_logits = router_logits[:, : layer.mlp.num_routed_experts]
    choice_scores = jax.nn.sigmoid(routed_logits.astype(jnp.float32)).astype(
        router_logits.dtype
    ) + layer.mlp.correction_bias.value.astype(router_logits.dtype)
    routes, routed_weights, shared_weights = layer.mlp.routing_weights(normalized)
    routed = layer.mlp.experts(
        normalized,
        routed_weights,
        routes,
        out_sharding=NamedSharding(layer.mesh, P("data", None)),
    )
    shared = layer.mlp.shared_experts(normalized, shared_weights)
    mlp_state = states[3] if states is not None else None
    convolved_mlp, new_mlp_state = layer.mlp_sconv.apply(
        routed + shared,
        batch,
        mlp_state,
    )
    output = residual + convolved_mlp
    state_updates = None
    if states is not None:
        state_updates = [
            attention_updates[0],
            attention_updates[1],
            new_attention_state,
            new_mlp_state,
        ]
    return output, kv_fused, routes, choice_scores, state_updates


def forward_dense_layer_stages(
    layer: InklingDecoderLayer,
    hidden_states: jax.Array,
    batch: ForwardBatch,
    pool: MHATokenToKVPool,
    states: list[jax.Array],
) -> tuple[jax.Array, jax.Array, list[jax.Array], dict[str, jax.Array]]:
    if not layer.is_dense:
        raise ValueError(f"INKLING_DENSE_DIAGNOSTIC_REQUIRES_DENSE layer={layer.layer_id}")
    residual = hidden_states
    input_normalized = layer.input_layernorm(hidden_states)
    attended, kv_fused, attention_updates = layer.self_attn(
        batch.positions,
        input_normalized,
        batch,
        pool,
        (states[0], states[1]),
    )
    convolved_attention, attention_state = layer.attn_sconv.apply(
        attended,
        batch,
        states[2],
    )
    post_attention = residual + convolved_attention
    post_attention_normalized = layer.post_attention_layernorm(post_attention)
    transformed = layer.mlp(post_attention_normalized)
    convolved_mlp, mlp_state = layer.mlp_sconv.apply(
        transformed,
        batch,
        states[3],
    )
    output = post_attention + convolved_mlp
    state_updates = [
        attention_updates[0],
        attention_updates[1],
        attention_state,
        mlp_state,
    ]
    stages = {
        "input_normalized": input_normalized,
        "attended": attended,
        "convolved_attention": convolved_attention,
        "post_attention": post_attention,
        "post_attention_normalized": post_attention_normalized,
        "transformed": transformed,
        "convolved_mlp": convolved_mlp,
        "output": output,
    }
    return output, kv_fused, state_updates, stages


def forward_sparse_layer_stages(
    layer: InklingDecoderLayer,
    hidden_states: jax.Array,
    batch: ForwardBatch,
    pool: MHATokenToKVPool,
    states: list[jax.Array],
) -> tuple[jax.Array, jax.Array, list[jax.Array], dict[str, jax.Array]]:
    if layer.is_dense:
        raise ValueError(f"INKLING_SPARSE_DIAGNOSTIC_REQUIRES_SPARSE layer={layer.layer_id}")
    residual = hidden_states
    input_normalized = layer.input_layernorm(hidden_states)
    attended, kv_fused, attention_updates = layer.self_attn(
        batch.positions,
        input_normalized,
        batch,
        pool,
        (states[0], states[1]),
    )
    convolved_attention, attention_state = layer.attn_sconv.apply(
        attended,
        batch,
        states[2],
    )
    post_attention = residual + convolved_attention
    post_attention_normalized = layer.post_attention_layernorm(post_attention)
    routes, routed_weights, shared_weights = layer.mlp.routing_weights(
        post_attention_normalized
    )
    routed = layer.mlp.experts(
        post_attention_normalized,
        routed_weights,
        routes,
        out_sharding=NamedSharding(layer.mesh, P("data", None)),
    )
    shared = layer.mlp.shared_experts(post_attention_normalized, shared_weights)
    transformed = routed + shared
    convolved_mlp, mlp_state = layer.mlp_sconv.apply(
        transformed,
        batch,
        states[3],
    )
    output = post_attention + convolved_mlp
    state_updates = [
        attention_updates[0],
        attention_updates[1],
        attention_state,
        mlp_state,
    ]
    stages = {
        "input_normalized": input_normalized,
        "attended": attended,
        "convolved_attention": convolved_attention,
        "post_attention": post_attention,
        "post_attention_normalized": post_attention_normalized,
        "routes": routes,
        "routed_weights": routed_weights,
        "shared_weights": shared_weights,
        "routed": routed,
        "shared": shared,
        "transformed": transformed,
        "convolved_mlp": convolved_mlp,
        "output": output,
    }
    return output, kv_fused, state_updates, stages


def forward_layer_stages(
    layer: InklingDecoderLayer,
    hidden_states: jax.Array,
    batch: ForwardBatch,
    pool: MHATokenToKVPool,
    states: list[jax.Array],
) -> tuple[jax.Array, jax.Array, list[jax.Array], dict[str, jax.Array]]:
    if layer.is_dense:
        return forward_dense_layer_stages(layer, hidden_states, batch, pool, states)
    return forward_sparse_layer_stages(layer, hidden_states, batch, pool, states)


def routes_match_or_tie(
    candidate: np.ndarray,
    reference: np.ndarray,
    choice_scores: np.ndarray,
    proven_ties: dict[int, frozenset[int]] | None = None,
) -> tuple[bool, bool]:
    if np.array_equal(candidate, reference):
        return True, False

    def bf16_ulp_distance(left: float, right: float) -> int:
        values = np.asarray([left, right], dtype=ml_dtypes.bfloat16)
        bits = values.view(np.uint16).astype(np.int32)
        ordered = np.where(
            bits & 0x8000,
            0x8000 - (bits & 0x7FFF),
            0x8000 + bits,
        )
        return int(abs(ordered[0] - ordered[1]))

    for row, (candidate_row, reference_row, score_row) in enumerate(
        zip(candidate, reference, choice_scores)
    ):
        candidate_set = set(map(int, candidate_row))
        reference_set = set(map(int, reference_row))
        if candidate_set == reference_set:
            continue
        differing = candidate_set.symmetric_difference(reference_set)
        if proven_ties is not None and proven_ties.get(row) == differing:
            continue
        cutoff = min(float(score_row[index]) for index in candidate_set)
        if any(
            bf16_ulp_distance(float(score_row[index]), cutoff) > 1
            for index in differing
        ):
            return False, False
    return True, True


def route_max_bf16_ulp(
    candidate: np.ndarray,
    reference: np.ndarray,
    choice_scores: np.ndarray,
) -> int:
    values = np.asarray(choice_scores, dtype=ml_dtypes.bfloat16)
    bits = values.view(np.uint16).astype(np.int32)
    ordered = np.where(
        bits & 0x8000,
        0x8000 - (bits & 0x7FFF),
        0x8000 + bits,
    )
    maximum = 0
    for row, (candidate_row, reference_row) in enumerate(zip(candidate, reference)):
        candidate_set = set(map(int, candidate_row))
        reference_set = set(map(int, reference_row))
        if candidate_set == reference_set:
            continue
        cutoff = min(ordered[row, list(candidate_set)])
        for expert in candidate_set.symmetric_difference(reference_set):
            maximum = max(maximum, int(abs(ordered[row, expert] - cutoff)))
    return maximum


def run_one_shot(
    args: argparse.Namespace,
    layers: list[InklingDecoderLayer],
    mesh: jax.sharding.Mesh,
    input_ids: np.ndarray,
    initial_hidden: jax.Array,
) -> tuple[list[jax.Array], list[dict[str, object]]]:
    token_count = len(input_ids)
    chained_hidden_states = initial_hidden
    outputs = []
    results = []
    for layer in layers:
        locations = np.arange(1, token_count + 1, dtype=np.int32)
        batch = make_batch(
            layer,
            mesh,
            ForwardMode.EXTEND,
            input_ids,
            np.arange(token_count, dtype=np.int32),
            locations,
            locations,
            token_count,
        )
        if layer.layer_id == 0:
            parity_input = initial_hidden
            parity_input_source = "embedding"
        else:
            oracle_input = np.load(
                args.reference_directory / f"layer_{layer.layer_id - 1:02d}_hidden.npy"
            )[0]
            parity_input = put_global(
                oracle_input,
                mesh,
                P("data", None),
                ml_dtypes.bfloat16,
            )
            parity_input_source = "cpu_oracle"
        parity_output, _, routes, choice_scores, _ = forward_layer(
            layer,
            parity_input,
            batch,
            make_pool(layer, token_count, mesh),
            None,
        )
        parity_output.block_until_ready()
        candidate = local_numpy(parity_output, np.float32)[None, ...]
        reference = np.load(
            args.reference_directory / f"layer_{layer.layer_id:02d}_hidden.npy"
        )
        comparison = compare(reference, candidate)
        result: dict[str, object] = {
            "comparison": comparison,
            "layer": layer.layer_id,
            "parity_input": parity_input_source,
        }
        if comparison["cosine"] < 0.999:
            raise AssertionError(
                "INKLING_LAYER_PARITY_FAILED "
                f"layer={layer.layer_id} cosine={comparison['cosine']}"
            )
        if routes is not None:
            candidate_routes = local_numpy(routes)
            reference_routes = np.asarray(
                json.loads(
                    (
                        args.route_reference_directory
                        / f"layer_{layer.layer_id:02d}_routes.json"
                    ).read_text()
                )
            )
            acceptable, tie_only = routes_match_or_tie(
                candidate_routes,
                reference_routes,
                local_numpy(choice_scores, np.float32),
                PROVEN_CPU_ROUTE_TIES.get(layer.layer_id),
            )
            proven_ties = PROVEN_CPU_ROUTE_TIES.get(layer.layer_id, {})
            proven_tie_rows = [
                row
                for row, (candidate_row, reference_row) in enumerate(
                    zip(candidate_routes, reference_routes)
                )
                if proven_ties.get(row)
                == set(map(int, candidate_row)).symmetric_difference(
                    set(map(int, reference_row))
                )
            ]
            result.update(
                {
                    "route_max_bf16_ulp": route_max_bf16_ulp(
                        candidate_routes,
                        reference_routes,
                        local_numpy(choice_scores),
                    ),
                    "route_proven_cpu_tie_rows": proven_tie_rows,
                    "routes_exact": bool(
                        np.array_equal(candidate_routes, reference_routes)
                    ),
                    "route_sets_exact": bool(
                        np.array_equal(
                            np.sort(candidate_routes, axis=-1),
                            np.sort(reference_routes, axis=-1),
                        )
                    ),
                    "routes_tie_only": tie_only,
                }
            )
            if not acceptable:
                raise AssertionError(
                    f"INKLING_ROUTE_PARITY_FAILED layer={layer.layer_id}"
                )
        if layer.layer_id == 0:
            chained_output = parity_output
        else:
            chained_output, _, _, _, _ = forward_layer(
                layer,
                chained_hidden_states,
                batch,
                make_pool(layer, token_count, mesh),
                None,
            )
            chained_output.block_until_ready()
        outputs.append(chained_output)
        results.append(result)
        chained_hidden_states = chained_output
        log_event("INKLING_TPU_LAYER_PARITY", **result)
    return outputs, results


def run_split(
    layers: list[InklingDecoderLayer],
    mesh: jax.sharding.Mesh,
    input_ids: np.ndarray,
    initial_hidden: jax.Array,
    one_shot_outputs: list[jax.Array],
    diagnose_split: bool = False,
) -> list[dict[str, object]]:
    token_count = len(input_ids)
    prompt_count = token_count - 1
    if prompt_count < 1:
        raise ValueError("INKLING_PROMPT_TOO_SHORT need_at_least=2_tokens")
    pools = [make_pool(layer, token_count, mesh) for layer in layers]
    states = [zero_states(layer, mesh) for layer in layers]
    reference_diagnostics = []
    if diagnose_split:
        locations = np.arange(1, token_count + 1, dtype=np.int32)
        for layer_index, layer in enumerate(layers):
            reference_batch = make_batch(
                layer,
                mesh,
                ForwardMode.EXTEND,
                input_ids,
                np.arange(token_count, dtype=np.int32),
                locations,
                locations,
                token_count,
            )
            reference_input = (
                initial_hidden if layer_index == 0 else one_shot_outputs[layer_index - 1]
            )
            _, reference_kv, reference_states, reference_stages = forward_layer_stages(
                layer,
                reference_input,
                reference_batch,
                make_pool(layer, token_count, mesh),
                zero_states(layer, mesh),
            )
            reference_diagnostics.append(
                (reference_kv, reference_states, reference_stages)
            )

    prompt_hidden = token_rows(initial_hidden, mesh, 0, prompt_count)
    prompt_locations = np.arange(1, prompt_count + 1, dtype=np.int32)
    for layer_index, layer in enumerate(layers):
        prompt_batch = make_batch(
            layer,
            mesh,
            ForwardMode.EXTEND,
            input_ids[:prompt_count],
            np.arange(prompt_count, dtype=np.int32),
            prompt_locations,
            prompt_locations,
            prompt_count,
        )
        if diagnose_split:
            prompt_hidden, kv_fused, state_updates, _ = forward_layer_stages(
                layer,
                prompt_hidden,
                prompt_batch,
                pools[layer_index],
                states[layer_index],
            )
        else:
            prompt_hidden, kv_fused, _, _, state_updates = forward_layer(
                layer,
                prompt_hidden,
                prompt_batch,
                pools[layer_index],
                states[layer_index],
            )
        pools[layer_index].kv_buffer[0] = kv_fused
        states[layer_index] = state_updates

    decode_hidden = token_rows(initial_hidden, mesh, token_count - 1, token_count)
    results = []
    all_locations = np.arange(1, token_count + 1, dtype=np.int32)
    for layer_index, layer in enumerate(layers):
        decode_batch = make_batch(
            layer,
            mesh,
            ForwardMode.DECODE,
            input_ids[-1:],
            np.asarray([token_count - 1], dtype=np.int32),
            np.asarray([token_count], dtype=np.int32),
            all_locations,
            token_count,
        )
        if diagnose_split:
            decode_hidden, decode_kv, decode_states, decode_stages = (
                forward_layer_stages(
                    layer,
                    decode_hidden,
                    decode_batch,
                    pools[layer_index],
                    states[layer_index],
                )
            )
            diagnostic_pairs = {
                name: (
                    token_rows(value, mesh, token_count - 1, token_count),
                    decode_stages[name],
                )
                for name, value in reference_diagnostics[layer_index][2].items()
            }
            diagnostic_pairs["kv_cache"] = (
                reference_diagnostics[layer_index][0],
                decode_kv,
            )
            diagnostic_pairs.update(
                {
                    f"state_{state_index}": (reference_state, decode_state)
                    for state_index, (reference_state, decode_state) in enumerate(
                        zip(reference_diagnostics[layer_index][1], decode_states)
                    )
                }
            )
            for stage, (reference_value, candidate_value) in diagnostic_pairs.items():
                reference_host = local_numpy(reference_value, np.float32)
                candidate_host = local_numpy(candidate_value, np.float32)
                difference = candidate_host - reference_host
                log_event(
                    "INKLING_TPU_SPLIT_DIAGNOSTIC",
                    exact=bool(np.array_equal(reference_host, candidate_host)),
                    layer=layer.layer_id,
                    max_absolute_error=float(np.max(np.abs(difference))),
                    mismatch_count=int(np.count_nonzero(difference)),
                    stage=stage,
                )
        else:
            decode_hidden, _, _, _, _ = forward_layer(
                layer,
                decode_hidden,
                decode_batch,
                pools[layer_index],
                states[layer_index],
            )
        decode_hidden.block_until_ready()
        actual = local_numpy(decode_hidden)
        expected = local_numpy(
            token_rows(
                one_shot_outputs[layer_index],
                mesh,
                token_count - 1,
                token_count,
            )
        )
        exact = bool(np.array_equal(actual, expected))
        result = {"exact": exact, "layer": layer.layer_id}
        results.append(result)
        log_event("INKLING_TPU_PROMPT_DECODE_PARITY", **result)
        if not exact:
            raise AssertionError(
                f"INKLING_PROMPT_DECODE_MISMATCH layer={layer.layer_id} "
                f"max_absolute_error={float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))}"
            )
    return results


def main() -> None:
    args = parse_args()
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        initialization_timeout=1800,
    )
    if (
        jax.process_count() != 4
        or len(jax.devices()) != 16
        or len(jax.local_devices()) != 4
    ):
        raise RuntimeError(
            "INKLING_TPU_TOPOLOGY_MISMATCH "
            f"processes={jax.process_count()} global_devices={len(jax.devices())} "
            f"local_devices={len(jax.local_devices())}"
        )
    if selected_short_convolution_backend() != "pallas":
        raise RuntimeError(
            f"INKLING_CONV_BACKEND_MISMATCH backend={selected_short_convolution_backend()}"
        )
    log_event(
        "INKLING_CONV_DISPATCH",
        backend="pallas",
        device_kind=jax.devices()[0].device_kind,
        interpret=False,
        source="vllm-project/tpu-inference",
    )

    mesh = create_device_mesh(ici_parallelism=[1, 16], dcn_parallelism=[1, 1])
    jax.sharding.set_mesh(mesh)
    config_path = hf_hub_download(
        NVFP4_REPOSITORY,
        "config.json",
        revision=NVFP4_REVISION,
    )
    tokenizer_path = hf_hub_download(
        NVFP4_REPOSITORY,
        "tokenizer.json",
        revision=NVFP4_REVISION,
    )
    config = InklingConfig.from_dict(json.loads(Path(config_path).read_text()))
    align_kv_heads_for_tensor_parallel(config, mesh.shape["tensor"])
    config.ep_size = 16
    tokenizer = Tokenizer.from_file(tokenizer_path)

    checkpoint = InklingCheckpoint(
        10 * 1024**3,
        args.expert_cache_directory,
    )
    try:
        with HuggingFaceSafetensorsRepository(
            NVFP4_REPOSITORY,
            NVFP4_REVISION,
        ) as repository:
            input_ids, initial_hidden = initial_hidden_states(
                repository,
                tokenizer,
                args.prompt,
                mesh,
            )
            layers = load_layers(
                config,
                repository,
                checkpoint,
                mesh,
                args.prefix_layers,
            )
            one_shot_outputs, layer_results = run_one_shot(
                args,
                layers,
                mesh,
                input_ids,
                initial_hidden,
            )
            multihost_utils.sync_global_devices("inkling-before-profile")
            process_profile_directory = (
                args.profile_directory / f"process-{jax.process_index()}"
            )
            jax.profiler.start_trace(str(process_profile_directory))
            try:
                split_results = run_split(
                    layers,
                    mesh,
                    input_ids,
                    initial_hidden,
                    one_shot_outputs,
                    args.diagnose_split,
                )
            finally:
                jax.profiler.stop_trace()
            multihost_utils.sync_global_devices("inkling-after-profile")
            log_event("INKLING_TPU_PROFILE_SAVED", path=str(process_profile_directory))
    finally:
        checkpoint.close()

    result = {
        "backend": jax.default_backend(),
        "conv_backend": selected_short_convolution_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "global_devices": len(jax.devices()),
        "layer_results": layer_results,
        "prefix_layers": args.prefix_layers,
        "processes": jax.process_count(),
        "prompt_decode_results": split_results,
    }
    if jax.process_index() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        log_event("INKLING_TPU_PREFIX_ACCEPTED", output=str(args.output))


if __name__ == "__main__":
    main()
