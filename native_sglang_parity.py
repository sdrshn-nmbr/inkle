import argparse
import gc
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tokenizers import Tokenizer

from checkpoint_io import (
    NVFP4_REPOSITORY,
    NVFP4_REVISION,
    HuggingFaceSafetensorsRepository,
)
from sgl_jax.srt.configs.inkling import InklingConfig
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.inkling import InklingDecoderLayer
from sgl_jax.srt.models.inkling_layout import decode_nvfp4_numpy
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from streaming_tpu_inference import InklingCheckpoint

CONFIG_PATH = Path(
    "/Users/sudarshan/.cache/huggingface/hub/models--thinkingmachines--Inkling-NVFP4/"
    "snapshots/deeb2d05eaa977db4ff7727db33670a2e05938cf/config.json"
)
TOKENIZER_PATH = CONFIG_PATH.with_name("tokenizer.json")
UNPADDED_VOCABULARY_SIZE = 200058


def replicate_kv_head_blocks(
    value: np.ndarray,
    *,
    target_heads: int,
    head_dim: int,
    axis: int,
) -> np.ndarray:
    chex.assert_rank(value, 2)
    normalized_axis = axis % value.ndim
    source_width = value.shape[normalized_axis]
    if source_width % head_dim != 0:
        raise ValueError(
            "INKLING_KV_HEAD_WIDTH_INVALID "
            f"width={source_width} head_dim={head_dim} axis={axis}"
        )
    source_heads = source_width // head_dim
    if target_heads % source_heads != 0:
        raise ValueError(
            "INKLING_KV_HEAD_REPLICATION_INVALID "
            f"source_heads={source_heads} target_heads={target_heads}"
        )
    if source_heads == target_heads:
        return value

    head_shape = list(value.shape)
    head_shape[normalized_axis : normalized_axis + 1] = [source_heads, head_dim]
    replicated = np.repeat(
        value.reshape(head_shape),
        target_heads // source_heads,
        axis=normalized_axis,
    )
    output_shape = list(value.shape)
    output_shape[normalized_axis] = target_heads * head_dim
    result = replicated.reshape(output_shape)
    chex.assert_shape(result, tuple(output_shape))
    return result


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.dtype == np.dtype("V2"):
        reference = reference.view(ml_dtypes.bfloat16)
    if candidate.dtype == np.dtype("V2"):
        candidate = candidate.view(ml_dtypes.bfloat16)
    reference = reference.astype(np.float32).ravel()
    candidate = candidate.astype(np.float32).ravel()
    difference = candidate - reference
    return {
        "cosine": float(
            np.vdot(reference, candidate)
            / (np.linalg.norm(reference) * np.linalg.norm(candidate))
        ),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "normalized_root_mean_square_error": float(
            np.sqrt(np.mean(np.square(difference)))
            / np.sqrt(np.mean(np.square(reference)))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    layer_selection = parser.add_mutually_exclusive_group(required=True)
    layer_selection.add_argument("--layer", type=int)
    layer_selection.add_argument("--all-layers", action="store_true")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--route-reference-directory", type=Path)
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument("--force-reference-routes", action="store_true")
    parser.add_argument("--save-hidden-directory", type=Path)
    parser.add_argument("--logits-output", type=Path)
    parser.add_argument(
        "--expert-cache-directory", type=Path, default=Path("/tmp/inkling-expert-cache")
    )
    args = parser.parse_args()
    if args.layer is not None and not 0 <= args.layer < 66:
        parser.error("--layer must be between 0 and 65")
    if args.all_layers and args.router_only:
        parser.error("--router-only cannot be used with --all-layers")
    if args.force_reference_routes and args.route_reference_directory is None:
        parser.error("--force-reference-routes requires --route-reference-directory")
    if args.logits_output is not None and not args.all_layers:
        parser.error("--logits-output requires --all-layers")
    return args


def assign(
    parameter_owner: object, name: str, value: np.ndarray, sharding: NamedSharding
) -> None:
    target_dtype = getattr(parameter_owner, name).value.dtype
    host_value = np.asarray(value).astype(target_dtype, copy=False)
    array = jax.make_array_from_callback(
        host_value.shape,
        sharding,
        lambda index: host_value[index],
    )
    setattr(parameter_owner, name, nnx.Param(array))


def load_attention_layer(
    layer: InklingDecoderLayer,
    repository: HuggingFaceSafetensorsRepository,
    layer_id: int,
    mesh: jax.sharding.Mesh,
) -> None:
    prefix = f"model.llm.layers.{layer_id}"
    linear_weights = (
        (layer.self_attn.q_proj, f"{prefix}.attn.wq_du.weight", (None, "tensor"), False),
        (layer.self_attn.k_proj, f"{prefix}.attn.wk_dv.weight", (None, "tensor"), True),
        (layer.self_attn.v_proj, f"{prefix}.attn.wv_dv.weight", (None, "tensor"), True),
        (layer.self_attn.r_proj, f"{prefix}.attn.wr_du.weight", (None, "tensor"), False),
        (layer.self_attn.o_proj, f"{prefix}.attn.wo_ud.weight", ("tensor", None), False),
    )
    for owner, tensor_name, spec, replicate_kv_heads in linear_weights:
        weight = repository.read_tensor(tensor_name).T
        if replicate_kv_heads:
            weight = replicate_kv_head_blocks(
                weight,
                target_heads=owner.weight.value.shape[1] // layer.self_attn.head_dim,
                head_dim=layer.self_attn.head_dim,
                axis=1,
            )
        assign(owner, "weight", weight, NamedSharding(mesh, P(*spec)))
        del weight
        gc.collect()

    vector_weights = (
        (layer.self_attn.q_norm, "scale", f"{prefix}.attn.q_norm.weight"),
        (layer.self_attn.k_norm, "scale", f"{prefix}.attn.k_norm.weight"),
        (layer.input_layernorm, "scale", f"{prefix}.attn_norm.weight"),
        (layer.post_attention_layernorm, "scale", f"{prefix}.mlp_norm.weight"),
    )
    for owner, name, tensor_name in vector_weights:
        assign(
            owner,
            name,
            repository.read_tensor(tensor_name),
            NamedSharding(mesh, P(None)),
        )

    convolution_weights = (
        (layer.self_attn.k_sconv, f"{prefix}.attn.k_sconv.weight", True),
        (layer.self_attn.v_sconv, f"{prefix}.attn.v_sconv.weight", True),
        (layer.attn_sconv, f"{prefix}.attn_sconv.weight", False),
        (layer.mlp_sconv, f"{prefix}.mlp_sconv.weight", False),
    )
    for owner, tensor_name, replicate_kv_heads in convolution_weights:
        weight = repository.read_tensor(tensor_name).reshape(-1, owner.kernel_size)
        if replicate_kv_heads:
            weight = replicate_kv_head_blocks(
                weight,
                target_heads=owner.hidden_size // layer.self_attn.head_dim,
                head_dim=layer.self_attn.head_dim,
                axis=0,
            )
        assign(
            owner,
            "weight",
            weight,
            NamedSharding(mesh, P("tensor", None)),
        )

    assign(
        layer.self_attn,
        "relative_projection",
        repository.read_tensor(f"{prefix}.attn.rel_logits_proj.proj"),
        NamedSharding(mesh, P(None, None)),
    )


def load_dense_layer(
    layer: InklingDecoderLayer,
    repository: HuggingFaceSafetensorsRepository,
    layer_id: int,
    mesh: jax.sharding.Mesh,
) -> None:
    load_attention_layer(layer, repository, layer_id, mesh)
    prefix = f"model.llm.layers.{layer_id}"
    for owner, tensor_name, spec in (
        (layer.mlp.w13, f"{prefix}.mlp.w13_dn.weight", (None, "tensor")),
        (layer.mlp.w2, f"{prefix}.mlp.w2_md.weight", ("tensor", None)),
    ):
        weight = repository.read_tensor(tensor_name).T
        assign(owner, "weight", weight, NamedSharding(mesh, P(*spec)))
        del weight
        gc.collect()
    assign(
        layer.mlp,
        "global_scale",
        repository.read_tensor(f"{prefix}.mlp.global_scale"),
        NamedSharding(mesh, P()),
    )


def load_sparse_router_layer(
    layer: InklingDecoderLayer,
    repository: HuggingFaceSafetensorsRepository,
    layer_id: int,
    mesh: jax.sharding.Mesh,
) -> None:
    load_attention_layer(layer, repository, layer_id, mesh)
    prefix = f"model.llm.layers.{layer_id}.mlp.gate"
    assign(
        layer.mlp.gate,
        "kernel",
        repository.read_tensor(f"{prefix}.weight").T,
        NamedSharding(mesh, P(None, None)),
    )
    assign(
        layer.mlp,
        "correction_bias",
        repository.read_tensor(f"{prefix}.bias"),
        NamedSharding(mesh, P(None)),
    )
    assign(
        layer.mlp,
        "global_scale",
        repository.read_tensor(f"{prefix}.global_scale"),
        NamedSharding(mesh, P()),
    )
    assign(
        layer.mlp.shared_experts,
        "w13",
        repository.read_tensor(
            f"model.llm.layers.{layer_id}.mlp.shared_experts.shared_w13_weight"
        ).transpose(0, 2, 1),
        NamedSharding(mesh, P(None, None, "tensor")),
    )
    assign(
        layer.mlp.shared_experts,
        "w2",
        repository.read_tensor(
            f"model.llm.layers.{layer_id}.mlp.shared_experts.shared_w2_weight"
        ).transpose(0, 2, 1),
        NamedSharding(mesh, P(None, "tensor", None)),
    )


def streamed_routed_experts(
    checkpoint: InklingCheckpoint,
    layer_id: int,
    hidden_states: jax.Array,
    routes: jax.Array,
    weights: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    output = jnp.zeros_like(hidden_states)
    route_array = np.asarray(routes)
    for expert_id in np.unique(route_array):
        token_indices, route_slots = np.where(route_array == expert_id)
        expert = checkpoint.load_expert(layer_id, int(expert_id))
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
        selected = hidden_states.at[token_indices].get(
            out_sharding=NamedSharding(mesh, P("data", None))
        )
        selected_weights = weights.at[(token_indices, route_slots)].get(
            out_sharding=NamedSharding(mesh, P("data"))
        )
        device_w13 = jax.device_put(w13, NamedSharding(mesh, P("tensor", None)))
        device_w2 = jax.device_put(w2, NamedSharding(mesh, P(None, "tensor")))
        gate_up = jnp.einsum(
            "th,ih->ti",
            selected,
            device_w13,
            out_sharding=NamedSharding(mesh, P("data", "tensor")),
        )
        gate, up = jnp.split(gate_up, 2, axis=-1)
        expert_output = jnp.einsum(
            "ti,hi->th",
            jax.nn.silu(gate) * up,
            device_w2,
            out_sharding=NamedSharding(mesh, P("data", None)),
        )
        contribution = (expert_output * selected_weights[:, None]).astype(output.dtype)
        output = output.at[token_indices].add(
            contribution,
            out_sharding=NamedSharding(mesh, P("data", None)),
        )
        output.block_until_ready()
        del device_w13, device_w2, gate_up, gate, up, expert_output, contribution
        gc.collect()
    return output


def initial_hidden_states(
    repository: HuggingFaceSafetensorsRepository,
    prompt: str,
    mesh: jax.sharding.Mesh,
) -> tuple[np.ndarray, jax.Array]:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    input_ids = np.asarray(
        tokenizer.encode(prompt, add_special_tokens=False).ids, dtype=np.int32
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
    return input_ids, jax.device_put(
        hidden.astype(jnp.bfloat16), NamedSharding(mesh, P("data", None))
    )


def load_reference_hidden(reference_directory: Path, layer_id: int) -> np.ndarray:
    hidden = np.load(reference_directory / f"layer_{layer_id:02d}_hidden.npy")
    if hidden.dtype == np.dtype("V2"):
        hidden = hidden.view(ml_dtypes.bfloat16)
    return hidden


def run_layer(
    args: argparse.Namespace,
    config: InklingConfig,
    repository: HuggingFaceSafetensorsRepository,
    expert_checkpoint: InklingCheckpoint,
    layer_id: int,
    input_ids: np.ndarray,
    hidden_states: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[jax.Array, dict[str, object]]:
    layer = nnx.eval_shape(
        lambda: InklingDecoderLayer(
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

    token_count = hidden_states.shape[0]
    pool = MHATokenToKVPool(
        size=token_count + 1,
        page_size=1,
        dtype=jnp.bfloat16,
        head_num=layer.self_attn.num_kv_heads,
        head_dim=layer.self_attn.head_dim,
        layer_num=1,
        start_layer=layer_id,
        end_layer=layer_id,
        mesh=mesh,
    )
    locations = jnp.arange(1, token_count + 1, dtype=jnp.int32)
    batch = ForwardBatch(
        bid=0,
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=jnp.asarray(input_ids),
        req_pool_indices=jnp.asarray([0], dtype=jnp.int32),
        seq_lens=jnp.asarray([token_count], dtype=jnp.int32),
        out_cache_loc=locations,
        positions=jnp.arange(token_count, dtype=jnp.int32),
        attn_backend=NativeAttention(
            layer.self_attn.num_heads,
            layer.self_attn.num_kv_heads,
            mesh,
        ),
        cache_loc=locations,
        extend_prefix_lens=jnp.asarray([0], dtype=jnp.int32),
        extend_seq_lens=jnp.asarray([token_count], dtype=jnp.int32),
    )
    if layer_id >= 2:
        if args.route_reference_directory is None:
            raise ValueError(f"INKLING_ROUTE_REFERENCE_REQUIRED layer={layer_id}")
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attended, _, _ = layer.self_attn(batch.positions, normalized, batch, pool)
        hidden_states = residual + layer.attn_sconv(attended, batch)
        normalized = layer.post_attention_layernorm(hidden_states)
        router_logits = layer.mlp.gate(normalized)
        routed_logits = router_logits[:, : layer.mlp.num_routed_experts]
        choice_scores = np.asarray(
            jax.nn.sigmoid(routed_logits.astype(jnp.float32)).astype(
                router_logits.dtype
            )
            + layer.mlp.correction_bias.value.astype(router_logits.dtype),
            dtype=np.float32,
        )
        routes, routed_weights, shared_weights = layer.mlp.routing_weights(normalized)
        candidate_routes = np.asarray(routes)
        reference_routes = np.asarray(
            json.loads(
                (
                    args.route_reference_directory / f"layer_{layer_id:02d}_routes.json"
                ).read_text()
            )
        )
        result = {
            "backend": jax.default_backend(),
            "candidate_routes": candidate_routes.tolist(),
            "event": "INKLING_NATIVE_ROUTER_PARITY",
            "layer": layer_id,
            "reference_routes": reference_routes.tolist(),
            "routed_weight_sum": float(np.asarray(routed_weights).sum()),
            "same_routes": bool(np.array_equal(candidate_routes, reference_routes)),
            "same_route_sets": bool(
                np.array_equal(
                    np.sort(candidate_routes, axis=-1),
                    np.sort(reference_routes, axis=-1),
                )
            ),
            "shared_weight_sum": float(np.asarray(shared_weights).sum()),
        }
        route_score_differences = []
        for row, (candidate_row, reference_row) in enumerate(
            zip(candidate_routes, reference_routes)
        ):
            candidate_set = set(map(int, candidate_row))
            reference_set = set(map(int, reference_row))
            if candidate_set == reference_set:
                continue
            differing = sorted(candidate_set.symmetric_difference(reference_set))
            cutoff = float(min(choice_scores[row, list(candidate_set)]))
            route_score_differences.append(
                {
                    "candidate_cutoff": cutoff,
                    "cutoff_ties": np.flatnonzero(
                        choice_scores[row] == cutoff
                    ).tolist(),
                    "experts": {
                        str(expert): float(choice_scores[row, expert])
                        for expert in differing
                    },
                    "row": row,
                }
            )
        result["route_score_differences"] = route_score_differences
        if args.router_only:
            return hidden_states, result

        if args.force_reference_routes:
            routes, routed_weights, shared_weights = layer.mlp.routing_weights(
                normalized,
                jnp.asarray(reference_routes, dtype=jnp.int32),
            )
            result["forced_reference_routes"] = True

        routed = streamed_routed_experts(
            expert_checkpoint,
            layer_id,
            normalized,
            routes,
            routed_weights,
            mesh,
        )
        shared = layer.mlp.shared_experts(normalized, shared_weights)
        transformed = routed + shared
        output = hidden_states + layer.mlp_sconv(transformed, batch)
        candidate = np.asarray(output, dtype=np.float32)[None, ...]
        reference = load_reference_hidden(args.reference_directory, layer_id)
        result["comparison"] = compare(reference, candidate)
        result["event"] = "INKLING_NATIVE_SPARSE_LAYER_PARITY"
        return output, result

    output, _, _, _ = layer(batch.positions, hidden_states, batch, pool)
    candidate = np.asarray(output, dtype=np.float32)[None, ...]
    reference = load_reference_hidden(args.reference_directory, layer_id)
    return output, {
        "backend": jax.default_backend(),
        "comparison": compare(reference, candidate),
        "event": "INKLING_NATIVE_LAYER_PARITY",
        "layer": layer_id,
    }


def compute_final_logits(
    repository: HuggingFaceSafetensorsRepository,
    hidden_states: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    norm_weight = jax.device_put(
        jnp.asarray(
            repository.read_tensor("model.llm.norm.weight"), dtype=jnp.bfloat16
        ),
        NamedSharding(mesh, P(None)),
    )
    last_hidden = hidden_states[-1]
    normalized = last_hidden.astype(jnp.float32) * jax.lax.rsqrt(
        jnp.mean(jnp.square(last_hidden.astype(jnp.float32))) + 1e-6
    )
    final_hidden = (normalized * norm_weight.astype(jnp.float32)).astype(
        jnp.bfloat16
    ) / 24
    logits = np.empty((UNPADDED_VOCABULARY_SIZE,), dtype=np.float32)
    chunk_size = 2048
    for start in range(0, UNPADDED_VOCABULARY_SIZE, chunk_size):
        stop = min(start + chunk_size, UNPADDED_VOCABULARY_SIZE)
        unembed = jax.device_put(
            jnp.asarray(
                repository.read_first_axis_slice(
                    "model.llm.unembed.weight", start, stop
                ),
                dtype=jnp.bfloat16,
            ),
            NamedSharding(mesh, P(None, None)),
        )
        logits[start:stop] = np.asarray(
            jnp.dot(unembed, final_hidden, precision=jax.lax.Precision.HIGHEST),
            dtype=np.float32,
        )
        if start % (chunk_size * 16) == 0:
            print(
                json.dumps(
                    {
                        "event": "INKLING_NATIVE_LOGITS_PROGRESS",
                        "processed_vocabulary": stop,
                        "vocabulary_size": UNPADDED_VOCABULARY_SIZE,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    top_indices = np.argsort(logits)[-10:][::-1]
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    top = [
        {
            "logit": float(logits[index]),
            "text": tokenizer.decode([int(index)]),
            "token_id": int(index),
        }
        for index in top_indices
    ]
    return logits, top


def main() -> None:
    args = parse_args()
    mesh = create_device_mesh(
        ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
    )
    jax.sharding.set_mesh(mesh)
    config = InklingConfig.from_dict(json.loads(CONFIG_PATH.read_text()))
    config.ep_size = 1

    logits_result = None
    with HuggingFaceSafetensorsRepository(
        NVFP4_REPOSITORY, NVFP4_REVISION
    ) as repository:
        if args.layer in (None, 0):
            input_ids, hidden_states = initial_hidden_states(
                repository, args.prompt, mesh
            )
        else:
            input_ids = np.empty((0,), dtype=np.int32)
            loaded_hidden = load_reference_hidden(
                args.reference_directory, args.layer - 1
            )[0]
            hidden_states = jax.device_put(
                loaded_hidden.astype(ml_dtypes.bfloat16),
                NamedSharding(mesh, P("data", None)),
            )

        layer_ids = (
            range(config.text_config.num_hidden_layers)
            if args.all_layers
            else [args.layer]
        )
        expert_checkpoint = InklingCheckpoint(1, args.expert_cache_directory)
        results = []
        try:
            for layer_id in layer_ids:
                hidden_states, result = run_layer(
                    args,
                    config,
                    repository,
                    expert_checkpoint,
                    layer_id,
                    input_ids,
                    hidden_states,
                    mesh,
                )
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.save_hidden_directory is not None:
                    args.save_hidden_directory.mkdir(parents=True, exist_ok=True)
                    np.save(
                        args.save_hidden_directory / f"layer_{layer_id:02d}_hidden.npy",
                        np.asarray(hidden_states, dtype=np.float32)[None, ...],
                    )
                    if "candidate_routes" in result:
                        (
                            args.save_hidden_directory
                            / f"layer_{layer_id:02d}_routes.json"
                        ).write_text(json.dumps(result["candidate_routes"]))
                input_ids = np.empty((0,), dtype=np.int32)
                gc.collect()
        finally:
            expert_checkpoint.close()

        if args.logits_output is not None:
            logits, top = compute_final_logits(repository, hidden_states, mesh)
            args.logits_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.logits_output, logits=logits, top_10=json.dumps(top)
            )
            logits_result = top

    if args.all_layers:
        comparisons = [result["comparison"] for result in results]
        sparse_results = [result for result in results if result["layer"] >= 2]
        summary = {
            "event": "INKLING_NATIVE_FULL_FORWARD_PARITY",
            "layers": len(results),
            "minimum_cosine": min(item["cosine"] for item in comparisons),
            "maximum_normalized_root_mean_square_error": max(
                item["normalized_root_mean_square_error"] for item in comparisons
            ),
            "same_route_sets": all(
                result["same_route_sets"] for result in sparse_results
            ),
            "same_routes": all(result["same_routes"] for result in sparse_results),
        }
        if logits_result is not None:
            summary["top_10"] = logits_result
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
