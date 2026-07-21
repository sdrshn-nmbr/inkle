import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp

Params = dict[str, jax.Array]


def linear(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    return jnp.einsum("...h,oh->...o", hidden_states, weight)


def rms_norm(hidden_states: jax.Array, weight: jax.Array, epsilon: float) -> jax.Array:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    normalized = hidden_states * lax.rsqrt(variance + epsilon)
    return (weight * normalized).astype(input_dtype)


def short_convolution(hidden_states: jax.Array, weight: jax.Array) -> jax.Array:
    residual = hidden_states
    channels_first = jnp.transpose(hidden_states.astype(jnp.float32), (0, 2, 1))
    kernel = weight.astype(jnp.float32)
    padding = kernel.shape[-1] - 1
    convolved = lax.conv_general_dilated(
        channels_first,
        kernel,
        window_strides=(1,),
        padding=((padding, 0),),
        feature_group_count=channels_first.shape[1],
        dimension_numbers=("NCH", "OIH", "NCH"),
    )
    return (jnp.transpose(convolved, (0, 2, 1)) + residual).astype(hidden_states.dtype)


def relative_logits(
    relative_states: jax.Array,
    projection: jax.Array,
    query_positions: jax.Array,
    key_positions: jax.Array,
) -> jax.Array:
    projected = jnp.einsum("bqhd,de->bhqe", relative_states, projection)
    distance = query_positions[:, None] - key_positions[None, :]
    gather_index = jnp.clip(distance, 0, projection.shape[-1] - 1)[None, None, :, :]
    position_bias = jnp.take_along_axis(projected, gather_index, axis=-1)
    valid = (distance >= 0) & (distance < projection.shape[-1])
    return jnp.where(valid[None, None, :, :], position_bias, 0.0)


def attention_mask(sequence_length: int, sliding_window: int | None) -> jax.Array:
    query_positions = jnp.arange(sequence_length)[:, None]
    key_positions = jnp.arange(sequence_length)[None, :]
    allowed = key_positions <= query_positions
    if sliding_window is not None:
        allowed &= key_positions > query_positions - sliding_window
    return jnp.where(allowed, 0.0, -jnp.inf)[None, None, :, :]


def attention(
    params: Params,
    hidden_states: jax.Array,
    layer_id: int,
    config: dict[str, object],
) -> jax.Array:
    prefix = f"model.layers.{layer_id}.self_attn"
    is_sliding = layer_id in config["local_layer_ids"]
    head_dim = int(config["swa_head_dim"] if is_sliding else config["head_dim"])
    num_heads = int(config["swa_num_attention_heads"] if is_sliding else config["num_attention_heads"])
    num_key_value_heads = int(config["swa_num_key_value_heads"] if is_sliding else config["num_key_value_heads"])
    batch_size, sequence_length, _ = hidden_states.shape

    query = linear(hidden_states, params[f"{prefix}.q_proj.weight"])
    key = short_convolution(
        linear(hidden_states, params[f"{prefix}.k_proj.weight"]),
        params[f"{prefix}.k_sconv.conv1d.weight"],
    )
    value = short_convolution(
        linear(hidden_states, params[f"{prefix}.v_proj.weight"]),
        params[f"{prefix}.v_sconv.conv1d.weight"],
    )
    relative_states = linear(hidden_states, params[f"{prefix}.r_proj.weight"])

    query = query.reshape(batch_size, sequence_length, num_heads, head_dim)
    key = key.reshape(batch_size, sequence_length, num_key_value_heads, head_dim)
    value = value.reshape(batch_size, sequence_length, num_key_value_heads, head_dim)
    query = jnp.transpose(
        rms_norm(query, params[f"{prefix}.q_norm.weight"], float(config["rms_norm_eps"])), (0, 2, 1, 3)
    )
    key = jnp.transpose(rms_norm(key, params[f"{prefix}.k_norm.weight"], float(config["rms_norm_eps"])), (0, 2, 1, 3))
    value = jnp.transpose(value, (0, 2, 1, 3))

    repeats = num_heads // num_key_value_heads
    key = jnp.repeat(key, repeats, axis=1)
    value = jnp.repeat(value, repeats, axis=1)

    relative_states = relative_states.reshape(batch_size, sequence_length, num_heads, int(config["d_rel"]))
    positions = jnp.arange(sequence_length)
    position_bias = relative_logits(
        relative_states,
        params[f"{prefix}.rel_logits_proj.proj"],
        positions,
        positions,
    )

    if not is_sliding and config["log_scaling_n_floor"] is not None:
        effective_n = (positions + 1).astype(jnp.float32)
        floor = float(config["log_scaling_n_floor"])
        tau = 1.0 + float(config["log_scaling_alpha"]) * jnp.log(jnp.maximum(effective_n / floor, 1.0))
        tau = tau.reshape(1, 1, -1, 1)
        query = (query.astype(jnp.float32) * tau).astype(query.dtype)
        position_bias = (position_bias.astype(jnp.float32) * tau).astype(position_bias.dtype)

    scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / head_dim
    sliding_window = int(config["sliding_window_size"]) if is_sliding else None
    scores = scores + position_bias + attention_mask(sequence_length, sliding_window)
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
    attended = jnp.matmul(probabilities, value)
    attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch_size, sequence_length, -1)
    return linear(attended, params[f"{prefix}.o_proj.weight"])


def dense_mlp(params: Params, hidden_states: jax.Array, layer_id: int) -> jax.Array:
    prefix = f"model.layers.{layer_id}.mlp"
    gate = linear(hidden_states, params[f"{prefix}.gate_proj.weight"])
    up = linear(hidden_states, params[f"{prefix}.up_proj.weight"])
    down = linear(jax.nn.silu(gate) * up, params[f"{prefix}.down_proj.weight"])
    return down * params[f"{prefix}.global_scale"]


def sparse_mlp(
    params: Params,
    hidden_states: jax.Array,
    layer_id: int,
    config: dict[str, object],
) -> jax.Array:
    prefix = f"model.layers.{layer_id}.mlp"
    input_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, input_shape[-1])
    router_logits = linear(flat, params[f"{prefix}.gate.weight"])
    n_shared_experts = int(config["n_shared_experts"])
    routed_logits = router_logits[:, :-n_shared_experts]
    scores_for_choice = jax.nn.sigmoid(routed_logits) + params[f"{prefix}.gate.e_score_correction_bias"]
    _, topk_indices = lax.top_k(scores_for_choice, int(config["num_experts_per_tok"]))

    chosen_logits = jnp.take_along_axis(routed_logits, topk_indices, axis=-1)
    shared_logits = router_logits[:, -n_shared_experts:]
    topk_logits = jnp.concatenate([chosen_logits, shared_logits], axis=-1)
    topk_log_probabilities = jax.nn.log_sigmoid(topk_logits)
    topk_weights = jnp.exp(topk_log_probabilities - logsumexp(topk_log_probabilities, axis=-1, keepdims=True))
    topk_weights *= float(config["route_scale"]) * params[f"{prefix}.gate.global_scale"]
    shared_gammas = topk_weights[:, -n_shared_experts:]
    topk_weights = topk_weights[:, :-n_shared_experts]

    gate_up_weights = params[f"{prefix}.experts.gate_up_proj"][topk_indices]
    gate_up = jnp.einsum("th,tkih->tki", flat, gate_up_weights)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    activated = jax.nn.silu(gate) * up
    down_weights = params[f"{prefix}.experts.down_proj"][topk_indices]
    routed = jnp.einsum("tki,tkhi->tkh", activated, down_weights)
    routed = jnp.sum(routed * topk_weights[:, :, None], axis=1)

    shared_gate = jnp.einsum("th,eih->eti", flat, params[f"{prefix}.shared_experts.gate_proj"])
    shared_up = jnp.einsum("th,eih->eti", flat, params[f"{prefix}.shared_experts.up_proj"])
    shared_activated = jax.nn.silu(shared_gate) * shared_up
    shared_activated *= jnp.transpose(shared_gammas, (1, 0))[:, :, None]
    shared = jnp.einsum("eti,ehi->eth", shared_activated, params[f"{prefix}.shared_experts.down_proj"])
    shared = jnp.sum(shared.astype(jnp.float32), axis=0).astype(flat.dtype)
    return (routed + shared).reshape(input_shape)


def decoder_layer(
    params: Params,
    hidden_states: jax.Array,
    layer_id: int,
    config: dict[str, object],
) -> jax.Array:
    prefix = f"model.layers.{layer_id}"
    residual = hidden_states
    normalized = rms_norm(hidden_states, params[f"{prefix}.input_layernorm.weight"], float(config["rms_norm_eps"]))
    attended = attention(params, normalized, layer_id, config)
    attended = short_convolution(attended, params[f"{prefix}.attn_sconv.conv1d.weight"])
    hidden_states = residual + attended

    residual = hidden_states
    normalized = rms_norm(
        hidden_states,
        params[f"{prefix}.post_attention_layernorm.weight"],
        float(config["rms_norm_eps"]),
    )
    if config["mlp_layer_types"][layer_id] == "sparse":
        transformed = sparse_mlp(params, normalized, layer_id, config)
    else:
        transformed = dense_mlp(params, normalized, layer_id)
    transformed = short_convolution(transformed, params[f"{prefix}.mlp_sconv.conv1d.weight"])
    return residual + transformed


def forward(
    params: Params,
    input_ids: jax.Array,
    config: dict[str, object],
) -> jax.Array:
    hidden_states = params["model.embed_tokens.weight"][input_ids]
    hidden_states = rms_norm(hidden_states, params["model.embed_norm.weight"], float(config["rms_norm_eps"]))
    for layer_id in range(int(config["num_hidden_layers"])):
        hidden_states = decoder_layer(params, hidden_states, layer_id, config)
    hidden_states = rms_norm(hidden_states, params["model.norm.weight"], float(config["rms_norm_eps"]))
    hidden_states /= float(config["logits_mup_width_multiplier"])
    logits = linear(hidden_states, params["lm_head.weight"])
    return logits[..., : int(config["unpadded_vocab_size"])]


def load_bundle(path: Path) -> tuple[Params, jax.Array, np.ndarray, dict[str, object]]:
    with np.load(path) as bundle:
        metadata = json.loads(bundle["metadata"].item())
        params = {
            name.removeprefix("state::"): jnp.asarray(bundle[name])
            for name in bundle.files
            if name.startswith("state::")
        }
        input_ids = jnp.asarray(bundle["input_ids"])
        if not np.all(bundle["attention_mask"] == 1):
            raise ValueError("the tiny JAX reference currently requires an unpadded input")
        reference_logits = np.asarray(bundle["reference_logits"])
    return params, input_ids, reference_logits, metadata["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()

    params, input_ids, reference_logits, config = load_bundle(args.bundle)

    def run(current_params: Params, current_input_ids: jax.Array) -> jax.Array:
        return forward(current_params, current_input_ids, config)

    execute = run if args.no_jit else jax.jit(run)
    logits = np.asarray(execute(params, input_ids))

    absolute_error = np.abs(logits - reference_logits)
    result = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "jit": not args.no_jit,
        "logits_shape": list(logits.shape),
        "logits_checksum": logits.astype(np.float64).sum().item(),
        "max_absolute_error": absolute_error.max().item(),
        "mean_absolute_error": absolute_error.mean().item(),
        "matches_reference": bool(np.allclose(logits, reference_logits, rtol=1e-4, atol=1e-6)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["matches_reference"]:
        raise RuntimeError(
            f"JAX logits do not match the PyTorch reference: max_absolute_error={result['max_absolute_error']}"
        )


if __name__ == "__main__":
    main()
