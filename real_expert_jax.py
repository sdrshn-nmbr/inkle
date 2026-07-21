# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax==0.10.2",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
# ]
# ///

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from inkling_layout import deinterleave_gate_up_numpy

HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 3072
W13_SHAPE = (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
W2_SHAPE = (HIDDEN_SIZE, INTERMEDIATE_SIZE)


def load_bfloat16(path: Path, shape: tuple[int, int]) -> np.ndarray:
    expected_bytes = int(np.prod(shape)) * 2
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{path} has {actual_bytes} bytes; expected {expected_bytes}")
    return np.fromfile(path, dtype=ml_dtypes.bfloat16).reshape(shape)


def expert_forward(x: jax.Array, w13: jax.Array, w2: jax.Array) -> jax.Array:
    gate_up = x @ w13.T
    gate, up = jnp.split(gate_up, 2, axis=-1)
    activated = jax.nn.silu(gate) * up
    return activated @ w2.T


def quantize_weight(weight: np.ndarray, bits: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    qmax = (1 << (bits - 1)) - 1
    output_size, input_size = weight.shape
    if input_size % group_size != 0:
        raise ValueError(f"Input size {input_size} is not divisible by group size {group_size}")
    weight_float32 = weight.astype(np.float32).reshape(output_size, input_size // group_size, group_size)
    scale = np.max(np.abs(weight_float32), axis=2) / qmax
    scale = np.maximum(scale, np.finfo(np.float32).tiny)
    quantized = np.clip(np.rint(weight_float32 / scale[:, :, None]), -qmax, qmax).astype(np.int8)
    return quantized.reshape(weight.shape), scale.astype(np.float32)


def quantized_linear(
    x: jax.Array,
    quantized_weight: jax.Array,
    weight_scale: jax.Array,
    bits: int,
) -> jax.Array:
    qmax = (1 << (bits - 1)) - 1
    output_size, input_size = quantized_weight.shape
    num_groups = weight_scale.shape[1]
    group_size = input_size // num_groups
    x_float32 = x.astype(jnp.float32)
    grouped_input = x_float32.reshape(x.shape[0], num_groups, group_size)
    grouped_weight = quantized_weight.reshape(output_size, num_groups, group_size)
    input_scale = jnp.max(jnp.abs(grouped_input), axis=-1) / qmax
    input_scale = jnp.maximum(input_scale, jnp.finfo(jnp.float32).tiny)
    quantized_input = jnp.clip(jnp.rint(grouped_input / input_scale[:, :, None]), -qmax, qmax).astype(jnp.int8)
    accumulated = jnp.einsum(
        "tgi,ogi->tgo",
        quantized_input,
        grouped_weight,
        preferred_element_type=jnp.int32,
    )
    scaled = accumulated.astype(jnp.float32) * input_scale[:, :, None] * weight_scale.T[None, :, :]
    return scaled.sum(axis=1)


def quantized_expert_forward(
    x: jax.Array,
    quantized_w13: jax.Array,
    w13_scale: jax.Array,
    quantized_w2: jax.Array,
    w2_scale: jax.Array,
    bits: int,
) -> jax.Array:
    gate_up = quantized_linear(x, quantized_w13, w13_scale, bits)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    activated = jax.nn.silu(gate) * up
    return quantized_linear(activated, quantized_w2, w2_scale, bits)


def dequantize_weight(quantized_weight: jax.Array, weight_scale: jax.Array) -> jax.Array:
    output_size, input_size = quantized_weight.shape
    num_groups = weight_scale.shape[1]
    group_size = input_size // num_groups
    grouped_weight = quantized_weight.reshape(output_size, num_groups, group_size)
    dequantized = grouped_weight.astype(jnp.float32) * weight_scale[:, :, None]
    return dequantized.reshape(quantized_weight.shape).astype(jnp.bfloat16)


def weight_only_expert_forward(
    x: jax.Array,
    quantized_w13: jax.Array,
    w13_scale: jax.Array,
    quantized_w2: jax.Array,
    w2_scale: jax.Array,
) -> jax.Array:
    return expert_forward(
        x,
        dequantize_weight(quantized_w13, w13_scale),
        dequantize_weight(quantized_w2, w2_scale),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w13", type=Path, required=True)
    parser.add_argument("--w2", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quantization-bits", type=int, choices=(4, 8), action="append", default=[])
    parser.add_argument("--quantization-group-size", type=int, default=128)
    parser.add_argument("--save-output", type=Path)
    return parser.parse_args()


def summarize_output(output: np.ndarray) -> dict[str, object]:
    return {
        "first_8_float32": output[0, :8].tolist(),
        "max_abs_float32": float(np.max(np.abs(output))),
        "sha256_float32": hashlib.sha256(output.tobytes()).hexdigest(),
        "shape": list(output.shape),
        "sum_float32": float(output.sum()),
    }


def compare_outputs(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate - reference
    reference_norm = np.linalg.norm(reference)
    candidate_norm = np.linalg.norm(candidate)
    return {
        "cosine_similarity": float(np.vdot(reference, candidate) / (reference_norm * candidate_norm)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "normalized_root_mean_square_error": float(
            np.sqrt(np.mean(np.square(difference))) / np.sqrt(np.mean(np.square(reference)))
        ),
    }


def memory_analysis(compiled: jax.stages.Compiled) -> object:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None
    if hasattr(analysis, "_asdict"):
        return analysis._asdict()
    return str(analysis)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    host_input = rng.standard_normal((args.tokens, HIDDEN_SIZE), dtype=np.float32).astype(ml_dtypes.bfloat16)
    host_w13 = deinterleave_gate_up_numpy(load_bfloat16(args.w13, W13_SHAPE))
    host_w2 = load_bfloat16(args.w2, W2_SHAPE)

    device = jax.devices()[0]
    x = jax.device_put(host_input, device)
    w13 = jax.device_put(host_w13, device)
    w2 = jax.device_put(host_w2, device)

    compile_started = time.perf_counter()
    compiled_forward = jax.jit(expert_forward).lower(x, w13, w2).compile()
    compile_seconds = time.perf_counter() - compile_started
    first_started = time.perf_counter()
    first_output = compiled_forward(x, w13, w2)
    first_output.block_until_ready()
    first_seconds = time.perf_counter() - first_started

    cached_started = time.perf_counter()
    repeated_output = compiled_forward(x, w13, w2)
    repeated_output.block_until_ready()
    cached_seconds = time.perf_counter() - cached_started

    output = np.asarray(first_output, dtype=np.float32)
    repeated = np.asarray(repeated_output, dtype=np.float32)
    if not np.array_equal(output, repeated):
        raise RuntimeError("Repeated execution produced a different result")
    if not np.isfinite(output).all():
        raise RuntimeError("Expert output contains a non-finite value")

    result: dict[str, object] = {
        "backend": jax.default_backend(),
        "cached_seconds": cached_seconds,
        "device": str(device),
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_seconds,
        "input_shape": list(host_input.shape),
        "jax_version": jax.__version__,
        "memory_analysis": memory_analysis(compiled_forward),
        "output": summarize_output(output),
        "repeated_output_exact": True,
        "w13_shape": list(host_w13.shape),
        "w2_shape": list(host_w2.shape),
        "weight_bytes": host_w13.nbytes + host_w2.nbytes,
    }

    quantization_results = {}
    for bits in args.quantization_bits:
        quantized_w13, w13_scale = quantize_weight(host_w13, bits, args.quantization_group_size)
        quantized_w2, w2_scale = quantize_weight(host_w2, bits, args.quantization_group_size)
        integer_compile_started = time.perf_counter()
        compiled_quantized_forward = (
            jax.jit(quantized_expert_forward, static_argnames=("bits",))
            .lower(
                x,
                jax.device_put(quantized_w13, device),
                jax.device_put(w13_scale, device),
                jax.device_put(quantized_w2, device),
                jax.device_put(w2_scale, device),
                bits,
            )
            .compile()
        )
        integer_compile_seconds = time.perf_counter() - integer_compile_started
        integer_started = time.perf_counter()
        quantized_output = compiled_quantized_forward(
            x,
            jax.device_put(quantized_w13, device),
            jax.device_put(w13_scale, device),
            jax.device_put(quantized_w2, device),
            jax.device_put(w2_scale, device),
        )
        quantized_output.block_until_ready()
        integer_seconds = time.perf_counter() - integer_started
        quantized_host_output = np.asarray(quantized_output, dtype=np.float32)

        weight_only_compile_started = time.perf_counter()
        compiled_weight_only_forward = (
            jax.jit(weight_only_expert_forward)
            .lower(
                x,
                jax.device_put(quantized_w13, device),
                jax.device_put(w13_scale, device),
                jax.device_put(quantized_w2, device),
                jax.device_put(w2_scale, device),
            )
            .compile()
        )
        weight_only_compile_seconds = time.perf_counter() - weight_only_compile_started
        weight_only_started = time.perf_counter()
        weight_only_output = compiled_weight_only_forward(
            x,
            jax.device_put(quantized_w13, device),
            jax.device_put(w13_scale, device),
            jax.device_put(quantized_w2, device),
            jax.device_put(w2_scale, device),
        )
        weight_only_output.block_until_ready()
        weight_only_seconds = time.perf_counter() - weight_only_started
        weight_only_host_output = np.asarray(weight_only_output, dtype=np.float32)
        quantization_results[str(bits)] = {
            "group_size": args.quantization_group_size,
            "integer_weights_and_activations": {
                "cached_seconds": integer_seconds,
                "compile_seconds": integer_compile_seconds,
                "comparison_to_bfloat16": compare_outputs(output, quantized_host_output),
                "memory_analysis": memory_analysis(compiled_quantized_forward),
                "output": summarize_output(quantized_host_output),
            },
            "logical_packed_weight_bytes": (quantized_w13.size + quantized_w2.size) * bits // 8,
            "stored_probe_weight_bytes": quantized_w13.nbytes + quantized_w2.nbytes,
            "weight_only_with_bfloat16_activations": {
                "cached_seconds": weight_only_seconds,
                "compile_seconds": weight_only_compile_seconds,
                "comparison_to_bfloat16": compare_outputs(output, weight_only_host_output),
                "memory_analysis": memory_analysis(compiled_weight_only_forward),
                "output": summarize_output(weight_only_host_output),
            },
        }
    result["quantization"] = quantization_results
    if args.save_output is not None:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_output, output)
        result["saved_output"] = str(args.save_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
