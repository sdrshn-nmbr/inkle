# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jax==0.10.2",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
# ]
# ///

import argparse
import json
import time
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
from inkling_layout import decode_nvfp4_jax, deinterleave_gate_up_numpy
from real_expert_jax import (
    HIDDEN_SIZE,
    W2_SHAPE,
    W13_SHAPE,
    compare_outputs,
    expert_forward,
    load_bfloat16,
    memory_analysis,
    summarize_output,
)

BLOCK_SIZE = 16
PACKED_W13_SHAPE = (W13_SHAPE[0], W13_SHAPE[1] // 2)
PACKED_W2_SHAPE = (W2_SHAPE[0], W2_SHAPE[1] // 2)
W13_SCALE_SHAPE = (W13_SHAPE[0], W13_SHAPE[1] // BLOCK_SIZE)
W2_SCALE_SHAPE = (W2_SHAPE[0], W2_SHAPE[1] // BLOCK_SIZE)


def load_array(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    expected_elements = int(np.prod(shape))
    if array.size != expected_elements:
        raise ValueError(f"{path} has {array.size} elements; expected {expected_elements}")
    return array.reshape(shape)


def nvfp4_expert_forward(
    x: jax.Array,
    packed_w13: jax.Array,
    w13_scale: jax.Array,
    w13_scale2: jax.Array,
    packed_w2: jax.Array,
    w2_scale: jax.Array,
    w2_scale2: jax.Array,
) -> jax.Array:
    return expert_forward(
        x,
        decode_nvfp4_jax(packed_w13, w13_scale, w13_scale2),
        decode_nvfp4_jax(packed_w2, w2_scale, w2_scale2),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-w13", type=Path, required=True)
    parser.add_argument("--bf16-w2", type=Path, required=True)
    parser.add_argument("--packed-w13", type=Path, required=True)
    parser.add_argument("--w13-scale", type=Path, required=True)
    parser.add_argument("--w13-scale2", type=Path, required=True)
    parser.add_argument("--packed-w2", type=Path, required=True)
    parser.add_argument("--w2-scale", type=Path, required=True)
    parser.add_argument("--w2-scale2", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    host_input = rng.standard_normal((args.tokens, HIDDEN_SIZE), dtype=np.float32).astype(ml_dtypes.bfloat16)
    host_bf16_w13 = deinterleave_gate_up_numpy(load_bfloat16(args.bf16_w13, W13_SHAPE))
    host_bf16_w2 = load_bfloat16(args.bf16_w2, W2_SHAPE)
    host_packed_w13 = deinterleave_gate_up_numpy(load_array(args.packed_w13, np.dtype(np.uint8), PACKED_W13_SHAPE))
    host_w13_scale = deinterleave_gate_up_numpy(load_array(args.w13_scale, ml_dtypes.float8_e4m3fn, W13_SCALE_SHAPE))
    host_w13_scale2 = load_array(args.w13_scale2, np.dtype(np.float32), ())
    host_packed_w2 = load_array(args.packed_w2, np.dtype(np.uint8), PACKED_W2_SHAPE)
    host_w2_scale = load_array(args.w2_scale, ml_dtypes.float8_e4m3fn, W2_SCALE_SHAPE)
    host_w2_scale2 = load_array(args.w2_scale2, np.dtype(np.float32), ())

    device = jax.devices()[0]
    x = jax.device_put(host_input, device)
    bf16_transfer_started = time.perf_counter()
    bf16_arguments = (
        x,
        jax.device_put(host_bf16_w13, device),
        jax.device_put(host_bf16_w2, device),
    )
    jax.block_until_ready(bf16_arguments)
    bf16_transfer_seconds = time.perf_counter() - bf16_transfer_started
    nvfp4_transfer_started = time.perf_counter()
    nvfp4_arguments = (
        x,
        jax.device_put(host_packed_w13, device),
        jax.device_put(host_w13_scale, device),
        jax.device_put(host_w13_scale2, device),
        jax.device_put(host_packed_w2, device),
        jax.device_put(host_w2_scale, device),
        jax.device_put(host_w2_scale2, device),
    )
    jax.block_until_ready(nvfp4_arguments)
    nvfp4_transfer_seconds = time.perf_counter() - nvfp4_transfer_started

    bf16_compiled = jax.jit(expert_forward).lower(*bf16_arguments).compile()
    nvfp4_compile_started = time.perf_counter()
    nvfp4_compiled = jax.jit(nvfp4_expert_forward).lower(*nvfp4_arguments).compile()
    nvfp4_compile_seconds = time.perf_counter() - nvfp4_compile_started

    bf16_output = bf16_compiled(*bf16_arguments)
    bf16_output.block_until_ready()
    nvfp4_started = time.perf_counter()
    nvfp4_output = nvfp4_compiled(*nvfp4_arguments)
    nvfp4_output.block_until_ready()
    nvfp4_seconds = time.perf_counter() - nvfp4_started

    bf16_host_output = np.asarray(bf16_output, dtype=np.float32)
    nvfp4_host_output = np.asarray(nvfp4_output, dtype=np.float32)
    nvfp4_stored_bytes = sum(
        array.nbytes
        for array in (
            host_packed_w13,
            host_w13_scale,
            host_w13_scale2,
            host_packed_w2,
            host_w2_scale,
            host_w2_scale2,
        )
    )

    result = {
        "backend": jax.default_backend(),
        "bfloat16_output": summarize_output(bf16_host_output),
        "bfloat16_transfer_seconds": bf16_transfer_seconds,
        "bfloat16_weight_bytes": host_bf16_w13.nbytes + host_bf16_w2.nbytes,
        "comparison_to_bfloat16": compare_outputs(bf16_host_output, nvfp4_host_output),
        "device": str(device),
        "nvfp4_cached_seconds": nvfp4_seconds,
        "nvfp4_compile_seconds": nvfp4_compile_seconds,
        "nvfp4_memory_analysis": memory_analysis(nvfp4_compiled),
        "nvfp4_output": summarize_output(nvfp4_host_output),
        "nvfp4_stored_bytes": nvfp4_stored_bytes,
        "nvfp4_transfer_seconds": nvfp4_transfer_seconds,
        "tokens": args.tokens,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
