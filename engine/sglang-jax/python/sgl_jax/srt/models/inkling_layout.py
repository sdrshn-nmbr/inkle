import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

NVFP4_BLOCK_SIZE = 16
E2M1_VALUES = np.asarray(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def unpack_nvfp4_numpy(packed_weight: np.ndarray) -> np.ndarray:
    if packed_weight.dtype != np.uint8:
        raise ValueError(f"INKLING_INVALID_NVFP4_DTYPE dtype={packed_weight.dtype}")
    unpacked = np.empty(
        (*packed_weight.shape[:-1], packed_weight.shape[-1] * 2), dtype=np.uint8
    )
    unpacked[..., 0::2] = packed_weight & 0x0F
    unpacked[..., 1::2] = packed_weight >> 4
    return unpacked


def decode_nvfp4_numpy(
    packed_weight: np.ndarray,
    block_scale: np.ndarray,
    global_scale: np.ndarray,
) -> np.ndarray:
    values = E2M1_VALUES[unpack_nvfp4_numpy(packed_weight)]
    if values.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError(f"INKLING_INVALID_NVFP4_WIDTH width={values.shape[-1]}")
    expected_scale_shape = (*values.shape[:-1], values.shape[-1] // NVFP4_BLOCK_SIZE)
    if block_scale.shape != expected_scale_shape:
        raise ValueError(
            "INKLING_INVALID_NVFP4_SCALE_SHAPE "
            f"expected={expected_scale_shape} actual={block_scale.shape}"
        )
    blocked = values.reshape(*values.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    global_scale = np.asarray(global_scale, dtype=np.float32)
    global_scale = global_scale.reshape(
        *global_scale.shape, *((1,) * (blocked.ndim - global_scale.ndim))
    )
    decoded = blocked * block_scale.astype(np.float32)[..., None] * global_scale
    return decoded.reshape(values.shape).astype(ml_dtypes.bfloat16)


def decode_nvfp4_jax(
    packed_weight: jax.Array,
    block_scale: jax.Array,
    global_scale: jax.Array,
) -> jax.Array:
    low = packed_weight & 0x0F
    high = packed_weight >> 4
    nibbles = jnp.stack((low, high), axis=-1).reshape(
        *packed_weight.shape[:-1], packed_weight.shape[-1] * 2
    )
    values = jnp.asarray(E2M1_VALUES)[nibbles]
    blocked = values.reshape(*values.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    scale = global_scale.astype(jnp.float32)
    scale = scale.reshape(*scale.shape, *((1,) * (blocked.ndim - scale.ndim)))
    decoded = blocked * block_scale.astype(jnp.float32)[..., None] * scale
    return decoded.reshape(values.shape).astype(jnp.bfloat16)


__all__ = [
    "E2M1_VALUES",
    "NVFP4_BLOCK_SIZE",
    "decode_nvfp4_jax",
    "decode_nvfp4_numpy",
    "unpack_nvfp4_numpy",
]
