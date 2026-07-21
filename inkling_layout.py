import jax
import jax.numpy as jnp
import numpy as np

NVFP4_BLOCK_SIZE = 16
E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def deinterleave_gate_up_numpy(weight: np.ndarray) -> np.ndarray:
    if weight.ndim < 2 or weight.shape[-2] % 2:
        raise ValueError(f"INKLING_INVALID_GATE_UP_SHAPE shape={weight.shape}")
    gate = weight[..., 0::2, :]
    up = weight[..., 1::2, :]
    return np.concatenate((gate, up), axis=-2)


def deinterleave_gate_up_jax(weight: jax.Array) -> jax.Array:
    if weight.ndim < 2 or weight.shape[-2] % 2:
        raise ValueError(f"INKLING_INVALID_GATE_UP_SHAPE shape={weight.shape}")
    return jnp.concatenate((weight[..., 0::2, :], weight[..., 1::2, :]), axis=-2)


def unpack_nvfp4_numpy(packed_weight: np.ndarray) -> np.ndarray:
    if packed_weight.dtype != np.uint8:
        raise ValueError(f"INKLING_INVALID_NVFP4_DTYPE dtype={packed_weight.dtype}")
    unpacked = np.empty((*packed_weight.shape[:-1], packed_weight.shape[-1] * 2), dtype=np.uint8)
    unpacked[..., 0::2] = packed_weight & 0x0F
    unpacked[..., 1::2] = packed_weight >> 4
    return unpacked


def decode_nvfp4_numpy(
    packed_weight: np.ndarray,
    block_scale: np.ndarray,
    global_scale: np.ndarray | np.generic | float,
) -> np.ndarray:
    values = E2M1_VALUES[unpack_nvfp4_numpy(packed_weight)]
    if values.shape[-1] % NVFP4_BLOCK_SIZE:
        raise ValueError(f"INKLING_INVALID_NVFP4_WIDTH width={values.shape[-1]}")
    expected_scale_shape = (*values.shape[:-1], values.shape[-1] // NVFP4_BLOCK_SIZE)
    if block_scale.shape != expected_scale_shape:
        raise ValueError(
            f"INKLING_INVALID_NVFP4_SCALE_SHAPE expected={expected_scale_shape} actual={block_scale.shape}"
        )
    blocked = values.reshape(*values.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    decoded = blocked * block_scale.astype(np.float32)[..., None] * np.asarray(global_scale, dtype=np.float32)
    return decoded.reshape(values.shape)


def decode_nvfp4_jax(
    packed_weight: jax.Array,
    block_scale: jax.Array,
    global_scale: jax.Array,
) -> jax.Array:
    low = packed_weight & 0x0F
    high = packed_weight >> 4
    nibbles = jnp.stack((low, high), axis=-1).reshape(*packed_weight.shape[:-1], packed_weight.shape[-1] * 2)
    values = jnp.asarray(E2M1_VALUES)[nibbles]
    blocked = values.reshape(*values.shape[:-1], -1, NVFP4_BLOCK_SIZE)
    decoded = blocked * block_scale.astype(jnp.float32)[..., None] * global_scale
    return decoded.reshape(values.shape).astype(jnp.bfloat16)
