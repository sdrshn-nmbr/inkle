import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessor


def test_select_hidden_states_preserves_tensor_sharding():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    states = jax.device_put(
        jnp.arange(32, dtype=jnp.float32).reshape(4, 8),
        NamedSharding(mesh, P("data", "tensor")),
    )
    indices = jax.device_put(
        jnp.asarray([0, 2], dtype=jnp.int32),
        NamedSharding(mesh, P("data")),
    )

    with jax.set_mesh(mesh):
        output = jax.jit(LogitsProcessor(8, mesh)._select_hidden_states)(states, indices)

    np.testing.assert_array_equal(np.asarray(output), np.asarray(states)[[0, 2]])
    assert output.sharding.spec == P("data", "tensor")
