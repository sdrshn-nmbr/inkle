import logging

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import nnx
from jax.scipy.special import logsumexp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.inkling_layout import decode_nvfp4_numpy
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.weight_utils import (
    SequentialSafetensorManager,
    WeightLoader,
    WeightMapping,
)

logger = logging.getLogger(__name__)


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", config)


def _slice_bounds(index: slice, size: int) -> tuple[int, int]:
    start, stop, step = index.indices(size)
    if step != 1:
        raise ValueError(f"INKLING_UNSUPPORTED_SHARD_STEP step={step}")
    return start, stop


def gather_fused_moe_tokens(
    hidden_states: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    return jax.shard_map(
        lambda x: jax.lax.all_gather(x, "tensor", axis=0, tiled=True),
        mesh=mesh,
        in_specs=P(("data", "tensor"), None),
        out_specs=P("data", None),
        check_vma=False,
    )(hidden_states)


def split_interleaved_gate_up(
    gate_up: jax.Array,
    mesh: jax.sharding.Mesh,
) -> tuple[jax.Array, jax.Array]:
    if gate_up.shape[-1] % 2:
        raise ValueError(f"INKLING_INVALID_GATE_UP_SHAPE shape={gate_up.shape}")
    middle_axes = (None,) * (gate_up.ndim - 2)
    paired = gate_up.reshape(
        *gate_up.shape[:-1],
        gate_up.shape[-1] // 2,
        2,
        out_sharding=NamedSharding(mesh, P("data", *middle_axes, "tensor", None)),
    )
    output_sharding = NamedSharding(mesh, P("data", *middle_axes, "tensor"))
    gate = paired.at[..., 0].get(out_sharding=output_sharding)
    up = paired.at[..., 1].get(out_sharding=output_sharding)
    return gate, up


def get_recurrent_state_row(
    state_table: jax.Array,
    row: int,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
    return state_table.at[row].get(out_sharding=NamedSharding(mesh, P("tensor", None)))


def stable_bfloat16_linear(
    linear: LinearBase,
    hidden_states: jax.Array,
) -> jax.Array:
    output, _ = linear(
        hidden_states,
        preferred_element_type=jnp.float32,
    )
    return output.astype(hidden_states.dtype)


class InklingShortConvolution(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        mesh: jax.sharding.Mesh,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nnx.Param(
            jnp.zeros(
                (hidden_size, kernel_size),
                dtype=jnp.float32,
                out_sharding=P("tensor", None),
            )
        )
        self.mesh = mesh

    def __call__(self, hidden_states: jax.Array, forward_batch: ForwardBatch) -> jax.Array:
        return self.apply(hidden_states, forward_batch, None)[0]

    def apply(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        state_table: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array | None]:
        if state_table is None and forward_batch.forward_mode != ForwardMode.EXTEND:
            raise ValueError(
                "INKLING_CONV_CACHE_REQUIRED native decode needs persistent short-convolution state"
            )
        if state_table is None:
            sequence_lengths = forward_batch.extend_seq_lens
            cache = jnp.zeros(
                (sequence_lengths.shape[0], self.hidden_size, self.kernel_size - 1),
                dtype=jnp.float32,
                out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            )
            state_indices = None
        else:
            if forward_batch.recurrent_indices is None:
                raise ValueError(
                    "INKLING_RECURRENT_INDICES_REQUIRED convolution state needs request slots"
                )
            state_sharding = NamedSharding(self.mesh, P("data", "tensor", None))
            state_table = jax.sharding.reshard(state_table, state_sharding)
            state_indices = jax.sharding.reshard(
                forward_batch.recurrent_indices,
                NamedSharding(self.mesh, P("data")),
            )
            cache = (
                state_table.at[state_indices].get(out_sharding=state_sharding).astype(jnp.float32)
            )
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                prefix_lengths = forward_batch.extend_prefix_lens
                if prefix_lengths is None:
                    prefix_lengths = jnp.zeros_like(forward_batch.extend_seq_lens)
                prefix_lengths = jax.sharding.reshard(
                    prefix_lengths,
                    NamedSharding(self.mesh, P("data")),
                )
                cache = jnp.where(
                    prefix_lengths[:, None, None] > 0,
                    cache,
                    jnp.zeros_like(cache),
                )
            sequence_lengths = (
                forward_batch.extend_seq_lens
                if forward_batch.forward_mode == ForwardMode.EXTEND
                else None
            )
        cumulative_lengths = None
        if sequence_lengths is not None:
            cumulative_lengths = jnp.concatenate(
                (
                    jnp.zeros((1,), dtype=sequence_lengths.dtype),
                    jnp.cumsum(sequence_lengths, dtype=sequence_lengths.dtype),
                )
            )
        input_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        convolution_input = jax.sharding.reshard(
            hidden_states.astype(jnp.float32),
            input_sharding,
        )
        convolved, new_cache = short_convolution(
            convolution_input,
            self.weight.value.astype(jnp.float32),
            cache,
            cumulative_lengths,
            forward_batch.forward_mode,
            activation=None,
            x_window_sharding=NamedSharding(self.mesh, P("data", None, "tensor")),
            cache_window_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
            backend="pallas",
        )
        new_state_table = None
        if state_table is not None:
            safe_indices = jnp.maximum(state_indices, 0)
            new_state_table = state_table.at[safe_indices].set(
                new_cache.astype(state_table.dtype),
                out_sharding=state_sharding,
            )
            new_state_table = new_state_table.at[0].set(
                get_recurrent_state_row(state_table, 0, self.mesh),
                out_sharding=state_sharding,
            )
        output = (convolved + convolution_input).astype(hidden_states.dtype)
        output = jax.sharding.reshard(output, input_sharding)
        return output, new_state_table


class InklingDenseMLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.w13 = LinearBase(
            input_size=hidden_size,
            output_size=2 * intermediate_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="w13",
        )
        self.w2 = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="w2",
        )
        self.global_scale = nnx.Param(jnp.ones((), dtype=dtype))
        self.mesh = mesh

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate_up = stable_bfloat16_linear(self.w13, hidden_states)
        gate, up = split_interleaved_gate_up(gate_up, self.mesh)
        output = stable_bfloat16_linear(self.w2, jax.nn.silu(gate) * up)
        return output * self.global_scale.value.astype(output.dtype)


class InklingSharedExperts(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.mesh = mesh
        self.w13 = nnx.Param(
            jnp.zeros(
                (num_experts, hidden_size, 2 * intermediate_size),
                dtype=dtype,
                out_sharding=P(None, None, "tensor"),
            )
        )
        self.w2 = nnx.Param(
            jnp.zeros(
                (num_experts, intermediate_size, hidden_size),
                dtype=dtype,
                out_sharding=P(None, "tensor", None),
            )
        )

    def __call__(self, hidden_states: jax.Array, weights: jax.Array) -> jax.Array:
        token_count = hidden_states.shape[0]
        token_padding = (-token_count) % 8
        hidden_states = jnp.pad(hidden_states, ((0, token_padding), (0, 0)))
        weights = jnp.pad(weights, ((0, token_padding), (0, 0)))
        gate_up = jnp.einsum(
            "th,ehi->tei",
            hidden_states,
            self.w13.value,
            preferred_element_type=jnp.float32,
            out_sharding=NamedSharding(self.mesh, P("data", None, "tensor")),
        )
        gate, up = split_interleaved_gate_up(gate_up, self.mesh)
        activated = jax.nn.silu(gate) * up
        output = jnp.einsum(
            "tei,eih->teh",
            activated,
            self.w2.value,
            preferred_element_type=jnp.float32,
            out_sharding=NamedSharding(self.mesh, P("data", None, None)),
        )
        output = output * weights[..., None].astype(output.dtype)
        output = jnp.sum(output.astype(jnp.float32), axis=1).astype(hidden_states.dtype)
        return output.at[:token_count].get(out_sharding=NamedSharding(self.mesh, P("data", None)))


class InklingMoE(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        root_config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.num_routed_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.route_scale = float(config.route_scale)
        self.mesh = mesh
        self.use_fused_routed_experts = getattr(root_config, "moe_backend", "epmoe") == "fused_v2"
        self.gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=self.num_routed_experts + self.num_shared_experts,
            weight_dtype=dtype,
            score_func=None,
        )
        self.gate.kernel = nnx.Param(
            jnp.zeros(
                (
                    config.hidden_size,
                    self.num_routed_experts + self.num_shared_experts,
                ),
                dtype=dtype,
                out_sharding=P(None, None),
            )
        )
        self.correction_bias = nnx.Param(jnp.zeros((self.num_routed_experts,), dtype=dtype))
        self.global_scale = nnx.Param(jnp.ones((), dtype=dtype))
        self.topk = TopK(
            topk=self.num_experts_per_tok,
            renormalize=False,
            layer_id=layer_id,
            mesh=mesh,
        )
        self.experts = EPMoE(
            hidden_size=config.hidden_size,
            num_experts=self.num_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            ep_size=getattr(root_config, "ep_size", 1),
            mesh=mesh,
            intermediate_dim=config.intermediate_size,
            weight_dtype=dtype,
            dtype=dtype,
            activation="silu",
            layer_id=layer_id,
            preferred_element_type=jnp.float32,
        )
        self.shared_experts = InklingSharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=self.num_shared_experts,
            mesh=mesh,
            dtype=dtype,
        )

    def routing_weights(
        self,
        hidden_states: jax.Array,
        topk_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        router_logits = self.gate(hidden_states)
        routed_logits = router_logits[:, : self.num_routed_experts]
        if topk_ids is None:
            choice_scores = jax.nn.sigmoid(routed_logits.astype(jnp.float32)).astype(
                router_logits.dtype
            ) + self.correction_bias.value.astype(router_logits.dtype)
            _, topk_ids = self.topk(choice_scores)
        chosen_logits = jnp.take_along_axis(routed_logits, topk_ids, axis=-1)
        shared_logits = router_logits[:, self.num_routed_experts :]
        selected_logits = jnp.concatenate((chosen_logits, shared_logits), axis=-1)
        log_weights = jax.nn.log_sigmoid(selected_logits.astype(jnp.float32)).astype(
            router_logits.dtype
        )
        weights = jnp.exp(log_weights - logsumexp(log_weights, axis=-1, keepdims=True))
        weights = weights * self.route_scale * self.global_scale.value
        return (
            topk_ids,
            weights[:, : self.num_experts_per_tok],
            weights[:, self.num_experts_per_tok :],
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        out_sharding: NamedSharding,
    ) -> tuple[jax.Array, jax.Array]:
        topk_ids, routed_weights, shared_weights = self.routing_weights(hidden_states)
        if self.use_fused_routed_experts and is_tpu_runtime():
            token_count = hidden_states.shape[0]
            padded_token_count = (
                (token_count + self.mesh.size - 1) // self.mesh.size * self.mesh.size
            )
            token_padding = padded_token_count - token_count
            hidden_states_for_fused = jnp.pad(
                hidden_states,
                ((0, token_padding), (0, 0)),
            )
            routed_weights_for_fused = jnp.pad(
                routed_weights.astype(jnp.float32),
                ((0, token_padding), (0, 0)),
            )
            shared_weights_for_fused = jnp.pad(
                shared_weights.astype(jnp.float32),
                ((0, token_padding), (0, 0)),
            )
            topk_ids_for_fused = jnp.pad(
                topk_ids,
                ((0, token_padding), (0, 0)),
                constant_values=-1,
            )
            token_sharding = NamedSharding(self.mesh, P(("data", "tensor"), None))
            expert_sharding = NamedSharding(self.mesh, P(("data", "tensor"), None, None))
            replicated = NamedSharding(self.mesh, P())
            shared_w13 = jax.sharding.reshard(self.shared_experts.w13.value, replicated)
            shared_w2 = jax.sharding.reshard(self.shared_experts.w2.value, replicated)
            shared_gate = shared_w13[..., 0::2].transpose(1, 0, 2).reshape(
                self.experts.hidden_size,
                -1,
            )
            shared_up = shared_w13[..., 1::2].transpose(1, 0, 2).reshape(
                self.experts.hidden_size,
                -1,
            )
            shared_down = shared_w2.reshape(-1, self.experts.hidden_size)
            routed = fused_ep_moe_v2(
                self.mesh,
                jax.sharding.reshard(hidden_states_for_fused, token_sharding),
                jax.sharding.reshard(self.experts.wi_0.value, expert_sharding),
                jax.sharding.reshard(self.experts.wo.value, expert_sharding),
                jax.sharding.reshard(self.experts.wi_1.value, expert_sharding),
                jax.sharding.reshard(routed_weights_for_fused, token_sharding),
                jax.sharding.reshard(topk_ids_for_fused, token_sharding),
                self.num_experts_per_tok,
                act_fn="silu",
                w1_shared=shared_gate,
                w2_shared=shared_down,
                w3_shared=shared_up,
                shared_weights=jax.sharding.reshard(
                    shared_weights_for_fused, token_sharding
                ),
            )
            routed = gather_fused_moe_tokens(routed, self.mesh)
            routed = routed[:token_count]
            routed = jax.sharding.reshard(routed, out_sharding)
        else:
            routed = self.experts(
                hidden_states,
                routed_weights,
                topk_ids,
                out_sharding=out_sharding,
            )
        if self.use_fused_routed_experts and is_tpu_runtime():
            return routed, topk_ids
        shared = self.shared_experts(hidden_states, shared_weights)
        return routed + shared, topk_ids


class InklingAttention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.is_sliding = layer_id in config.local_layer_ids
        self.num_kv_heads = (
            config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        )
        self.relative_dimension = config.d_rel
        self.relative_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.log_scaling_n_floor = getattr(config, "log_scaling_n_floor", None)
        self.log_scaling_alpha = float(getattr(config, "log_scaling_alpha", 0.0))
        self.q_proj = LinearBase(
            config.hidden_size,
            self.num_heads * self.head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="v_proj",
        )
        self.r_proj = LinearBase(
            config.hidden_size,
            self.num_heads * self.relative_dimension,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="r_proj",
        )
        self.o_proj = LinearBase(
            self.num_heads * self.head_dim,
            config.hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, epsilon=config.rms_norm_eps, param_dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, epsilon=config.rms_norm_eps, param_dtype=dtype)
        self.relative_projection = nnx.Param(
            jnp.zeros(
                (self.relative_dimension, self.relative_extent),
                dtype=dtype,
                out_sharding=P(None, None),
            )
        )
        self.k_sconv = InklingShortConvolution(
            self.num_kv_heads * self.head_dim,
            config.sconv_kernel_size,
            mesh,
        )
        self.v_sconv = InklingShortConvolution(
            self.num_kv_heads * self.head_dim,
            config.sconv_kernel_size,
            mesh,
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scaling=1.0 / self.head_dim,
            layer_id=layer_id,
            sliding_window_size=config.sliding_window_size if self.is_sliding else 0,
            softmax_dtype=jnp.float32,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        convolution_states: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array] | None]:
        token_count = hidden_states.shape[0]
        query = stable_bfloat16_linear(self.q_proj, hidden_states)
        key = stable_bfloat16_linear(self.k_proj, hidden_states)
        value = stable_bfloat16_linear(self.v_proj, hidden_states)
        relative = stable_bfloat16_linear(self.r_proj, hidden_states)
        key_state = convolution_states[0] if convolution_states is not None else None
        value_state = convolution_states[1] if convolution_states is not None else None
        key, new_key_state = self.k_sconv.apply(key, forward_batch, key_state)
        value, new_value_state = self.v_sconv.apply(value, forward_batch, value_state)

        query = self.q_norm(query.reshape(token_count, self.num_heads, self.head_dim))
        key = self.k_norm(key.reshape(token_count, self.num_kv_heads, self.head_dim))
        value = value.reshape(token_count, self.num_kv_heads, self.head_dim)
        relative = relative.reshape(token_count, self.num_heads, self.relative_dimension)

        if not self.is_sliding and self.log_scaling_n_floor is not None:
            effective_n = positions.astype(jnp.float32) + 1.0
            tau = 1.0 + self.log_scaling_alpha * jnp.log(
                jnp.maximum(effective_n / float(self.log_scaling_n_floor), 1.0)
            )
            query = query * tau[:, None, None].astype(query.dtype)
            relative = relative * tau[:, None, None].astype(relative.dtype)

        attended, kv_fused = self.attn(
            query,
            key,
            value,
            forward_batch,
            token_to_kv_pool,
            relative_states=relative,
            relative_projection=self.relative_projection,
        )
        output = stable_bfloat16_linear(self.o_proj, attended)
        state_updates = None
        if convolution_states is not None:
            state_updates = (new_key_state, new_value_state)
        return output, kv_fused, state_updates


class InklingDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        root_config: PretrainedConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.is_dense = layer_id < config.dense_mlp_idx
        self.self_attn = InklingAttention(config, layer_id, mesh, dtype)
        self.input_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype
        )
        self.attn_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, mesh
        )
        self.mlp_sconv = InklingShortConvolution(config.hidden_size, config.sconv_kernel_size, mesh)
        if self.is_dense:
            self.mlp = InklingDenseMLP(
                config.hidden_size, config.dense_intermediate_size, mesh, dtype
            )
        else:
            self.mlp = InklingMoE(config, root_config, layer_id, mesh, dtype)

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        convolution_states: list[jax.Array] | None = None,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array | None,
        list[jax.Array] | None,
    ]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_states = (
            (convolution_states[0], convolution_states[1])
            if convolution_states is not None
            else None
        )
        attended, kv_fused, attention_state_updates = self.self_attn(
            positions,
            normalized,
            forward_batch,
            token_to_kv_pool,
            attention_states,
        )
        attn_state = convolution_states[2] if convolution_states is not None else None
        convolved_attention, new_attn_state = self.attn_sconv.apply(
            attended, forward_batch, attn_state
        )
        hidden_states = residual + convolved_attention

        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        topk_ids = None
        if self.is_dense:
            transformed = self.mlp(normalized)
        else:
            transformed, topk_ids = self.mlp(
                normalized,
                NamedSharding(self.mesh, P("data", None)),
            )
        mlp_state = convolution_states[3] if convolution_states is not None else None
        convolved_mlp, new_mlp_state = self.mlp_sconv.apply(transformed, forward_batch, mlp_state)
        hidden_states = residual + convolved_mlp
        state_updates = None
        if convolution_states is not None:
            state_updates = [
                attention_state_updates[0],
                attention_state_updates[1],
                new_attn_state,
                new_mlp_state,
            ]
        return hidden_states, kv_fused, topk_ids, state_updates


class InklingModel(nnx.Module):
    def __init__(
        self,
        root_config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype,
    ):
        config = _text_config(root_config)
        self.config = config
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.embed_norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype
        )
        self.layers = nnx.data(
            [
                InklingDecoderLayer(config, root_config, layer_id, mesh, dtype)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype)

    def __call__(
        self, forward_batch: ForwardBatch, memory_pools: MemoryPools
    ) -> tuple[
        jax.Array,
        dict[str, object],
        list[jax.Array | None],
    ]:
        hidden_states = self.embed_norm(self.embed_tokens(forward_batch.input_ids))
        layers_kv_fused = []
        layers_topk_ids = []
        recurrent_state_pool = getattr(memory_pools, "recurrent_state_pool", None)
        conv_updates = None
        if recurrent_state_pool is not None:
            conv_updates = [list(states) for states in recurrent_state_pool.conv_buffers]
        for layer in self.layers:
            layer_states = None
            if recurrent_state_pool is not None:
                layer_index = recurrent_state_pool.layers_mapping[layer.layer_id]
                layer_states = conv_updates[layer_index]
            hidden_states, kv_fused, topk_ids, state_updates = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                memory_pools.token_to_kv_pool,
                layer_states,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)
            if state_updates is not None:
                conv_updates[layer_index] = state_updates
        pool_updates = {"token_to_kv_pool": layers_kv_fused}
        if recurrent_state_pool is not None:
            pool_updates["recurrent_state_pool"] = (
                list(recurrent_state_pool.recurrent_buffers),
                conv_updates,
            )
        return self.norm(hidden_states), pool_updates, layers_topk_ids


class InklingForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.text_config = _text_config(config)
        self.mesh = mesh
        self.dtype = dtype
        self.model = InklingModel(config, mesh, dtype)
        self.lm_head = ParallelLMHead(
            self.text_config.vocab_size,
            self.text_config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.logits_processor = LogitsProcessor(
            self.text_config.unpadded_vocab_size,
            mesh=mesh,
            soft_cap=self.text_config.final_logit_softcapping,
            mask_padded_vocab=True,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, pool_updates, layers_topk_ids = self.model(forward_batch, memory_pools)
        hidden_states = hidden_states / float(self.text_config.logits_mup_width_multiplier)
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return (
            output,
            pool_updates,
            True,
            layers_topk_ids,
        )

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(mappings)
        self._load_experts(loader)
        logger.info(
            "INKLING_WEIGHTS_LOADED layers=%d dtype=%s",
            self.text_config.num_hidden_layers,
            self.dtype,
        )

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        fused_shared_experts = getattr(self.config, "moe_backend", "epmoe") == "fused_v2"
        shared_w13_sharding = (
            (None, None, None) if fused_shared_experts else (None, None, "tensor")
        )
        shared_w2_sharding = (
            (None, None, None) if fused_shared_experts else (None, "tensor", None)
        )
        mappings = {
            "model.llm.embed.weight": WeightMapping(
                "model.embed_tokens.embedding", sharding=("tensor", None)
            ),
            "model.llm.embed_norm.weight": WeightMapping(
                "model.embed_norm.scale", sharding=(None,)
            ),
            "model.llm.norm.weight": WeightMapping("model.norm.scale", sharding=(None,)),
            "model.llm.unembed.weight": WeightMapping(
                "lm_head.embedding", sharding=("tensor", None)
            ),
        }
        for layer_id, layer in enumerate(self.model.layers):
            source = f"model.llm.layers.{layer_id}"
            target = f"model.layers.{layer_id}"
            attention = layer.self_attn
            for source_name, target_name, sharding in (
                ("wq_du", "q_proj", (None, "tensor")),
                ("wk_dv", "k_proj", (None, "tensor")),
                ("wv_dv", "v_proj", (None, "tensor")),
                ("wr_du", "r_proj", (None, "tensor")),
                ("wo_ud", "o_proj", ("tensor", None)),
            ):
                mappings[f"{source}.attn.{source_name}.weight"] = WeightMapping(
                    f"{target}.self_attn.{target_name}.weight",
                    sharding=sharding,
                    transpose=True,
                    kv_head_padding=source_name in ("wk_dv", "wv_dv"),
                )
            mappings[f"{source}.attn.q_norm.weight"] = WeightMapping(
                f"{target}.self_attn.q_norm.scale", sharding=(None,)
            )
            mappings[f"{source}.attn.k_norm.weight"] = WeightMapping(
                f"{target}.self_attn.k_norm.scale", sharding=(None,)
            )
            mappings[f"{source}.attn.rel_logits_proj.proj"] = WeightMapping(
                f"{target}.self_attn.relative_projection", sharding=(None, None)
            )
            for conv_name, conv_target, width in (
                ("attn.k_sconv", "self_attn.k_sconv", attention.num_kv_heads * attention.head_dim),
                ("attn.v_sconv", "self_attn.v_sconv", attention.num_kv_heads * attention.head_dim),
                ("attn_sconv", "attn_sconv", self.text_config.hidden_size),
                ("mlp_sconv", "mlp_sconv", self.text_config.hidden_size),
            ):
                mappings[f"{source}.{conv_name}.weight"] = WeightMapping(
                    f"{target}.{conv_target}.weight",
                    sharding=("tensor", None),
                    reshape=(width, self.text_config.sconv_kernel_size),
                )
            mappings[f"{source}.attn_norm.weight"] = WeightMapping(
                f"{target}.input_layernorm.scale", sharding=(None,)
            )
            mappings[f"{source}.mlp_norm.weight"] = WeightMapping(
                f"{target}.post_attention_layernorm.scale", sharding=(None,)
            )
            if layer.is_dense:
                mappings[f"{source}.mlp.w13_dn.weight"] = WeightMapping(
                    f"{target}.mlp.w13.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                )
                mappings[f"{source}.mlp.w2_md.weight"] = WeightMapping(
                    f"{target}.mlp.w2.weight",
                    sharding=("tensor", None),
                    transpose=True,
                )
                mappings[f"{source}.mlp.global_scale"] = WeightMapping(
                    f"{target}.mlp.global_scale", sharding=()
                )
            else:
                mappings[f"{source}.mlp.gate.weight"] = WeightMapping(
                    f"{target}.mlp.gate.kernel",
                    sharding=(None, None),
                    transpose=True,
                )
                mappings[f"{source}.mlp.gate.bias"] = WeightMapping(
                    f"{target}.mlp.correction_bias", sharding=(None,)
                )
                mappings[f"{source}.mlp.gate.global_scale"] = WeightMapping(
                    f"{target}.mlp.global_scale", sharding=()
                )
                mappings[f"{source}.mlp.shared_experts.shared_w13_weight"] = WeightMapping(
                    f"{target}.mlp.shared_experts.w13",
                    sharding=shared_w13_sharding,
                    transpose_axes=(0, 2, 1),
                    reshape=(
                        self.text_config.n_shared_experts,
                        self.text_config.hidden_size,
                        2 * self.text_config.intermediate_size,
                    ),
                )
                mappings[f"{source}.mlp.shared_experts.shared_w2_weight"] = WeightMapping(
                    f"{target}.mlp.shared_experts.w2",
                    sharding=shared_w2_sharding,
                    transpose_axes=(0, 2, 1),
                    reshape=(
                        self.text_config.n_shared_experts,
                        self.text_config.intermediate_size,
                        self.text_config.hidden_size,
                    ),
                )
        return mappings

    @staticmethod
    def _reader(
        file_manager: SequentialSafetensorManager,
        weight_info: dict[str, list[dict]],
        name: str,
    ):
        if name not in weight_info:
            raise KeyError(f"INKLING_EXPERT_TENSOR_MISSING tensor={name}")
        info = weight_info[name][0]
        return file_manager.get_handle(info["file"]).get_slice(name)

    def _load_experts(self, loader: WeightLoader) -> None:
        weight_info = loader._scan_weight_info()
        with SequentialSafetensorManager() as file_manager:
            for layer_id, layer in enumerate(self.model.layers):
                if layer.is_dense:
                    continue
                self._load_layer_experts(file_manager, weight_info, layer_id, layer.mlp.experts)
                logger.info("INKLING_EXPERT_LAYER_LOADED layer=%d", layer_id)

    def _load_layer_experts(
        self,
        file_manager: SequentialSafetensorManager,
        weight_info: dict[str, list[dict]],
        layer_id: int,
        experts: EPMoE,
    ) -> None:
        prefix = f"model.llm.layers.{layer_id}.mlp.experts"
        w13_name = f"{prefix}.w13_weight"
        w2_name = f"{prefix}.w2_weight"
        is_nvfp4 = f"{w13_name}.scale" in weight_info

        w13_reader = self._reader(file_manager, weight_info, w13_name)
        w2_reader = self._reader(file_manager, weight_info, w2_name)
        if is_nvfp4:
            w13_scale_reader = self._reader(file_manager, weight_info, f"{w13_name}.scale")
            w13_scale2_reader = self._reader(file_manager, weight_info, f"{w13_name}.scale2")
            w2_scale_reader = self._reader(file_manager, weight_info, f"{w2_name}.scale")
            w2_scale2_reader = self._reader(file_manager, weight_info, f"{w2_name}.scale2")

        def read_w13(index: tuple[slice, ...], parity: int) -> np.ndarray:
            expert_slice, hidden_slice, intermediate_slice = index
            expert_start, expert_stop = _slice_bounds(
                expert_slice, self.text_config.n_routed_experts
            )
            hidden_start, hidden_stop = _slice_bounds(hidden_slice, self.text_config.hidden_size)
            intermediate_start, intermediate_stop = _slice_bounds(
                intermediate_slice, self.text_config.intermediate_size
            )
            if hidden_start != 0 or hidden_stop != self.text_config.hidden_size:
                raise ValueError(
                    "INKLING_EXPERT_HIDDEN_SHARD_UNSUPPORTED "
                    f"start={hidden_start} stop={hidden_stop}"
                )
            rows = slice(2 * intermediate_start, 2 * intermediate_stop)
            expert_rows = slice(expert_start, expert_stop)
            raw = np.asarray(w13_reader[expert_rows, rows, :])
            if is_nvfp4:
                block_scale = np.asarray(w13_scale_reader[expert_rows, rows, :])
                global_scale = np.asarray(w13_scale2_reader[expert_rows])
                raw = decode_nvfp4_numpy(raw, block_scale, global_scale)
            else:
                raw = raw.astype(ml_dtypes.bfloat16, copy=False)
            return np.transpose(raw[:, parity::2, :], (0, 2, 1))

        def read_w2(index: tuple[slice, ...]) -> np.ndarray:
            expert_slice, intermediate_slice, hidden_slice = index
            expert_start, expert_stop = _slice_bounds(
                expert_slice, self.text_config.n_routed_experts
            )
            intermediate_start, intermediate_stop = _slice_bounds(
                intermediate_slice, self.text_config.intermediate_size
            )
            hidden_start, hidden_stop = _slice_bounds(hidden_slice, self.text_config.hidden_size)
            if hidden_start != 0 or hidden_stop != self.text_config.hidden_size:
                raise ValueError(
                    "INKLING_EXPERT_OUTPUT_SHARD_UNSUPPORTED "
                    f"start={hidden_start} stop={hidden_stop}"
                )
            expert_rows = slice(expert_start, expert_stop)
            if is_nvfp4:
                if intermediate_start % 16 or intermediate_stop % 16:
                    raise ValueError(
                        "INKLING_NVFP4_INTERMEDIATE_SHARD_MISALIGNED "
                        f"start={intermediate_start} stop={intermediate_stop}"
                    )
                raw = np.asarray(
                    w2_reader[
                        expert_rows,
                        :,
                        slice(intermediate_start // 2, intermediate_stop // 2),
                    ]
                )
                block_scale = np.asarray(
                    w2_scale_reader[
                        expert_rows,
                        :,
                        slice(intermediate_start // 16, intermediate_stop // 16),
                    ]
                )
                global_scale = np.asarray(w2_scale2_reader[expert_rows])
                raw = decode_nvfp4_numpy(raw, block_scale, global_scale)
            else:
                raw = np.asarray(
                    w2_reader[
                        expert_rows,
                        :,
                        slice(intermediate_start, intermediate_stop),
                    ]
                ).astype(ml_dtypes.bfloat16, copy=False)
            return np.transpose(raw, (0, 2, 1))

        w13_sharding = NamedSharding(experts.moe_mesh, P("expert", None, "tensor"))
        w2_sharding = NamedSharding(experts.moe_mesh, P("expert", "tensor", None))
        experts.wi_0.value = jax.make_array_from_callback(
            experts.wi_0.shape,
            w13_sharding,
            lambda index: read_w13(index, 0),
        ).astype(self.dtype)
        experts.wi_1.value = jax.make_array_from_callback(
            experts.wi_1.shape,
            w13_sharding,
            lambda index: read_w13(index, 1),
        ).astype(self.dtype)
        experts.wo.value = jax.make_array_from_callback(
            experts.wo.shape,
            w2_sharding,
            read_w2,
        ).astype(self.dtype)

    def get_embed_and_head(self):
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value


class InklingForConditionalGeneration(InklingForCausalLM):
    pass


EntryClass = [InklingForCausalLM, InklingForConditionalGeneration]
