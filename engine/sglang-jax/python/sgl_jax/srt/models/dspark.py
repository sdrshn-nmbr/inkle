from __future__ import annotations

import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.dspark import DSparkDraftConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.qwen3 import Qwen3MLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def create_dspark_weight_mappings(num_hidden_layers: int) -> dict[str, WeightMapping]:
    if num_hidden_layers <= 0:
        raise ValueError(f"DSpark num_hidden_layers must be positive, got {num_hidden_layers}")

    mappings: dict[str, WeightMapping] = {
        "fc.weight": WeightMapping(
            target_path="fc.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        "hidden_norm.weight": WeightMapping(
            target_path="hidden_norm.scale",
            sharding=(None,),
        ),
        "norm.weight": WeightMapping(
            target_path="norm.scale",
            sharding=(None,),
        ),
        "markov_head.markov_w1.weight": WeightMapping(
            target_path="markov_head.markov_w1.embedding",
            sharding=("tensor", None),
        ),
        "markov_head.markov_w2.weight": WeightMapping(
            target_path="markov_head.markov_w2.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        "confidence_head.proj.weight": WeightMapping(
            target_path="confidence_head.proj.weight",
            sharding=(None, None),
            transpose=True,
        ),
        "confidence_head.proj.bias": WeightMapping(
            target_path="confidence_head.proj.bias",
            sharding=(None,),
        ),
    }
    layer_mappings = {
        "input_layernorm.weight": ("input_layernorm.scale", (None,), False, False),
        "post_attention_layernorm.weight": (
            "post_attention_layernorm.scale",
            (None,),
            False,
            False,
        ),
        "self_attn.q_proj.weight": (
            "self_attn.q_proj.weight",
            (None, "tensor"),
            True,
            False,
        ),
        "self_attn.k_proj.weight": (
            "self_attn.k_proj.weight",
            (None, "tensor"),
            True,
            True,
        ),
        "self_attn.v_proj.weight": (
            "self_attn.v_proj.weight",
            (None, "tensor"),
            True,
            True,
        ),
        "self_attn.o_proj.weight": (
            "self_attn.o_proj.weight",
            ("tensor", None),
            True,
            False,
        ),
        "self_attn.q_norm.weight": ("self_attn.q_norm.scale", (None,), False, False),
        "self_attn.k_norm.weight": ("self_attn.k_norm.scale", (None,), False, False),
        "mlp.gate_proj.weight": ("mlp.gate_proj.weight", (None, "tensor"), True, False),
        "mlp.up_proj.weight": ("mlp.up_proj.weight", (None, "tensor"), True, False),
        "mlp.down_proj.weight": ("mlp.down_proj.weight", ("tensor", None), True, False),
    }
    for layer_id in range(num_hidden_layers):
        for source_suffix, (
            target_suffix,
            sharding,
            transpose,
            kv_head_padding,
        ) in layer_mappings.items():
            mappings[f"layers.{layer_id}.{source_suffix}"] = WeightMapping(
                target_path=f"layers.{layer_id}.{target_suffix}",
                sharding=sharding,
                transpose=transpose,
                kv_head_padding=kv_head_padding,
            )
    return mappings


class DSparkDraftInputs(NamedTuple):
    noise_embeddings: jax.Array
    target_hidden_states: jax.Array
    position_ids: jax.Array
    attention_mask: jax.Array | None = None


class DSparkProposal(NamedTuple):
    token_ids: jax.Array
    corrected_logits: jax.Array
    confidence_logits: jax.Array


class DSparkDraftIntermediates(NamedTuple):
    projected_target_hidden_states: jax.Array
    layer_hidden_states: tuple[jax.Array, ...]
    final_hidden_states: jax.Array


class DSparkLayerContext(NamedTuple):
    key: jax.Array
    value: jax.Array


class DSparkContextCache(NamedTuple):
    layers: tuple[DSparkLayerContext, ...]


class DSparkAttention(nnx.Module):
    def __init__(
        self,
        config: DSparkDraftConfig,
        mesh: jax.sharding.Mesh,
        *,
        layer_id: int,
        dtype: jnp.dtype,
    ) -> None:
        self.layer_id = layer_id
        self.mesh = mesh
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = LinearBase(
            config.hidden_size,
            self.num_heads * self.head_dim,
            mesh,
            use_bias=config.attention_bias,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            mesh,
            use_bias=config.attention_bias,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            mesh,
            use_bias=config.attention_bias,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            self.num_heads * self.head_dim,
            config.hidden_size,
            mesh,
            use_bias=config.attention_bias,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            scope_name="o_proj",
        )
        self.q_norm = RMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="q_norm",
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="k_norm",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rope_theta),
            rope_scaling=dict(config.rope_scaling) if config.rope_scaling is not None else None,
            is_neox_style=True,
            dtype=dtype,
        )

    def _apply_rope(
        self,
        query: jax.Array,
        key: jax.Array,
        position_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        batch_size, query_length = query.shape[:2]
        key_length = key.shape[1]
        query_positions = position_ids[:, key_length - query_length :]

        query_flat = query.reshape(batch_size * query_length, self.num_heads, self.head_dim)
        query_flat, _ = self.rotary_emb(query_positions, query_flat, query_flat)
        key_flat = key.reshape(batch_size * key_length, self.num_kv_heads, self.head_dim)
        _, key_flat = self.rotary_emb(position_ids, key_flat, key_flat)
        return (
            query_flat.reshape(batch_size, query_length, self.num_heads, self.head_dim),
            key_flat.reshape(batch_size, key_length, self.num_kv_heads, self.head_dim),
        )

    def encode_context(
        self,
        target_hidden_states: jax.Array,
        position_ids: jax.Array,
    ) -> DSparkLayerContext:
        batch_size, context_length, _ = target_hidden_states.shape
        if position_ids.shape != (batch_size, context_length):
            raise ValueError(
                "DSpark context positions must match the cached context, "
                f"expected {(batch_size, context_length)}, got {position_ids.shape}"
            )
        key, _ = self.k_proj(target_hidden_states)
        value, _ = self.v_proj(target_hidden_states)
        key = key.reshape(batch_size, context_length, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, context_length, self.num_kv_heads, self.head_dim)
        key = self.k_norm(key)
        key_flat = key.reshape(batch_size * context_length, self.num_kv_heads, self.head_dim)
        _, key_flat = self.rotary_emb(position_ids.reshape(-1), key_flat, key_flat)
        return DSparkLayerContext(
            key=key_flat.reshape(batch_size, context_length, self.num_kv_heads, self.head_dim),
            value=value,
        )

    def apply_cached_context(
        self,
        hidden_states: jax.Array,
        context: DSparkLayerContext,
        draft_position_ids: jax.Array,
        attention_mask: jax.Array | None,
    ) -> jax.Array:
        batch_size, query_length, _ = hidden_states.shape
        context_length = context.key.shape[1]
        key_length = context_length + query_length
        if draft_position_ids.shape != (batch_size, query_length):
            raise ValueError(
                "DSpark draft positions must match the draft block, "
                f"expected {(batch_size, query_length)}, got {draft_position_ids.shape}"
            )
        if attention_mask is not None and attention_mask.shape != (
            batch_size,
            query_length,
            key_length,
        ):
            raise ValueError(
                "DSpark cached attention mask must have shape "
                "[batch, draft, context + draft], "
                f"expected {(batch_size, query_length, key_length)}, "
                f"got {attention_mask.shape}"
            )

        query, _ = self.q_proj(hidden_states)
        draft_key, _ = self.k_proj(hidden_states)
        draft_value, _ = self.v_proj(hidden_states)
        query = query.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        draft_key = draft_key.reshape(batch_size, query_length, self.num_kv_heads, self.head_dim)
        draft_value = draft_value.reshape(
            batch_size, query_length, self.num_kv_heads, self.head_dim
        )
        query = self.q_norm(query)
        draft_key = self.k_norm(draft_key)
        query_flat = query.reshape(batch_size * query_length, self.num_heads, self.head_dim)
        query_flat, _ = self.rotary_emb(draft_position_ids.reshape(-1), query_flat, query_flat)
        key_flat = draft_key.reshape(batch_size * query_length, self.num_kv_heads, self.head_dim)
        _, key_flat = self.rotary_emb(draft_position_ids.reshape(-1), key_flat, key_flat)
        query = query_flat.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        draft_key = key_flat.reshape(batch_size, query_length, self.num_kv_heads, self.head_dim)
        attention_sharding = NamedSharding(
            self.mesh,
            P("data", None, "tensor", None),
        )
        query = jax.sharding.reshard(query, attention_sharding)
        draft_key = jax.sharding.reshard(draft_key, attention_sharding)
        draft_value = jax.sharding.reshard(draft_value, attention_sharding)
        context_key = jax.sharding.reshard(context.key, attention_sharding)
        context_value = jax.sharding.reshard(context.value, attention_sharding)
        key = jnp.concatenate((context_key, draft_key), axis=1)
        value = jnp.concatenate((context_value, draft_value), axis=1)

        key = jnp.repeat(
            key,
            self.num_key_value_groups,
            axis=2,
            out_sharding=attention_sharding,
        )
        value = jnp.repeat(
            value,
            self.num_key_value_groups,
            axis=2,
            out_sharding=attention_sharding,
        )
        scores = (
            jnp.einsum(
                "bqhd,bkhd->bhqk",
                query,
                key,
                preferred_element_type=jnp.float32,
            )
            * self.scale
        )
        if attention_mask is not None:
            scores = jnp.where(attention_mask[:, None, :, :], scores, -jnp.inf)
        probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value)
        attended = attended.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        output, _ = self.o_proj(
            attended,
            out_sharding=NamedSharding(self.mesh, P("data", None, None)),
        )
        return output

    def __call__(
        self,
        hidden_states: jax.Array,
        target_hidden_states: jax.Array,
        position_ids: jax.Array,
        attention_mask: jax.Array | None,
    ) -> jax.Array:
        batch_size, query_length, _ = hidden_states.shape
        context_length = target_hidden_states.shape[1]
        key_length = context_length + query_length
        if position_ids.shape != (batch_size, key_length):
            raise ValueError(
                "DSpark position_ids must cover context and draft tokens, "
                f"expected {(batch_size, key_length)}, got {position_ids.shape}"
            )
        if attention_mask is not None and attention_mask.shape != (
            batch_size,
            query_length,
            key_length,
        ):
            raise ValueError(
                "DSpark attention_mask must have shape [batch, draft, context + draft], "
                f"expected {(batch_size, query_length, key_length)}, got {attention_mask.shape}"
            )

        query, _ = self.q_proj(hidden_states)
        context_key, _ = self.k_proj(target_hidden_states)
        draft_key, _ = self.k_proj(hidden_states)
        context_value, _ = self.v_proj(target_hidden_states)
        draft_value, _ = self.v_proj(hidden_states)

        query = query.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        key = jnp.concatenate((context_key, draft_key), axis=1).reshape(
            batch_size, key_length, self.num_kv_heads, self.head_dim
        )
        value = jnp.concatenate((context_value, draft_value), axis=1).reshape(
            batch_size, key_length, self.num_kv_heads, self.head_dim
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self._apply_rope(query, key, position_ids)

        repeated_head_sharding = NamedSharding(self.mesh, P("data", None, "tensor", None))
        key = jnp.repeat(
            key,
            self.num_key_value_groups,
            axis=2,
            out_sharding=repeated_head_sharding,
        )
        value = jnp.repeat(
            value,
            self.num_key_value_groups,
            axis=2,
            out_sharding=repeated_head_sharding,
        )
        scores = (
            jnp.einsum(
                "bqhd,bkhd->bhqk",
                query,
                key,
                preferred_element_type=jnp.float32,
            )
            * self.scale
        )
        if attention_mask is not None:
            scores = jnp.where(attention_mask[:, None, :, :], scores, -jnp.inf)
        probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value)
        attended = attended.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        output, _ = self.o_proj(
            attended,
            out_sharding=NamedSharding(self.mesh, P("data", None, None)),
        )
        return output


class DSparkDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: DSparkDraftConfig,
        mesh: jax.sharding.Mesh,
        *,
        layer_id: int,
        dtype: jnp.dtype,
    ) -> None:
        self.self_attn = DSparkAttention(
            config,
            mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.mlp = Qwen3MLP(
            config.hidden_size,
            config.intermediate_size,
            mesh,
            layer_id=layer_id,
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        target_hidden_states: jax.Array,
        position_ids: jax.Array,
        attention_mask: jax.Array | None,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden_states,
            position_ids,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            out_sharding=NamedSharding(self.self_attn.mesh, P("data", None, None)),
        )
        return residual + hidden_states

    def apply_cached_context(
        self,
        hidden_states: jax.Array,
        context: DSparkLayerContext,
        draft_position_ids: jax.Array,
        attention_mask: jax.Array | None,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.apply_cached_context(
            hidden_states,
            context,
            draft_position_ids,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            out_sharding=NamedSharding(self.self_attn.mesh, P("data", None, None)),
        )
        return residual + hidden_states


class DSparkMarkovHead(nnx.Module):
    def __init__(
        self,
        config: DSparkDraftConfig,
        mesh: jax.sharding.Mesh,
        *,
        dtype: jnp.dtype,
    ) -> None:
        self.markov_w1 = Embed(
            config.vocab_size,
            config.markov_rank,
            mesh=mesh,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.markov_w2 = LinearBase(
            config.markov_rank,
            config.vocab_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="markov_w2",
        )

    def embedding(self, token_ids: jax.Array) -> jax.Array:
        return self.markov_w1(token_ids)

    def bias(self, previous_embeddings: jax.Array) -> jax.Array:
        bias, _ = self.markov_w2(previous_embeddings)
        return bias


class DSparkConfidenceHead(nnx.Module):
    def __init__(
        self,
        config: DSparkDraftConfig,
        mesh: jax.sharding.Mesh,
        *,
        dtype: jnp.dtype,
    ) -> None:
        input_size = config.hidden_size
        if config.confidence_head_with_markov:
            input_size += config.markov_rank
        self.with_markov = config.confidence_head_with_markov
        self.proj = LinearBase(
            input_size,
            1,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="confidence",
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        markov_embeddings: jax.Array,
    ) -> jax.Array:
        features = (
            jnp.concatenate((hidden_states, markov_embeddings), axis=-1)
            if self.with_markov
            else hidden_states
        )
        logits, _ = self.proj(features)
        return logits[..., 0]


class DSparkDraftModel(nnx.Module):
    def __init__(
        self,
        config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.config = DSparkDraftConfig.from_hf_config(config)
        self.mesh = mesh
        self.dtype = dtype
        self.layers = nnx.data(
            [
                DSparkDecoderLayer(
                    self.config,
                    mesh,
                    layer_id=layer_id,
                    dtype=dtype,
                )
                for layer_id in range(self.config.num_hidden_layers)
            ]
        )
        self.fc = LinearBase(
            self.config.context_width,
            self.config.hidden_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            scope_name="context_projection",
        )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            epsilon=self.config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="hidden_norm",
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            epsilon=self.config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )
        self.markov_head = DSparkMarkovHead(self.config, mesh, dtype=dtype)
        self.confidence_head = DSparkConfidenceHead(self.config, mesh, dtype=dtype)

    def project_target_hidden(self, target_hidden_states: jax.Array) -> jax.Array:
        if target_hidden_states.ndim != 3:
            raise ValueError(
                "DSpark target_hidden_states must have shape [batch, context, features], "
                f"got {target_hidden_states.shape}"
            )
        if target_hidden_states.shape[-1] != self.config.context_width:
            raise ValueError(
                "DSpark target hidden width does not match the configured layer taps, "
                f"expected {self.config.context_width}, got {target_hidden_states.shape[-1]}"
            )
        projected, _ = self.fc(
            target_hidden_states,
            out_sharding=NamedSharding(self.mesh, P("data", None, None)),
        )
        return self.hidden_norm(projected)

    def forward_with_intermediates(self, inputs: DSparkDraftInputs) -> DSparkDraftIntermediates:
        noise_embeddings = inputs.noise_embeddings
        if noise_embeddings.ndim != 3 or noise_embeddings.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "DSpark noise_embeddings must have shape [batch, draft, hidden_size], "
                f"got {noise_embeddings.shape}"
            )
        target_hidden_states = self.project_target_hidden(inputs.target_hidden_states)
        if target_hidden_states.shape[0] != noise_embeddings.shape[0]:
            raise ValueError(
                "DSpark target and noise batch sizes differ, "
                f"got {target_hidden_states.shape[0]} and {noise_embeddings.shape[0]}"
            )

        hidden_states = noise_embeddings
        layer_hidden_states = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                target_hidden_states,
                inputs.position_ids,
                inputs.attention_mask,
            )
            layer_hidden_states.append(hidden_states)
        final_hidden_states = self.norm(hidden_states)
        return DSparkDraftIntermediates(
            projected_target_hidden_states=target_hidden_states,
            layer_hidden_states=tuple(layer_hidden_states),
            final_hidden_states=final_hidden_states,
        )

    def __call__(self, inputs: DSparkDraftInputs) -> jax.Array:
        return self.forward_with_intermediates(inputs).final_hidden_states

    def encode_context(
        self,
        target_hidden_states: jax.Array,
        position_ids: jax.Array,
    ) -> DSparkContextCache:
        projected = self.project_target_hidden(target_hidden_states)
        return DSparkContextCache(
            layers=tuple(
                layer.self_attn.encode_context(projected, position_ids) for layer in self.layers
            )
        )

    def forward_cached(
        self,
        noise_embeddings: jax.Array,
        context: DSparkContextCache,
        draft_position_ids: jax.Array,
        attention_mask: jax.Array | None,
    ) -> jax.Array:
        if len(context.layers) != len(self.layers):
            raise ValueError(
                "DSpark context cache layer count does not match the draft model, "
                f"expected {len(self.layers)}, got {len(context.layers)}"
            )
        hidden_states = noise_embeddings
        for layer, layer_context in zip(self.layers, context.layers):
            hidden_states = layer.apply_cached_context(
                hidden_states,
                layer_context,
                draft_position_ids,
                attention_mask,
            )
        return self.norm(hidden_states)

    def compute_base_logits(
        self,
        hidden_states: jax.Array,
        target_lm_head: jax.Array,
        *,
        target_vocab_size: int,
        logits_mup_width_multiplier: float,
    ) -> jax.Array:
        if target_lm_head.ndim != 2 or target_lm_head.shape[1] != self.config.hidden_size:
            raise ValueError(
                "DSpark target_lm_head must have shape [vocab, hidden_size], "
                f"got {target_lm_head.shape}"
            )
        if not 0 < target_vocab_size <= target_lm_head.shape[0]:
            raise ValueError(
                f"DSpark target_vocab_size must be in [1, {target_lm_head.shape[0]}], "
                f"got {target_vocab_size}"
            )
        if logits_mup_width_multiplier <= 0:
            raise ValueError(
                "DSpark logits_mup_width_multiplier must be positive, "
                f"got {logits_mup_width_multiplier}"
            )
        scaled_hidden = hidden_states / logits_mup_width_multiplier
        logits = jnp.einsum(
            "...h,vh->...v",
            scaled_hidden,
            target_lm_head,
            preferred_element_type=jnp.float32,
            out_sharding=NamedSharding(self.mesh, P("data", None, "tensor")),
        )
        return logits[..., :target_vocab_size]

    def greedy_propose(
        self,
        base_logits: jax.Array,
        hidden_states: jax.Array,
        anchor_token_ids: jax.Array,
    ) -> DSparkProposal:
        if base_logits.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "DSpark base logits and hidden states must share batch and draft dimensions, "
                f"got {base_logits.shape} and {hidden_states.shape}"
            )
        if base_logits.shape[-1] != self.config.vocab_size:
            raise ValueError(
                f"DSpark base logits require vocab size {self.config.vocab_size}, "
                f"got {base_logits.shape[-1]}"
            )
        if anchor_token_ids.shape != (base_logits.shape[0],):
            raise ValueError(
                f"DSpark anchor_token_ids must have shape {(base_logits.shape[0],)}, "
                f"got {anchor_token_ids.shape}"
            )

        def step(previous_tokens, step_inputs):
            step_logits, step_hidden = step_inputs
            previous_embeddings = self.markov_head.embedding(previous_tokens)
            corrected_logits = step_logits + self.markov_head.bias(previous_embeddings)
            token_ids = jnp.argmax(corrected_logits, axis=-1).astype(jnp.int32)
            confidence_logits = self.confidence_head(step_hidden, previous_embeddings)
            return token_ids, (token_ids, corrected_logits, confidence_logits)

        anchor_token_ids = jax.device_put(
            anchor_token_ids.astype(jnp.int32),
            NamedSharding(self.mesh, P("data")),
        )
        _, (token_ids, corrected_logits, confidence_logits) = jax.lax.scan(
            step,
            anchor_token_ids,
            (jnp.swapaxes(base_logits, 0, 1), jnp.swapaxes(hidden_states, 0, 1)),
        )
        return DSparkProposal(
            token_ids=jnp.swapaxes(token_ids, 0, 1),
            corrected_logits=jnp.swapaxes(corrected_logits, 0, 1),
            confidence_logits=jnp.swapaxes(confidence_logits, 0, 1),
        )

    def load_weights(self, model_config: ModelConfig) -> None:
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("DSpark draft weights loaded successfully")

    def _create_weight_mappings(self) -> dict[str, WeightMapping]:
        return create_dspark_weight_mappings(self.config.num_hidden_layers)


def build_dspark_draft_model(
    config,
    mesh: jax.sharding.Mesh,
    *,
    dtype: jnp.dtype = jnp.bfloat16,
) -> DSparkDraftModel:
    return DSparkDraftModel(config, mesh, dtype=dtype)


EntryClass = DSparkDraftModel
