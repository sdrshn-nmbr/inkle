from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.configs.dspark import DSparkDraftConfig
from sgl_jax.srt.models.dspark import (
    DSparkDraftInputs,
    DSparkDraftModel,
    build_dspark_draft_model,
    create_dspark_weight_mappings,
)
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


MESH = create_device_mesh(
    ici_parallelism=[1, 1],
    dcn_parallelism=[1, 1],
    devices=[jax.devices()[0]],
)
jax.sharding.set_mesh(MESH)


def make_config(*, num_hidden_layers: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DSparkDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        block_size=3,
        markov_rank=4,
        layer_types=["full_attention"] * num_hidden_layers,
        dflash_config={"mask_token_id": 31, "target_layer_ids": [1, 3]},
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 128,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "rope_theta": 80_000.0,
        },
        attention_bias=False,
    )


def expected_checkpoint_keys(num_hidden_layers: int) -> set[str]:
    global_keys = {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }
    layer_suffixes = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    return global_keys | {
        f"layers.{layer_id}.{suffix}"
        for layer_id in range(num_hidden_layers)
        for suffix in layer_suffixes
    }


def test_real_checkpoint_config_contract() -> None:
    config = make_config(num_hidden_layers=6)
    config.hidden_size = 4096
    config.intermediate_size = 12288
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.head_dim = 128
    config.vocab_size = 201024
    config.block_size = 7
    config.markov_rank = 256
    config.dflash_config = {
        "mask_token_id": 200064,
        "target_layer_ids": [1, 6, 12, 17, 23, 28, 34, 39],
    }
    config.max_position_embeddings = 1048576
    config.rope_parameters = {
        "rope_type": "yarn",
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "rope_theta": 8000000,
    }

    parsed = DSparkDraftConfig.from_hf_config(config)

    assert parsed.context_width == 32768
    assert parsed.rope_theta == 8000000
    assert parsed.rope_scaling is not None
    assert parsed.rope_scaling["factor"] == 128.0
    assert parsed.mask_token_id == 200064


def test_checkpoint_mapping_covers_all_73_tensors() -> None:
    mappings = create_dspark_weight_mappings(6)

    assert len(mappings) == 73
    assert set(mappings) == expected_checkpoint_keys(6)
    assert mappings["layers.0.self_attn.k_proj.weight"].kv_head_padding
    assert mappings["layers.0.self_attn.v_proj.weight"].kv_head_padding
    assert mappings["markov_head.markov_w1.weight"].transpose is False
    assert mappings["confidence_head.proj.weight"].transpose


def test_factory_and_registry_resolve_dspark() -> None:
    model = build_dspark_draft_model(make_config(), MESH, dtype=jnp.float32)
    model_class, architecture = ModelRegistry.resolve_model_cls(["DSparkDraftModel"])

    assert isinstance(model, DSparkDraftModel)
    assert model_class is DSparkDraftModel
    assert architecture == "DSparkDraftModel"


def test_small_draft_forward_and_heads() -> None:
    model = build_dspark_draft_model(make_config(), MESH, dtype=jnp.float32)
    batch_size = 2
    context_length = 3
    draft_length = 2
    inputs = DSparkDraftInputs(
        noise_embeddings=jax.random.normal(
            jax.random.key(1),
            (batch_size, draft_length, model.config.hidden_size),
        ),
        target_hidden_states=jax.random.normal(
            jax.random.key(2),
            (batch_size, context_length, model.config.context_width),
        ),
        position_ids=jnp.broadcast_to(
            jnp.arange(context_length + draft_length),
            (batch_size, context_length + draft_length),
        ),
        attention_mask=jnp.ones(
            (batch_size, draft_length, context_length + draft_length),
            dtype=jnp.bool_,
        ),
    )

    hidden_states = jax.jit(lambda draft_inputs: model(draft_inputs))(inputs)
    lm_head = jax.random.normal(
        jax.random.key(3),
        (model.config.vocab_size, model.config.hidden_size),
    )
    base_logits = model.compute_base_logits(
        hidden_states,
        lm_head,
        target_vocab_size=model.config.vocab_size,
        logits_mup_width_multiplier=24.0,
    )
    proposal = model.greedy_propose(
        base_logits,
        hidden_states,
        jnp.array([1, 2], dtype=jnp.int32),
    )

    assert hidden_states.shape == (batch_size, draft_length, model.config.hidden_size)
    assert base_logits.shape == (batch_size, draft_length, model.config.vocab_size)
    assert proposal.token_ids.shape == (batch_size, draft_length)
    assert proposal.corrected_logits.shape == base_logits.shape
    assert proposal.confidence_logits.shape == (batch_size, draft_length)
    assert np.isfinite(np.asarray(hidden_states)).all()
    assert np.isfinite(np.asarray(proposal.corrected_logits)).all()


def test_cached_context_matches_direct_forward() -> None:
    model = build_dspark_draft_model(make_config(), MESH, dtype=jnp.float32)
    batch_size = 2
    context_length = 3
    draft_length = 2
    noise = jax.random.normal(
        jax.random.key(11),
        (batch_size, draft_length, model.config.hidden_size),
    )
    target = jax.random.normal(
        jax.random.key(12),
        (batch_size, context_length, model.config.context_width),
    )
    context_positions = jnp.broadcast_to(
        jnp.arange(context_length), (batch_size, context_length)
    )
    draft_positions = jnp.broadcast_to(
        jnp.arange(context_length, context_length + draft_length),
        (batch_size, draft_length),
    )
    mask = jnp.ones(
        (batch_size, draft_length, context_length + draft_length),
        dtype=jnp.bool_,
    )

    direct = model(
        DSparkDraftInputs(
            noise_embeddings=noise,
            target_hidden_states=target,
            position_ids=jnp.concatenate((context_positions, draft_positions), axis=1),
            attention_mask=mask,
        )
    )
    context = model.encode_context(target, context_positions)
    cached = model.forward_cached(noise, context, draft_positions, mask)

    np.testing.assert_allclose(np.asarray(cached), np.asarray(direct), rtol=1e-5, atol=1e-5)


def test_config_rejects_unsupported_draft_layer() -> None:
    config = make_config()
    config.layer_types = ["full_attention", "sliding_attention"]

    with pytest.raises(ValueError, match="full-attention layers only"):
        DSparkDraftConfig.from_hf_config(config)
