import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

from sgl_jax.srt.configs.inkling import InklingConfig, InklingTextConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.attention.native_backend import relative_position_bias
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.inkling import (
    InklingDenseMLP,
    InklingForCausalLM,
    InklingMoE,
    InklingShortConvolution,
)
from sgl_jax.srt.models.inkling_layout import decode_nvfp4_numpy
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.srt.utils.weight_utils import SequentialSafetensorManager, WeightLoader

MESH = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(MESH)


def make_config(
    num_layers: int = 3,
    hidden_size: int = 32,
    head_dim: int = 8,
    intermediate_size: int = 8,
):
    text = SimpleNamespace(
        vocab_size=64,
        unpadded_vocab_size=60,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        d_rel=4,
        rel_extent=16,
        local_layer_ids=list(range(num_layers)),
        swa_num_key_value_heads=2,
        sliding_window_size=8,
        log_scaling_n_floor=128000,
        log_scaling_alpha=0.1,
        rms_norm_eps=1e-6,
        dense_mlp_idx=2,
        sconv_kernel_size=4,
        dense_intermediate_size=24,
        intermediate_size=intermediate_size,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=2,
        route_scale=8.0,
        logits_mup_width_multiplier=24.0,
        final_logit_softcapping=None,
    )
    return SimpleNamespace(text_config=text, ep_size=1)


class TestInklingLayout(unittest.TestCase):
    def test_nvfp4_decode_uses_low_nibble_first_and_per_expert_scale(self):
        packed = np.zeros((2, 1, 8), dtype=np.uint8)
        packed[0, 0, 0] = 0x21
        packed[1, 0, 0] = 0x43
        block_scale = np.ones((2, 1, 1), dtype=np.float32)
        global_scale = np.asarray([2.0, 3.0], dtype=np.float32)
        decoded = decode_nvfp4_numpy(packed, block_scale, global_scale).astype(np.float32)
        np.testing.assert_array_equal(
            decoded[..., :2],
            np.asarray([[[1.0, 2.0]], [[4.5, 6.0]]], dtype=np.float32),
        )

    def test_relative_bias_respects_packed_sequence_boundaries(self):
        relative = jnp.arange(5 * 2 * 2, dtype=jnp.float32).reshape(5, 2, 2)
        projection = jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4) / 10
        actual = np.asarray(
            relative_position_bias(
                relative,
                projection,
                jnp.asarray([3, 2], dtype=jnp.int32),
                jnp.asarray([0, 0], dtype=jnp.int32),
                jnp.asarray([3, 2], dtype=jnp.int32),
                ForwardMode.EXTEND,
                5,
                MESH,
            )
        )
        projected = np.einsum("qhd,de->qhe", np.asarray(relative), np.asarray(projection))
        expected = np.zeros((5, 2, 5), dtype=np.float32)
        batches = np.asarray([0, 0, 0, 1, 1])
        positions = np.asarray([0, 1, 2, 0, 1])
        for query in range(5):
            for key in range(5):
                distance = positions[query] - positions[key]
                if batches[query] == batches[key] and 0 <= distance < 4:
                    expected[query, :, key] = projected[query, :, distance]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_relative_bias_traces_with_sharded_sequence_lengths(self):
        relative_sharding = jax.sharding.NamedSharding(
            MESH, jax.sharding.PartitionSpec("data", None, None)
        )
        lengths_sharding = jax.sharding.NamedSharding(
            MESH, jax.sharding.PartitionSpec("data")
        )
        output_sharding = jax.sharding.NamedSharding(
            MESH, jax.sharding.PartitionSpec("data", None, None)
        )
        projection = jnp.ones((2, 4), dtype=jnp.float32)
        prefix_lengths = jnp.zeros((2,), dtype=jnp.int32)

        run_bias = jax.jit(
            lambda relative, lengths: relative_position_bias(
                relative,
                projection,
                lengths,
                prefix_lengths,
                lengths,
                ForwardMode.EXTEND,
                5,
                MESH,
            ),
            in_shardings=(relative_sharding, lengths_sharding),
            out_shardings=output_sharding,
        )
        result = run_bias(
            jnp.ones((5, 2, 2), dtype=jnp.float32),
            jnp.asarray([3, 2], dtype=jnp.int32),
        )

        self.assertEqual(result.shape, (5, 2, 5))


class TestInklingModel(unittest.TestCase):
    def test_padded_vocabulary_rows_are_masked_without_slicing(self):
        model = InklingForCausalLM(
            make_config(num_layers=1),
            MESH,
            dtype=jnp.bfloat16,
        )

        logits = model.logits_processor._get_logits(
            jnp.ones((1, 32), dtype=jnp.bfloat16),
            model.lm_head,
        )

        self.assertEqual(logits.shape, (1, 64))
        np.testing.assert_array_equal(
            np.asarray(logits[:, 60:], dtype=np.float32),
            np.full((1, 4), float(jnp.finfo(jnp.bfloat16).min), dtype=np.float32),
        )

    def test_local_config_preserves_released_text_shape(self):
        config = InklingConfig(
            architectures=["InklingForConditionalGeneration"],
            text_config={
                "hidden_size": 6144,
                "num_hidden_layers": 66,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "swa_num_key_value_heads": 16,
                "local_layer_ids": [0, 1, 2],
            },
        )
        self.assertIsInstance(config.text_config, InklingTextConfig)
        self.assertEqual(config.text_config.hidden_size, 6144)
        self.assertEqual(config.text_config.swa_num_key_value_heads, 16)
        self.assertEqual(config.architectures, ["InklingForConditionalGeneration"])

    def test_kv_pool_uses_larger_local_attention_head_count(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(model_type="inkling_mm_model")
        model_config.hf_text_config = SimpleNamespace(
            num_key_value_heads=8, swa_num_key_value_heads=16
        )
        self.assertEqual(model_config.get_total_num_kv_heads_with_replication(1), 16)
        self.assertEqual(model_config.get_total_num_kv_heads_with_replication(8), 16)
        self.assertEqual(model_config.get_total_num_kv_heads_with_replication(16), 16)

    def test_full_attention_kv_heads_replicate_without_changing_local_heads(self):
        model_config = object.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(model_type="inkling_mm_model")
        model_config.hf_text_config = SimpleNamespace(
            num_key_value_heads=16,
            swa_num_key_value_heads=16,
            head_dim=128,
            swa_head_dim=128,
        )
        model_config._original_hf_num_key_value_heads = 8
        model_config._original_swa_num_key_value_heads = 16
        loader = object.__new__(WeightLoader)
        loader.model_config = model_config
        loader.sharding_size = 16
        loader.head_dim = 128

        full = jnp.arange(8 * 128, dtype=jnp.float32).reshape(1, 8 * 128)
        replicated = np.asarray(
            loader._apply_kv_head_padding(full, "model.llm.layers.5.attn.wk_dv.weight")
        ).reshape(16, 128)
        original = np.asarray(full).reshape(8, 128)
        np.testing.assert_array_equal(replicated[0], original[0])
        np.testing.assert_array_equal(replicated[1], original[0])
        np.testing.assert_array_equal(replicated[2], original[1])

        local = jnp.arange(16 * 128, dtype=jnp.float32).reshape(1, 16 * 128)
        unchanged = loader._apply_kv_head_padding(
            local, "model.llm.layers.4.attn.wk_dv.weight"
        )
        np.testing.assert_array_equal(np.asarray(unchanged), np.asarray(local))

    def test_dense_w13_is_interleaved_gate_then_up(self):
        mlp = InklingDenseMLP(4, 3, MESH, jnp.float32)
        raw = np.arange(4 * 6, dtype=np.float32).reshape(4, 6) / 20
        down = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) / 10
        mlp.w13.weight = nnx.Param(jnp.asarray(raw))
        mlp.w2.weight = nnx.Param(jnp.asarray(down))
        mlp.global_scale = nnx.Param(jnp.asarray(0.75, dtype=jnp.float32))
        hidden = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 7
        actual = np.asarray(mlp(hidden))
        gate_up = np.asarray(hidden) @ raw
        expected = (
            jax.nn.silu(jnp.asarray(gate_up[:, 0::2]))
            * jnp.asarray(gate_up[:, 1::2])
        ) @ jnp.asarray(down)
        np.testing.assert_allclose(actual, np.asarray(expected) * 0.75, rtol=1e-6, atol=1e-6)

    def test_router_jointly_normalizes_routed_and_shared_experts(self):
        config = make_config().text_config
        moe = InklingMoE(config, make_config(), 2, MESH, jnp.float32)
        kernel = jnp.arange(32 * 10, dtype=jnp.float32).reshape(32, 10) / 200
        bias = jnp.linspace(-0.2, 0.2, 8, dtype=jnp.float32)
        moe.gate.kernel = nnx.Param(kernel)
        moe.correction_bias = nnx.Param(bias)
        moe.global_scale = nnx.Param(jnp.asarray(0.5, dtype=jnp.float32))
        hidden = jnp.arange(64, dtype=jnp.float32).reshape(2, 32) / 100
        ids, routed, shared = moe.routing_weights(hidden)
        logits = hidden @ kernel
        expected_ids = jax.lax.top_k(jax.nn.sigmoid(logits[:, :8]) + bias, 2)[1]
        selected = jnp.concatenate(
            (jnp.take_along_axis(logits[:, :8], expected_ids, axis=-1), logits[:, 8:]),
            axis=-1,
        )
        log_weights = jax.nn.log_sigmoid(selected)
        expected = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights, axis=-1, keepdims=True)) * 4
        np.testing.assert_array_equal(np.asarray(ids), np.asarray(expected_ids))
        np.testing.assert_allclose(np.asarray(routed), np.asarray(expected[:, :2]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(shared), np.asarray(expected[:, 2:]), rtol=1e-6)

    def test_router_preserves_bfloat16_reference_rounding(self):
        config = make_config().text_config
        moe = InklingMoE(config, make_config(), 2, MESH, jnp.bfloat16)
        kernel = (
            jnp.arange(32 * 10, dtype=jnp.float32).reshape(32, 10) / 200
        ).astype(jnp.bfloat16)
        bias = jnp.linspace(-0.2, 0.2, 8, dtype=jnp.bfloat16)
        scale = jnp.asarray(0.00622559, dtype=jnp.bfloat16)
        moe.gate.kernel = nnx.Param(kernel)
        moe.correction_bias = nnx.Param(bias)
        moe.global_scale = nnx.Param(scale)
        hidden = (jnp.arange(64, dtype=jnp.float32).reshape(2, 32) / 100).astype(
            jnp.bfloat16
        )

        ids, routed, shared = moe.routing_weights(hidden)
        logits = jnp.dot(hidden, kernel, precision=jax.lax.Precision.HIGHEST)
        choice_scores = (
            jax.nn.sigmoid(logits[:, :8].astype(jnp.float32)).astype(jnp.bfloat16)
            + bias
        )
        expected_ids = jax.lax.top_k(choice_scores.astype(jnp.float32), 2)[1]
        selected = jnp.concatenate(
            (jnp.take_along_axis(logits[:, :8], expected_ids, axis=-1), logits[:, 8:]),
            axis=-1,
        )
        log_weights = jax.nn.log_sigmoid(selected.astype(jnp.float32)).astype(jnp.bfloat16)
        expected = (
            jnp.exp(
                log_weights
                - jax.scipy.special.logsumexp(log_weights, axis=-1, keepdims=True)
            )
            * config.route_scale
            * scale
        )

        self.assertEqual(moe.gate.kernel.value.dtype, jnp.bfloat16)
        self.assertEqual(moe.correction_bias.value.dtype, jnp.bfloat16)
        self.assertEqual(moe.global_scale.value.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(ids), np.asarray(expected_ids))
        np.testing.assert_array_equal(np.asarray(routed), np.asarray(expected[:, :2]))
        np.testing.assert_array_equal(np.asarray(shared), np.asarray(expected[:, 2:]))

    def test_checkpoint_mapping_covers_dense_and_sparse_layouts(self):
        model = InklingForCausalLM(make_config(), MESH, dtype=jnp.bfloat16)
        mappings = model._create_weight_mappings()
        self.assertIn("model.llm.layers.0.mlp.w13_dn.weight", mappings)
        self.assertIn("model.llm.layers.2.mlp.gate.weight", mappings)
        self.assertIn("model.llm.layers.2.mlp.shared_experts.shared_w13_weight", mappings)
        self.assertNotIn("model.llm.layers.2.mlp.experts.w13_weight", mappings)
        self.assertEqual(
            mappings["model.llm.layers.0.mlp.w13_dn.weight"].target_path,
            "model.layers.0.mlp.w13.weight",
        )

    def test_short_convolution_weights_remain_float32(self):
        model = InklingForCausalLM(
            make_config(num_layers=1, hidden_size=512, head_dim=128),
            MESH,
            dtype=jnp.bfloat16,
        )
        layer = model.model.layers[0]
        convolutions = (
            layer.self_attn.k_sconv,
            layer.self_attn.v_sconv,
            layer.attn_sconv,
            layer.mlp_sconv,
        )
        self.assertTrue(all(conv.weight.value.dtype == jnp.float32 for conv in convolutions))

    def test_convolution_state_pool_allocates_four_layer_specific_histories(self):
        config = InklingTextConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            swa_num_key_value_heads=3,
            head_dim=8,
            local_layer_ids=[0],
        )
        params = config.linear_state_params
        pool = RecurrentStatePool(
            linear_recurrent_layer_ids=params.layers,
            size=1,
            num_heads=params.num_heads,
            head_dim=params.head_dim,
            conv_kernel_size=params.conv_kernel_size,
            mesh=MESH,
            temporal_dtype=params.dtype.temporal,
            conv_dtype=params.dtype.conv,
            conv_channel_sizes=params.conv_channel_sizes,
            has_temporal_state=params.has_temporal_state,
        )

        self.assertEqual(params.conv_channel_sizes[0], [24, 24, 32, 32])
        self.assertEqual(params.conv_channel_sizes[1], [16, 16, 32, 32])
        self.assertEqual(
            [state.shape for state in pool.conv_buffers[0]],
            [(2, 24, 3), (2, 24, 3), (2, 32, 3), (2, 32, 3)],
        )
        self.assertTrue(all(state.dtype == jnp.float32 for state in pool.conv_buffers[0]))

    def test_prompt_then_decode_matches_one_shot_convolution(self):
        convolution = InklingShortConvolution(4, 4, MESH)
        convolution.weight = nnx.Param(
            jnp.asarray(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [-0.2, 0.1, 0.5, 0.3],
                    [0.7, -0.1, 0.2, 0.4],
                    [0.3, 0.6, -0.2, 0.1],
                ],
                dtype=jnp.float32,
            )
        )
        hidden = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 10
        one_shot_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=jnp.asarray([4], dtype=jnp.int32),
            recurrent_indices=None,
        )
        one_shot, _ = convolution.apply(hidden, one_shot_batch, None)

        state = jnp.zeros((2, 4, 3), dtype=jnp.float32)
        prompt_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens=jnp.asarray([0], dtype=jnp.int32),
            extend_seq_lens=jnp.asarray([3], dtype=jnp.int32),
            recurrent_indices=jnp.asarray([1], dtype=jnp.int32),
        )
        prompt, state = convolution.apply(hidden[:3], prompt_batch, state)
        decode_batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
            recurrent_indices=jnp.asarray([1], dtype=jnp.int32),
        )
        decoded, state = convolution.apply(hidden[3:], decode_batch, state)

        np.testing.assert_allclose(
            np.concatenate((np.asarray(prompt), np.asarray(decoded))),
            np.asarray(one_shot),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(np.asarray(state[0]), np.zeros((4, 3)))
        np.testing.assert_allclose(
            np.asarray(state[1]),
            np.asarray(hidden[1:].T),
            rtol=0,
            atol=0,
        )
        repeated_prompt, _ = convolution.apply(hidden[:3], prompt_batch, state)
        np.testing.assert_allclose(
            np.asarray(repeated_prompt),
            np.asarray(prompt),
            rtol=0,
            atol=1e-6,
        )

    def test_short_convolution_can_be_traced(self):
        convolution = InklingShortConvolution(4, 4, MESH)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=jnp.asarray([4], dtype=jnp.int32),
            recurrent_indices=None,
        )

        jax.make_jaxpr(lambda hidden: convolution.apply(hidden, batch, None)[0])(
            jnp.ones((4, 4), dtype=jnp.bfloat16)
        )

    def test_dense_layer_prefill_runs_through_native_attention(self):
        model = InklingForCausalLM(
            make_config(num_layers=1, hidden_size=512, head_dim=128),
            MESH,
            dtype=jnp.bfloat16,
        )
        pool = MHATokenToKVPool(
            size=8,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=1,
            mesh=MESH,
        )
        batch = ForwardBatch(
            bid=0,
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=jnp.asarray([1, 2, 3], dtype=jnp.int32),
            req_pool_indices=jnp.asarray([0], dtype=jnp.int32),
            seq_lens=jnp.asarray([3], dtype=jnp.int32),
            out_cache_loc=jnp.asarray([1, 2, 3], dtype=jnp.int32),
            positions=jnp.asarray([0, 1, 2], dtype=jnp.int32),
            attn_backend=NativeAttention(4, 2, MESH),
            cache_loc=jnp.asarray([1, 2, 3], dtype=jnp.int32),
            extend_prefix_lens=jnp.asarray([0], dtype=jnp.int32),
            extend_seq_lens=jnp.asarray([3], dtype=jnp.int32),
        )
        hidden = (
            jnp.arange(3 * 512, dtype=jnp.float32).reshape(3, 512).astype(jnp.bfloat16)
        )
        output, kv_fused, topk_ids, conv_updates = model.model.layers[0](
            batch.positions, hidden, batch, pool
        )
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(kv_fused.ndim, 5)
        self.assertIsNone(topk_ids)
        self.assertIsNone(conv_updates)
        self.assertTrue(np.isfinite(np.asarray(output, dtype=np.float32)).all())

    def test_dense_layer_prompt_then_decode_matches_one_shot(self):
        config = make_config(num_layers=1, hidden_size=512, head_dim=128)
        layer = InklingForCausalLM(config, MESH, dtype=jnp.bfloat16).model.layers[0]
        hidden = (
            jnp.arange(4 * 512, dtype=jnp.float32).reshape(4, 512) / 100
        ).astype(jnp.bfloat16)

        def make_pool():
            return MHATokenToKVPool(
                size=8,
                page_size=1,
                dtype=jnp.bfloat16,
                head_num=2,
                head_dim=128,
                layer_num=1,
                mesh=MESH,
            )

        def make_batch(mode, positions, out_cache_loc, cache_loc, seq_len):
            token_count = len(positions)
            return ForwardBatch(
                bid=0,
                forward_mode=mode,
                batch_size=1,
                input_ids=jnp.arange(token_count, dtype=jnp.int32),
                req_pool_indices=jnp.asarray([0], dtype=jnp.int32),
                seq_lens=jnp.asarray([seq_len], dtype=jnp.int32),
                out_cache_loc=jnp.asarray(out_cache_loc, dtype=jnp.int32),
                positions=jnp.asarray(positions, dtype=jnp.int32),
                attn_backend=NativeAttention(4, 2, MESH),
                cache_loc=jnp.asarray(cache_loc, dtype=jnp.int32),
                extend_prefix_lens=jnp.asarray(
                    [seq_len - token_count], dtype=jnp.int32
                ),
                extend_seq_lens=jnp.asarray([token_count], dtype=jnp.int32),
                recurrent_indices=jnp.asarray([1], dtype=jnp.int32),
            )

        def zero_states():
            return [
                jnp.zeros((2, 256, 3), dtype=jnp.float32),
                jnp.zeros((2, 256, 3), dtype=jnp.float32),
                jnp.zeros((2, 512, 3), dtype=jnp.float32),
                jnp.zeros((2, 512, 3), dtype=jnp.float32),
            ]

        one_shot_batch = make_batch(
            ForwardMode.EXTEND, [0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4], 4
        )
        one_shot, _, _, _ = layer(
            one_shot_batch.positions,
            hidden,
            one_shot_batch,
            make_pool(),
            zero_states(),
        )

        split_pool = make_pool()
        prompt_batch = make_batch(
            ForwardMode.EXTEND, [0, 1, 2], [1, 2, 3], [1, 2, 3], 3
        )
        _, prompt_kv, _, states = layer(
            prompt_batch.positions,
            hidden[:3],
            prompt_batch,
            split_pool,
            zero_states(),
        )
        split_pool.replace_buffer([prompt_kv])
        decode_batch = make_batch(ForwardMode.DECODE, [3], [4], [1, 2, 3, 4], 4)
        decoded, _, _, _ = layer(
            decode_batch.positions,
            hidden[3:],
            decode_batch,
            split_pool,
            states,
        )

        np.testing.assert_allclose(
            np.asarray(decoded, dtype=np.float32),
            np.asarray(one_shot[-1:], dtype=np.float32),
            rtol=0,
            atol=0,
        )

    def test_nvfp4_experts_decode_directly_into_epmoe_layout(self):
        config = make_config(intermediate_size=16)
        model = InklingForCausalLM(config, MESH, dtype=jnp.bfloat16)
        experts = model.model.layers[2].mlp.experts
        expert_count = config.text_config.n_routed_experts
        hidden_size = config.text_config.hidden_size
        intermediate_size = config.text_config.intermediate_size
        rng = np.random.default_rng(7)
        packed_w13 = rng.integers(
            0,
            256,
            size=(expert_count, 2 * intermediate_size, hidden_size // 2),
            dtype=np.uint8,
        )
        packed_w2 = rng.integers(
            0,
            256,
            size=(expert_count, hidden_size, intermediate_size // 2),
            dtype=np.uint8,
        )
        w13_scale = np.ones(
            (expert_count, 2 * intermediate_size, hidden_size // 16), dtype=np.float32
        )
        w2_scale = np.ones(
            (expert_count, hidden_size, intermediate_size // 16), dtype=np.float32
        )
        w13_scale2 = np.linspace(0.5, 1.2, expert_count, dtype=np.float32)
        w2_scale2 = np.linspace(0.7, 1.4, expert_count, dtype=np.float32)
        prefix = "model.llm.layers.2.mlp.experts"
        tensors = {
            f"{prefix}.w13_weight": packed_w13,
            f"{prefix}.w13_weight.scale": w13_scale,
            f"{prefix}.w13_weight.scale2": w13_scale2,
            f"{prefix}.w2_weight": packed_w2,
            f"{prefix}.w2_weight.scale": w2_scale,
            f"{prefix}.w2_weight.scale2": w2_scale2,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_file(tensors, checkpoint)
            weight_info = {name: [{"file": str(checkpoint)}] for name in tensors}
            with SequentialSafetensorManager() as file_manager:
                model._load_layer_experts(file_manager, weight_info, 2, experts)

        decoded_w13 = decode_nvfp4_numpy(packed_w13, w13_scale, w13_scale2)
        decoded_w2 = decode_nvfp4_numpy(packed_w2, w2_scale, w2_scale2)
        expected_gate = np.transpose(decoded_w13[:, 0::2, :], (0, 2, 1)).astype(
            np.float32
        )
        expected_up = np.transpose(decoded_w13[:, 1::2, :], (0, 2, 1)).astype(
            np.float32
        )
        expected_down = np.transpose(decoded_w2, (0, 2, 1)).astype(np.float32)
        np.testing.assert_array_equal(
            np.asarray(experts.wi_0.value, dtype=np.float32), expected_gate
        )
        np.testing.assert_array_equal(
            np.asarray(experts.wi_1.value, dtype=np.float32), expected_up
        )
        np.testing.assert_array_equal(
            np.asarray(experts.wo.value, dtype=np.float32), expected_down
        )

        hidden = jnp.arange(2 * hidden_size, dtype=jnp.float32).reshape(2, hidden_size) / 50
        routes = jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32)
        weights = jnp.asarray([[0.6, 0.4], [0.25, 0.75]], dtype=jnp.float32)
        actual = experts(
            hidden.astype(jnp.bfloat16),
            weights,
            routes,
            out_sharding=jax.sharding.NamedSharding(MESH, jax.sharding.PartitionSpec("data", None)),
        )
        expected = np.zeros((2, hidden_size), dtype=np.float32)
        hidden_numpy = np.asarray(hidden, dtype=np.float32)
        for token in range(2):
            for slot in range(2):
                expert_id = int(routes[token, slot])
                gate = hidden_numpy[token] @ expected_gate[expert_id]
                up = hidden_numpy[token] @ expected_up[expert_id]
                activated = np.asarray(jax.nn.silu(jnp.asarray(gate))) * up
                expected[token] += (
                    activated @ expected_down[expert_id] * float(weights[token, slot])
                )
        actual_numpy = np.asarray(actual, dtype=np.float32)
        cosine = np.vdot(actual_numpy.ravel(), expected.ravel()) / (
            np.linalg.norm(actual_numpy) * np.linalg.norm(expected)
        )
        normalized_error = np.sqrt(np.mean(np.square(actual_numpy - expected))) / np.sqrt(
            np.mean(np.square(expected))
        )
        self.assertGreater(float(cosine), 0.999)
        self.assertLess(float(normalized_error), 0.02)


if __name__ == "__main__":
    unittest.main()
