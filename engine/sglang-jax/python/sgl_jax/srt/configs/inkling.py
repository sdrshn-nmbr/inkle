import jax.numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.mem_cache.recurrent_state_pool import (
    LinearRecurrentStateParams,
    RecurrentStateDType,
)


class InklingTextConfig(PretrainedConfig):
    model_type = "inkling_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        hidden_size: int = 6144,
        num_hidden_layers: int = 66,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        d_rel: int = 16,
        rel_extent: int = 1024,
        rms_norm_eps: float = 1e-6,
        local_layer_ids: list[int] | None = None,
        dense_mlp_idx: int = 2,
        sconv_kernel_size: int = 4,
        dense_intermediate_size: int = 24576,
        intermediate_size: int = 3072,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 16,
        sliding_window_size: int = 512,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 2,
        route_scale: float = 8.0,
        logits_mup_width_multiplier: float = 24.0,
        log_scaling_n_floor: int | None = 128000,
        log_scaling_alpha: float = 0.1,
        final_logit_softcapping: float | None = None,
        model_max_length: int = 1048576,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.rms_norm_eps = rms_norm_eps
        self.local_layer_ids = list(local_layer_ids or [])
        self.dense_mlp_idx = dense_mlp_idx
        self.sconv_kernel_size = sconv_kernel_size
        self.dense_intermediate_size = dense_intermediate_size
        self.intermediate_size = intermediate_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.sliding_window_size = sliding_window_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.route_scale = route_scale
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.final_logit_softcapping = final_logit_softcapping
        self.model_max_length = model_max_length
        self.max_position_embeddings = model_max_length
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return list(range(self.num_hidden_layers))

    @property
    def linear_state_params(self) -> LinearRecurrentStateParams:
        conv_channel_sizes = []
        for layer_id in range(self.num_hidden_layers):
            kv_heads = (
                self.swa_num_key_value_heads
                if layer_id in self.local_layer_ids
                else self.num_key_value_heads
            )
            kv_width = kv_heads * self.head_dim
            conv_channel_sizes.append(
                [kv_width, kv_width, self.hidden_size, self.hidden_size]
            )
        return LinearRecurrentStateParams(
            layers=list(range(self.num_hidden_layers)),
            num_heads=self.num_attention_heads,
            head_dim=1,
            conv_kernel_size=self.sconv_kernel_size,
            dtype=RecurrentStateDType(conv=jnp.float32, temporal=jnp.float32),
            conv_channel_sizes=conv_channel_sizes,
            has_temporal_state=False,
        )


class InklingConfig(PretrainedConfig):
    model_type = "inkling_mm_model"
    sub_configs = {"text_config": InklingTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict | InklingTextConfig | None = None,
        vision_config: dict | None = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = InklingTextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["InklingConfig", "InklingTextConfig"]
