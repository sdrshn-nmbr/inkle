from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


INKLING_SMALL_DSPARK_REPO = "RadixArk/Inkling-Small-DSpark"
INKLING_SMALL_DSPARK_REVISION = "736501c3901cfc6bbb53ba382781eb0e5d9ad66a"


def _positive_int(config: Any, name: str) -> int:
    value = int(getattr(config, name))
    if value <= 0:
        raise ValueError(f"DSpark {name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class DSparkDraftConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int
    markov_rank: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: Mapping[str, Any] | None
    attention_bias: bool
    layer_types: tuple[str, ...]
    confidence_head_with_markov: bool

    @property
    def context_width(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    @classmethod
    def from_hf_config(cls, config: Any) -> DSparkDraftConfig:
        dflash = getattr(config, "dflash_config", None)
        if not isinstance(dflash, Mapping):
            raise ValueError("DSpark config requires a dflash_config mapping")

        target_layer_ids = tuple(int(value) for value in dflash.get("target_layer_ids", ()))
        if not target_layer_ids:
            raise ValueError("DSpark config requires dflash_config.target_layer_ids")
        if len(set(target_layer_ids)) != len(target_layer_ids):
            raise ValueError(f"DSpark target_layer_ids must be unique, got {target_layer_ids}")
        if any(value < 0 for value in target_layer_ids):
            raise ValueError(
                f"DSpark target_layer_ids must be non-negative, got {target_layer_ids}"
            )

        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        layer_types = tuple(getattr(config, "layer_types", ("full_attention",) * num_hidden_layers))
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "DSpark layer_types must contain one value per draft layer, "
                f"got {len(layer_types)} values for {num_hidden_layers} layers"
            )
        unsupported_layer_types = set(layer_types) - {"full_attention"}
        if unsupported_layer_types:
            raise ValueError(
                "The JAX DSpark draft currently supports full-attention layers only, "
                f"got {sorted(unsupported_layer_types)}"
            )

        num_attention_heads = _positive_int(config, "num_attention_heads")
        num_key_value_heads = _positive_int(config, "num_key_value_heads")
        if num_attention_heads % num_key_value_heads:
            raise ValueError(
                "DSpark num_attention_heads must be divisible by num_key_value_heads, "
                f"got {num_attention_heads} and {num_key_value_heads}"
            )

        hidden_size = _positive_int(config, "hidden_size")
        head_dim = int(getattr(config, "head_dim", hidden_size // num_attention_heads))
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError(
                "DSpark hidden_size must equal num_attention_heads * head_dim, "
                f"got {hidden_size} != {num_attention_heads} * {head_dim}"
            )

        markov_rank = _positive_int(config, "markov_rank")
        markov_head_type = str(getattr(config, "markov_head_type", "vanilla")).lower()
        if markov_head_type != "vanilla":
            raise ValueError(
                "The Inkling-Small DSpark checkpoint requires the vanilla Markov head, "
                f"got {markov_head_type!r}"
            )
        if not bool(getattr(config, "enable_confidence_head", True)):
            raise ValueError("The Inkling-Small DSpark checkpoint requires its confidence head")

        rope_parameters = getattr(config, "rope_parameters", None)
        rope_scaling = rope_parameters
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and isinstance(rope_parameters, Mapping):
            rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            rope_theta = 10_000.0

        mask_token_id = dflash.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError("DSpark config requires dflash_config.mask_token_id")

        return cls(
            hidden_size=hidden_size,
            intermediate_size=_positive_int(config, "intermediate_size"),
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=_positive_int(config, "vocab_size"),
            block_size=_positive_int(config, "block_size"),
            markov_rank=markov_rank,
            target_layer_ids=target_layer_ids,
            mask_token_id=int(mask_token_id),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            rope_theta=float(rope_theta),
            max_position_embeddings=_positive_int(config, "max_position_embeddings"),
            rope_scaling=rope_scaling,
            attention_bias=bool(getattr(config, "attention_bias", False)),
            layer_types=layer_types,
            confidence_head_with_markov=bool(getattr(config, "confidence_head_with_markov", True)),
        )
