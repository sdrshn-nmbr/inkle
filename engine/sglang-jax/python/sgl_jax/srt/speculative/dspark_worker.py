from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.models.dspark import (
    DSparkContextCache,
    DSparkDraftModel,
    DSparkLayerContext,
)
from sgl_jax.srt.speculative.base_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    build_chain_verify_inputs_device,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.scheduler import GenerationBatchResult


@jax.tree_util.register_pytree_node_class
@dataclass
class DSparkDraftInput(EagleDraftInput):
    """Minimal cross-round state for DSpark's independent proposal model."""

    requires_full_draft_state = False

    def filter_batch(
        self,
        new_indices: np.ndarray,
        has_been_filtered: bool = True,
    ) -> None:
        del has_been_filtered
        indices = np.asarray(new_indices, dtype=np.int32)
        for field in (
            "hidden_states",
            "verified_id",
            "allocate_lens",
            "new_seq_lens",
            "accept_length",
            "accept_length_cpu",
            "future_indices",
        ):
            value = getattr(self, field, None)
            if value is not None:
                setattr(self, field, np.asarray(value)[indices])

    def merge_batch(self, other: EagleDraftInput) -> None:
        if not isinstance(other, DSparkDraftInput):
            raise TypeError(
                f"DSPARK_STATE_TYPE_MISMATCH expected=DSparkDraftInput got={type(other).__name__}"
            )
        for field in (
            "hidden_states",
            "verified_id",
            "allocate_lens",
            "new_seq_lens",
            "accept_length",
            "accept_length_cpu",
            "future_indices",
        ):
            left = getattr(self, field, None)
            right = getattr(other, field, None)
            if left is None and right is None:
                continue
            if left is None or right is None:
                raise ValueError(f"DSPARK_STATE_FIELD_MISMATCH field={field}")
            setattr(
                self,
                field,
                np.concatenate((np.asarray(left), np.asarray(right))),
            )


@dataclass(frozen=True)
class DSparkChainDecision:
    verify_tokens: int
    per_request_verify_tokens: np.ndarray
    confidence: np.ndarray


def choose_confidence_chain_length(
    confidence_logits: jax.Array | np.ndarray,
    *,
    threshold: float | None,
) -> DSparkChainDecision:
    logits = np.asarray(jax.device_get(confidence_logits), dtype=np.float32)
    confidence = 1.0 / (1.0 + np.exp(-logits))
    maximum = logits.shape[1] + 1
    if threshold is None:
        per_request = np.full(logits.shape[0], maximum, dtype=np.int32)
    else:
        survival = np.cumprod(confidence, axis=1)
        per_request = 1 + np.sum(survival >= threshold, axis=1, dtype=np.int32)
        per_request = np.clip(per_request, 2, maximum)
    return DSparkChainDecision(
        verify_tokens=int(np.min(per_request, initial=maximum)),
        per_request_verify_tokens=per_request,
        confidence=confidence,
    )


def build_dspark_cache_loc(
    req_to_token: np.ndarray,
    request_indices: np.ndarray,
    seq_lens: np.ndarray,
    allocated_lens: np.ndarray,
    *,
    page_size: int,
    capacity: int,
) -> np.ndarray:
    cache_loc = np.zeros(capacity, dtype=np.int32)
    cursor = 0
    for request_index, seq_len, allocated_len in zip(
        request_indices,
        seq_lens,
        allocated_lens,
        strict=True,
    ):
        if seq_len <= 0:
            continue
        aligned_len = (int(allocated_len) + page_size - 1) // page_size * page_size
        if cursor + aligned_len > cache_loc.size:
            raise ValueError(
                f"DSPARK_CACHE_LOC_OVERFLOW needed={cursor + aligned_len} capacity={cache_loc.size}"
            )
        cache_loc[cursor : cursor + int(allocated_len)] = req_to_token[
            request_index,
            : int(allocated_len),
        ]
        cursor += aligned_len
    return cache_loc


def compact_dspark_verified_ids(
    verified_ids: jax.Array | np.ndarray,
    accept_lens: jax.Array | np.ndarray,
    selector: np.ndarray,
    *,
    batch_size: int,
    verify_tokens: int,
) -> np.ndarray:
    verified = np.asarray(jax.device_get(verified_ids), dtype=np.int32).reshape(
        batch_size,
        verify_tokens,
    )
    accepted = np.asarray(jax.device_get(accept_lens), dtype=np.int32)
    selected_accept = accepted[selector]
    if np.any(selected_accept <= 0) or np.any(selected_accept > verify_tokens):
        raise ValueError(
            "DSPARK_ACCEPT_LENGTH_INVALID "
            f"accepted={selected_accept.tolist()} verify_tokens={verify_tokens}"
        )
    return verified[selector, selected_accept - 1]


@nnx.jit
def _encode_context(
    model: DSparkDraftModel,
    target_hidden_states: jax.Array,
    position_ids: jax.Array,
) -> DSparkContextCache:
    return model.encode_context(target_hidden_states, position_ids)


@nnx.jit(
    static_argnames=("target_vocab_size", "logits_mup_width_multiplier"),
)
def _run_proposal(
    model: DSparkDraftModel,
    noise_embeddings: jax.Array,
    context: DSparkContextCache,
    draft_position_ids: jax.Array,
    attention_mask: jax.Array,
    target_lm_head: jax.Array,
    anchor_token_ids: jax.Array,
    *,
    target_vocab_size: int,
    logits_mup_width_multiplier: float,
):
    hidden_states = model.forward_cached(
        noise_embeddings,
        context,
        draft_position_ids,
        attention_mask,
    )
    base_logits = model.compute_base_logits(
        hidden_states,
        target_lm_head,
        target_vocab_size=target_vocab_size,
        logits_mup_width_multiplier=logits_mup_width_multiplier,
    )
    return model.greedy_propose(base_logits, hidden_states, anchor_token_ids)


class DSparkDraftWorker(BaseDraftWorker):
    def __init__(self, server_args, target_worker: ModelWorker) -> None:
        if server_args.dp_size != 1:
            raise ValueError("DSPARK_DP_UNSUPPORTED dp_size must be 1")
        if server_args.speculative_draft_model_path is None:
            raise ValueError("DSPARK_DRAFT_MODEL_REQUIRED")

        self.server_args = server_args
        self.target_worker = target_worker
        self.mesh = target_worker.mesh
        self.gamma = int(server_args.speculative_dspark_block_size)
        self.maximum_verify_tokens = self.gamma + 1
        self.context_window = int(server_args.speculative_dspark_context_window)
        self.confidence_threshold = server_args.speculative_dspark_confidence_threshold
        EagleDraftInput.ALLOC_LEN_PER_DECODE = self.maximum_verify_tokens

        target_model = target_worker.model_runner.model
        if not hasattr(target_model, "set_dspark_layers_to_capture"):
            raise ValueError(
                "DSPARK_TARGET_UNSUPPORTED target model does not expose DSpark hidden states"
            )
        target_model.set_dspark_layers_to_capture()
        target_worker.model_runner.initialize_jit()

        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.speculative_draft_model_path,
            model_revision=server_args.speculative_draft_model_revision,
            is_draft_model=True,
        )
        with jax.set_mesh(self.mesh):
            self.model = DSparkDraftModel(
                self.model_config.hf_config,
                self.mesh,
                dtype=jnp.bfloat16,
            )
        if self.model.config.block_size != self.gamma:
            raise ValueError(
                "DSPARK_BLOCK_SIZE_MISMATCH "
                f"checkpoint={self.model.config.block_size} runtime={self.gamma}"
            )
        with jax.set_mesh(self.mesh):
            self.model.load_weights(self.model_config)

        self.target_embed, target_lm_head = target_model.get_embed_and_head_modules()
        self.target_lm_head = target_lm_head.embedding.value
        target_config = target_worker.model_config.hf_text_config
        self.target_vocab_size = int(target_worker.model_config.vocab_size)
        self.logits_mup_width_multiplier = float(target_config.logits_mup_width_multiplier)
        self._draft_model_runner = target_worker.model_runner
        _, self.bs_paddings, self.cache_loc_paddings = target_worker.get_precompile_paddings()

        req_pool_size = target_worker.model_runner.req_to_token_pool.size
        self.req_pool_size = req_pool_size
        self.discard_request_index = req_pool_size
        cache_sharding = NamedSharding(self.mesh, P(None, None, "tensor", None))
        cache_shape = (
            req_pool_size + 1,
            self.context_window,
            self.model.config.num_key_value_heads,
            self.model.config.head_dim,
        )
        self.context_key = tuple(
            jax.device_put(jnp.zeros(cache_shape, dtype=jnp.bfloat16), cache_sharding)
            for _ in self.model.layers
        )
        self.context_value = tuple(
            jax.device_put(jnp.zeros(cache_shape, dtype=jnp.bfloat16), cache_sharding)
            for _ in self.model.layers
        )
        self.rounds = 0
        self.proposed_tokens = 0
        self.verify_tokens = 0
        self.accepted_tokens = 0
        self.verified_requests = 0
        self.last_decision: DSparkChainDecision | None = None
        logger.info(
            "DSPARK_READY gamma=%d verify_tokens=%d context_window=%d confidence_threshold=%s",
            self.gamma,
            self.maximum_verify_tokens,
            self.context_window,
            self.confidence_threshold,
        )

    @property
    def draft_model_runner(self):
        return self._draft_model_runner

    def _append_hidden(
        self,
        hidden_states: jax.Array,
        request_indices: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        if request_indices.size == 0:
            return
        if hidden_states.shape[0] != request_indices.size:
            raise ValueError(
                "DSPARK_HISTORY_SHAPE_MISMATCH "
                f"hidden={hidden_states.shape[0]} requests={request_indices.size}"
            )
        encoded = _encode_context(
            self.model,
            hidden_states[:, None, :],
            jnp.asarray(positions, dtype=jnp.int32)[:, None],
        )
        request_indices_device = jnp.asarray(request_indices, dtype=jnp.int32)
        ring_positions = jnp.asarray(positions % self.context_window, dtype=jnp.int32)
        cache_row_sharding = NamedSharding(self.mesh, P(None, "tensor", None))
        for layer_id, layer_context in enumerate(encoded.layers):
            key_rows = jax.sharding.reshard(layer_context.key[:, 0], cache_row_sharding)
            value_rows = jax.sharding.reshard(layer_context.value[:, 0], cache_row_sharding)
            self.context_key = (
                self.context_key[:layer_id]
                + (
                    self.context_key[layer_id]
                    .at[request_indices_device, ring_positions]
                    .set(key_rows),
                )
                + self.context_key[layer_id + 1 :]
            )
            self.context_value = (
                self.context_value[:layer_id]
                + (
                    self.context_value[layer_id]
                    .at[request_indices_device, ring_positions]
                    .set(value_rows),
                )
                + self.context_value[layer_id + 1 :]
            )

    def _append_prefill_hidden(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
    ) -> None:
        extend_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)
        req_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        lengths = extend_lens[selector]
        total = int(np.sum(lengths))
        hidden_count = hidden_states.shape[0]
        if total > hidden_count:
            raise ValueError(
                f"DSPARK_PREFILL_LAYOUT_MISMATCH real_tokens={total} hidden_rows={hidden_count}"
            )
        request_rows = np.full(hidden_count, self.discard_request_index, dtype=np.int32)
        request_rows[:total] = np.repeat(req_indices[selector], lengths)
        source_positions = np.asarray(
            jax.device_get(model_worker_batch.positions), dtype=np.int32
        ).reshape(-1)
        positions = np.zeros(hidden_count, dtype=np.int32)
        positions[:total] = source_positions[:total]
        self._append_hidden(hidden_states, request_rows, positions)

    def _gather_context(
        self,
        request_indices: np.ndarray,
        verified_seq_lens: np.ndarray,
    ) -> tuple[DSparkContextCache, jax.Array, jax.Array]:
        retained = np.minimum(np.maximum(verified_seq_lens, 0), self.context_window)
        maximum = max(int(np.max(retained, initial=0)), 1)
        context_tokens = min(1 << (maximum - 1).bit_length(), self.context_window)
        offsets = np.arange(context_tokens, dtype=np.int32)
        absolute_positions = verified_seq_lens[:, None] - context_tokens + offsets[None, :]
        oldest = np.maximum(verified_seq_lens - self.context_window, 0)
        valid = (absolute_positions >= oldest[:, None]) & (
            absolute_positions < verified_seq_lens[:, None]
        )
        ring_positions = np.mod(absolute_positions, self.context_window).astype(np.int32)
        safe_positions = np.where(valid, ring_positions, 0)
        req = jnp.asarray(request_indices, dtype=jnp.int32)[:, None]
        pos = jnp.asarray(safe_positions, dtype=jnp.int32)
        context_sharding = NamedSharding(self.mesh, P("data", None, "tensor", None))
        context = DSparkContextCache(
            layers=tuple(
                DSparkLayerContext(
                    key=jax.sharding.reshard(key[req, pos], context_sharding),
                    value=jax.sharding.reshard(value[req, pos], context_sharding),
                )
                for key, value in zip(self.context_key, self.context_value)
            )
        )
        return (
            context,
            jnp.asarray(valid, dtype=jnp.bool_),
            jnp.asarray(verified_seq_lens, dtype=jnp.int32),
        )

    def _prepare_target_cache_loc(
        self,
        model_worker_batch: ModelWorkerBatch,
        spec_info: EagleDraftInput,
    ) -> None:
        req_to_token_pool, _ = self.target_worker.get_memory_pool()
        batch_size = len(model_worker_batch.seq_lens)
        try:
            bucket_index = self.bs_paddings.index(batch_size)
        except ValueError as error:
            raise ValueError(
                f"DSPARK_BATCH_BUCKET_MISSING batch_size={batch_size} buckets={self.bs_paddings}"
            ) from error
        model_worker_batch.cache_loc = build_dspark_cache_loc(
            np.asarray(req_to_token_pool.req_to_token, dtype=np.int32),
            np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32),
            np.asarray(model_worker_batch.seq_lens, dtype=np.int32),
            np.asarray(spec_info.allocate_lens, dtype=np.int32),
            page_size=self.server_args.page_size,
            capacity=self.cache_loc_paddings[bucket_index],
        )

    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        spec_info = model_worker_batch.spec_info_padded
        if not isinstance(spec_info, EagleDraftInput):
            raise TypeError(f"DSPARK_EXPECTED_DRAFT_INPUT got={type(spec_info).__name__}")
        spec_info.prepare_for_draft_decode(model_worker_batch, topk=1, num_steps=self.gamma)
        self._prepare_target_cache_loc(model_worker_batch, spec_info)

        seq_lens = np.asarray(model_worker_batch.seq_lens, dtype=np.int32)
        verified_seq_lens = np.maximum(seq_lens - 1, 0)
        batch_size = seq_lens.shape[0]
        request_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)
        anchor = np.asarray(jax.device_get(spec_info.verified_id), dtype=np.int32).reshape(-1)
        if anchor.shape[0] < batch_size:
            anchor = np.pad(anchor, (0, batch_size - anchor.shape[0]))
        anchor = anchor[:batch_size]

        context, context_valid, draft_start = self._gather_context(
            request_indices,
            verified_seq_lens,
        )
        draft_ids = (
            jnp.full(
                (batch_size, self.gamma),
                self.model.config.mask_token_id,
                dtype=jnp.int32,
            )
            .at[:, 0]
            .set(jnp.asarray(anchor, dtype=jnp.int32))
        )
        noise_embeddings = self.target_embed(draft_ids)
        draft_positions = draft_start[:, None] + jnp.arange(self.gamma, dtype=jnp.int32)[None, :]
        attention_mask = jnp.concatenate(
            (
                jnp.broadcast_to(
                    context_valid[:, None, :],
                    (batch_size, self.gamma, context_valid.shape[1]),
                ),
                jnp.ones((batch_size, self.gamma, self.gamma), dtype=jnp.bool_),
            ),
            axis=-1,
        )
        proposal = _run_proposal(
            self.model,
            noise_embeddings,
            context,
            draft_positions,
            attention_mask,
            self.target_lm_head,
            jnp.asarray(anchor, dtype=jnp.int32),
            target_vocab_size=self.target_vocab_size,
            logits_mup_width_multiplier=self.logits_mup_width_multiplier,
        )

        real_bs = int(model_worker_batch.real_bs)
        confidence_logits = np.asarray(
            jax.device_get(proposal.confidence_logits),
            dtype=np.float32,
        )[:real_bs]
        decision = choose_confidence_chain_length(
            confidence_logits,
            threshold=self.confidence_threshold,
        )
        num_verify_tokens = decision.verify_tokens
        anchor_device = jax.sharding.reshard(
            jnp.asarray(anchor, dtype=jnp.int32),
            NamedSharding(self.mesh, P()),
        )
        proposal_ids = jax.sharding.reshard(
            proposal.token_ids,
            NamedSharding(self.mesh, P()),
        )
        verified_seq_lens_device = jax.sharding.reshard(
            jnp.asarray(verified_seq_lens, dtype=jnp.int32),
            NamedSharding(self.mesh, P()),
        )
        packed = build_chain_verify_inputs_device(
            anchor_device,
            proposal_ids,
            verified_seq_lens_device,
            num_verify_tokens,
            batch_size,
        )
        packed = jax.device_put(packed, NamedSharding(self.mesh, P()))
        model_worker_batch.spec_info_padded = EagleVerifyInput(
            draft_token=packed[0],
            custom_mask=None,
            positions=packed[1],
            retrive_index=packed[2].reshape(batch_size, num_verify_tokens),
            retrive_next_token=packed[3].reshape(batch_size, num_verify_tokens),
            retrive_next_sibling=packed[4].reshape(batch_size, num_verify_tokens),
            retrive_cum_len=None,
            spec_steps=num_verify_tokens - 1,
            topk=1,
            draft_token_num=num_verify_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            seq_lens_cpu=seq_lens,
        )
        self.rounds += 1
        self.proposed_tokens += real_bs * self.gamma
        self.verify_tokens += real_bs * num_verify_tokens
        self.last_decision = decision
        if self.rounds == 1 or self.rounds % 100 == 0:
            logger.info(
                "DSPARK_CHAIN round=%d real_bs=%d verify_tokens=%d "
                "predicted_range=[%d,%d] mean_confidence=%.4f",
                self.rounds,
                real_bs,
                num_verify_tokens,
                int(np.min(decision.per_request_verify_tokens)),
                int(np.max(decision.per_request_verify_tokens)),
                float(np.mean(decision.confidence)),
            )

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
        self._append_prefill_hidden(model_worker_batch, hidden_states)
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        verified = np.asarray(jax.device_get(next_token_ids), dtype=np.int32)[selector]
        model_worker_batch.spec_info_padded = DSparkDraftInput(
            verified_id=verified,
            allocate_lens=np.asarray(model_worker_batch.seq_lens, dtype=np.int32)[selector],
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    def record_verified_hidden(
        self,
        *,
        request_indices: np.ndarray,
        selector: np.ndarray,
        positions: jax.Array,
        hidden_states: jax.Array,
        accept_lens: jax.Array,
        verify_tokens: int,
    ) -> None:
        batch_size = request_indices.shape[0]
        hidden = hidden_states.reshape(batch_size, verify_tokens, -1)
        position_matrix = positions.reshape(batch_size, verify_tokens)
        accept = np.asarray(jax.device_get(accept_lens), dtype=np.int32)
        accepted_mask = np.zeros((batch_size, verify_tokens), dtype=np.bool_)
        for row in selector:
            accepted_mask[row, : int(accept[row])] = True
        request_matrix = np.broadcast_to(request_indices[:, None], (batch_size, verify_tokens))
        request_rows = np.where(
            accepted_mask,
            request_matrix,
            self.discard_request_index,
        ).reshape(-1)
        self._append_hidden(
            hidden.reshape(batch_size * verify_tokens, -1),
            request_rows,
            np.asarray(jax.device_get(position_matrix), dtype=np.int32).reshape(-1),
        )

    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output: GenerationBatchResult,
    ) -> None:
        return None


class DSparkWorker(BaseSpecWorker):
    def __init__(self, server_args, target_worker: ModelWorker) -> None:
        super().__init__(
            server_args,
            target_worker,
            DSparkDraftWorker(server_args, target_worker),
        )

    def verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        cur_allocate_lens: jax.Array,
    ) -> GenerationBatchResult:
        spec_info = model_worker_batch.spec_info_padded
        if not isinstance(spec_info, EagleVerifyInput):
            raise TypeError(f"DSPARK_EXPECTED_VERIFY_INPUT got={type(spec_info).__name__}")
        verify_tokens = spec_info.draft_token_num
        request_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32).copy()
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32).copy()
        result = super().verify(model_worker_batch, cur_allocate_lens)
        accepted = np.asarray(result.accept_lens, dtype=np.int32)[selector]
        self.draft_worker.accepted_tokens += int(np.sum(accepted))
        self.draft_worker.verified_requests += int(selector.size)
        if self.draft_worker.rounds == 1 or self.draft_worker.rounds % 100 == 0:
            logger.info(
                "DSPARK_ACCEPT round=%d real_bs=%d accepted_mean=%.4f "
                "accepted_range=[%d,%d] cumulative_mean=%.4f",
                self.draft_worker.rounds,
                selector.size,
                float(np.mean(accepted)),
                int(np.min(accepted)),
                int(np.max(accepted)),
                self.draft_worker.accepted_tokens / max(self.draft_worker.verified_requests, 1),
            )
        next_state = result.next_draft_input
        compact_verified_ids = compact_dspark_verified_ids(
            next_state.verified_id,
            result.accept_lens,
            selector,
            batch_size=request_indices.shape[0],
            verify_tokens=verify_tokens,
        )
        dspark_state = DSparkDraftInput(
            verified_id=compact_verified_ids,
            allocate_lens=next_state.allocate_lens,
            new_seq_lens=np.asarray(next_state.new_seq_lens, dtype=np.int32)[selector],
            capture_hidden_mode=next_state.capture_hidden_mode,
        )
        result.next_draft_input = dspark_state
        model_worker_batch.spec_info_padded = dspark_state
        self.draft_worker.record_verified_hidden(
            request_indices=request_indices,
            selector=selector,
            positions=model_worker_batch.positions,
            hidden_states=result.logits_output.hidden_states,
            accept_lens=result.accept_lens,
            verify_tokens=verify_tokens,
        )
        return result
