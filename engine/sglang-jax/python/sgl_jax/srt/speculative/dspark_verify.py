from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.speculative.eagle_util import verify_compact_chain_greedy

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flax import nnx
    from jax.sharding import Mesh

    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


def gather_compact_cache_locations(
    req_to_token: np.ndarray,
    request_indices: np.ndarray,
    positions: np.ndarray,
    verify_lens: np.ndarray,
) -> np.ndarray:
    mapping = np.asarray(req_to_token, dtype=np.int32)
    requests = np.asarray(request_indices, dtype=np.int32)
    token_positions = np.asarray(positions, dtype=np.int32)
    lengths = np.asarray(verify_lens, dtype=np.int32)
    if requests.shape != lengths.shape:
        raise ValueError(
            "DSPARK_REQUEST_LAYOUT_SHAPE_MISMATCH "
            f"requests={requests.shape} lengths={lengths.shape}"
        )
    if np.any(lengths < 0):
        raise ValueError(f"DSPARK_RESERVED_CACHE_LENGTH_INVALID lengths={lengths.tolist()}")
    expected_tokens = int(np.sum(lengths))
    if token_positions.shape != (expected_tokens,):
        raise ValueError(
            "DSPARK_POSITION_LAYOUT_SHAPE_MISMATCH "
            f"positions={token_positions.shape} expected={(expected_tokens,)}"
        )
    compact_requests = np.repeat(requests, lengths)
    if np.any((compact_requests < 0) | (compact_requests >= mapping.shape[0])):
        raise ValueError(
            "DSPARK_REQUEST_INDEX_OUT_OF_RANGE "
            f"requests={compact_requests.tolist()} capacity={mapping.shape[0]}"
        )
    if np.any((token_positions < 0) | (token_positions >= mapping.shape[1])):
        raise ValueError(
            "DSPARK_TOKEN_POSITION_OUT_OF_RANGE "
            f"positions={token_positions.tolist()} capacity={mapping.shape[1]}"
        )
    return mapping[compact_requests, token_positions]


@register_pytree_node_class
@dataclass(frozen=True)
class DSparkVerifyLayout:
    verify_lens: np.ndarray
    query_indptr: np.ndarray
    compact_to_strided: np.ndarray
    strided_to_compact: np.ndarray
    graph_num_tokens: int

    @classmethod
    def from_verify_lens(
        cls,
        verify_lens: np.ndarray,
        *,
        maximum_verify_tokens: int,
    ) -> DSparkVerifyLayout:
        lengths = np.asarray(verify_lens, dtype=np.int32)
        if lengths.ndim != 1 or np.any((lengths < 0) | (lengths > maximum_verify_tokens)):
            raise ValueError(
                "DSPARK_VERIFY_LAYOUT_LENGTH_INVALID "
                f"shape={lengths.shape} lengths={lengths.tolist()}"
            )
        query_indptr = np.concatenate(
            (np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32))
        )
        compact_to_strided = np.concatenate(
            [
                row * maximum_verify_tokens + np.arange(length, dtype=np.int32)
                for row, length in enumerate(lengths)
            ]
        )
        strided_to_compact = np.full((lengths.size, maximum_verify_tokens), -1, dtype=np.int32)
        strided_to_compact.reshape(-1)[compact_to_strided] = np.arange(
            compact_to_strided.size, dtype=np.int32
        )
        return cls(
            verify_lens=lengths,
            query_indptr=query_indptr,
            compact_to_strided=compact_to_strided,
            strided_to_compact=strided_to_compact,
            graph_num_tokens=int(query_indptr[-1]),
        )

    def tree_flatten(self):
        return (
            self.verify_lens,
            self.query_indptr,
            self.compact_to_strided,
            self.strided_to_compact,
        ), self.graph_num_tokens

    @classmethod
    def tree_unflatten(cls, graph_num_tokens, children):
        return cls(*children, graph_num_tokens)


@register_pytree_node_class
@dataclass
class DSparkVerifyInput:
    verify_ids_strided: jax.Array
    compact_input_ids: jax.Array
    compact_positions: jax.Array
    layout: DSparkVerifyLayout
    seq_lens_cpu: np.ndarray
    seq_lens_sum: int
    max_verify_tokens: int
    capture_hidden_mode: CaptureHiddenMode
    topk: int = 1
    allocate_lens: jax.Array | np.ndarray | None = None
    hidden_states: jax.Array | None = None
    custom_mask: jax.Array | None = None

    @property
    def draft_token(self) -> jax.Array:
        return self.compact_input_ids

    @property
    def positions(self) -> jax.Array:
        return self.compact_positions

    @property
    def dense_draft_token(self) -> jax.Array:
        return self.verify_ids_strided.reshape(-1)

    @property
    def verify_lens(self) -> np.ndarray:
        return self.layout.verify_lens

    @property
    def draft_token_num(self) -> int:
        return self.max_verify_tokens

    @property
    def spec_steps(self) -> int:
        return self.max_verify_tokens - 1

    def is_draft_input(self) -> bool:
        return False

    def is_verify_input(self) -> bool:
        return True

    def get_spec_adjust_token_coefficient(self) -> int:
        return self.max_verify_tokens

    def get_logical_token_num(self, bs: int) -> np.ndarray:
        return np.ones(bs, dtype=np.int32)

    def get_allocated_token_num(self) -> np.ndarray | None:
        return None

    def get_verify_token_num(self, bs: int) -> int:
        del bs
        return self.layout.graph_num_tokens

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None:
        raise NotImplementedError("DSparkVerifyInput is consumed within one round")

    def merge_batch(self, other) -> None:
        raise NotImplementedError("DSparkVerifyInput is consumed within one round")

    def prepare_for_verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        page_size: int,
        target_worker,
    ) -> None:
        del page_size
        selector = model_worker_batch.logits_indices_selector
        req_to_token_pool, _ = target_worker.get_memory_pool()
        compact_positions = np.asarray(jax.device_get(self.compact_positions), dtype=np.int32)
        compact_cache_locations = gather_compact_cache_locations(
            req_to_token_pool.req_to_token,
            model_worker_batch.req_pool_indices,
            compact_positions,
            self.layout.verify_lens,
        )
        model_worker_batch.seq_lens[selector] -= 1
        model_worker_batch.input_ids = self.compact_input_ids
        model_worker_batch.positions = self.compact_positions
        model_worker_batch.return_hidden_states = False
        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.spec_info_padded = self
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        model_worker_batch.extend_seq_lens = self.layout.verify_lens
        model_worker_batch.out_cache_loc = compact_cache_locations
        if os.getenv("SGL_JAX_DSPARK_DIAGNOSTIC_DIR"):
            self._diagnostic_cache_locations = compact_cache_locations.copy()
        if np.any(model_worker_batch.out_cache_loc <= 0):
            raise ValueError(
                "DSPARK_COMPACT_CACHE_LOCATION_INVALID "
                f"locations={model_worker_batch.out_cache_loc.tolist()}"
            )
        if int(np.count_nonzero(self.layout.verify_lens)) == 1:
            logger.info(
                "DSPARK_VERIFY_LAYOUT requests=%s seq_lens=%s verify_lens=%s "
                "positions=%s cache_locations=%s tokens=%s",
                np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32).tolist(),
                np.asarray(self.seq_lens_cpu, dtype=np.int32).tolist(),
                self.layout.verify_lens.tolist(),
                compact_positions.tolist(),
                compact_cache_locations.tolist(),
                np.asarray(jax.device_get(self.compact_input_ids), dtype=np.int32).tolist(),
            )

    def sample(
        self,
        model_worker_batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        rng: nnx.Rngs,
        mesh: Mesh,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del rng
        if not model_worker_batch.sampling_info.is_all_greedy:
            raise ValueError("DSPARK_COMPACT_VERIFY_REQUIRES_GREEDY_SAMPLING")
        target_predict = np.asarray(
            jax.device_get(jnp.argmax(logits_output.next_token_logits, axis=-1)),
            dtype=np.int32,
        )
        if os.getenv("SGL_JAX_DSPARK_DIAGNOSTIC_DIR"):
            self._diagnostic_raw_logits = np.asarray(
                jax.device_get(logits_output.next_token_logits)
            )
            self._diagnostic_hidden_states = np.asarray(jax.device_get(logits_output.hidden_states))
            self._diagnostic_target_predict = target_predict.copy()
            self._diagnostic_layers_topk_ids = getattr(
                logits_output,
                "diagnostic_layers_topk_ids",
                None,
            )
            self._diagnostic_component_states = getattr(
                logits_output,
                "diagnostic_component_states",
                None,
            )
            transaction = logits_output.recurrent_state_transaction
            self._diagnostic_candidate_inputs = (
                None if transaction is None else transaction.candidate_inputs
            )
        result = verify_compact_chain_greedy(
            np.asarray(jax.device_get(self.verify_ids_strided), dtype=np.int32),
            target_predict,
            self.layout.verify_lens,
        )
        if int(np.count_nonzero(self.layout.verify_lens)) == 1:
            replicated_logits = jax.sharding.reshard(
                logits_output.next_token_logits,
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            )
            top_values, top_ids = jax.lax.top_k(
                replicated_logits.astype(jnp.float32),
                2,
            )
            top_ids_host = np.asarray(jax.device_get(top_ids), dtype=np.int32)
            top_values_host = np.asarray(jax.device_get(top_values), dtype=np.float32)
            logger.info(
                "DSPARK_VERIFY_RESULT proposals=%s target_predict=%s predict=%s "
                "verified=%s accepted=%s accept_index=%s",
                np.asarray(jax.device_get(self.verify_ids_strided), dtype=np.int32).tolist(),
                target_predict.tolist(),
                np.asarray(result[0], dtype=np.int32).tolist(),
                np.asarray(result[1], dtype=np.int32).tolist(),
                np.asarray(result[2], dtype=np.int32).tolist(),
                np.asarray(result[3], dtype=np.int32).tolist(),
            )
            logger.info(
                "DSPARK_VERIFY_TOP2 ids=%s values=%s margins=%s",
                top_ids_host.tolist(),
                top_values_host.tolist(),
                (top_values_host[:, 0] - top_values_host[:, 1]).tolist(),
            )
        return result

    def tree_flatten(self):
        children = (
            self.verify_ids_strided,
            self.compact_input_ids,
            self.compact_positions,
            self.layout,
            self.seq_lens_cpu,
            np.asarray(self.seq_lens_sum, dtype=np.int32),
            self.allocate_lens,
            self.hidden_states,
            self.custom_mask,
        )
        aux = (self.max_verify_tokens, self.capture_hidden_mode, self.topk)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        maximum, capture_hidden_mode, topk = aux
        return cls(
            verify_ids_strided=children[0],
            compact_input_ids=children[1],
            compact_positions=children[2],
            layout=children[3],
            seq_lens_cpu=children[4],
            seq_lens_sum=children[5],
            max_verify_tokens=maximum,
            capture_hidden_mode=capture_hidden_mode,
            topk=topk,
            allocate_lens=children[6],
            hidden_states=children[7],
            custom_mask=children[8],
        )
