from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class DSparkScheduleConfig:
    gamma: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    @property
    def resolved_max_verify_len(self) -> int:
        return self.max_verify_len or self.gamma + 1

    def validate(self) -> None:
        maximum = self.resolved_max_verify_len
        if self.gamma < 1:
            raise ValueError(f"DSPARK_GAMMA_INVALID gamma={self.gamma}")
        if not 0 <= self.min_verify_len <= maximum <= self.gamma + 1:
            raise ValueError(
                "DSPARK_VERIFY_LENGTH_INVALID "
                f"min={self.min_verify_len} max={maximum} gamma={self.gamma}"
            )
        if self.survival_eps < 0:
            raise ValueError(
                f"DSPARK_SURVIVAL_EPS_INVALID survival_eps={self.survival_eps}"
            )


@dataclass(frozen=True, slots=True)
class SpsCostTable:
    sample_batch_tokens: tuple[int, ...]
    sample_steps_per_sec: tuple[float, ...]
    max_batch_tokens: int

    def validate(self) -> None:
        if not self.sample_batch_tokens:
            raise ValueError("DSPARK_SPS_TABLE_EMPTY")
        if len(self.sample_batch_tokens) != len(self.sample_steps_per_sec):
            raise ValueError("DSPARK_SPS_TABLE_LENGTH_MISMATCH")
        if any(right <= left for left, right in zip(
            self.sample_batch_tokens,
            self.sample_batch_tokens[1:],
            strict=False,
        )):
            raise ValueError("DSPARK_SPS_TABLE_TOKENS_NOT_INCREASING")
        if any(value <= 0 for value in self.sample_steps_per_sec):
            raise ValueError("DSPARK_SPS_TABLE_RATE_NONPOSITIVE")
        if self.max_batch_tokens < self.sample_batch_tokens[-1]:
            raise ValueError(
                "DSPARK_SPS_TABLE_MAX_UNDERSIZED "
                f"max={self.max_batch_tokens} largest_sample={self.sample_batch_tokens[-1]}"
            )


def load_sps_cost_table(path: str | Path) -> SpsCostTable:
    payload = json.loads(Path(path).read_text())
    table = SpsCostTable(
        sample_batch_tokens=tuple(int(value) for value in payload["sample_batch_tokens"]),
        sample_steps_per_sec=tuple(
            float(value) for value in payload["sample_steps_per_sec"]
        ),
        max_batch_tokens=int(payload["max_batch_tokens"]),
    )
    table.validate()
    return table


def load_sts_temperatures(path: str | Path, *, gamma: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    temperatures = np.asarray(payload["temperatures"], dtype=np.float32)
    if temperatures.shape != (gamma,) or np.any(temperatures <= 0):
        raise ValueError(
            "DSPARK_STS_TEMPERATURE_INVALID "
            f"shape={temperatures.shape} gamma={gamma}"
        )
    return temperatures


@dataclass(frozen=True, slots=True)
class VerifyBudgetDecision:
    budget: int
    predicted_step_seconds: float
    predicted_output_tokens_per_second: float


def confidence_to_survival(confidence: np.ndarray) -> np.ndarray:
    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"DSPARK_CONFIDENCE_RANK expected=2 got={values.ndim}")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("DSPARK_CONFIDENCE_OUT_OF_RANGE")
    return np.cumprod(values, axis=1)


def _lookup_steps_per_second(table: SpsCostTable, batch_tokens: np.ndarray) -> np.ndarray:
    probes = np.asarray(table.sample_batch_tokens, dtype=np.int64)
    rates = np.asarray(table.sample_steps_per_sec, dtype=np.float64)
    indices = np.searchsorted(probes, batch_tokens, side="right") - 1
    return rates[np.clip(indices, 0, probes.size - 1)]


def compute_verify_token_budget(
    history_survival_probabilities: np.ndarray,
    sps_table: SpsCostTable,
    config: DSparkScheduleConfig,
) -> VerifyBudgetDecision:
    config.validate()
    sps_table.validate()
    survival = np.asarray(history_survival_probabilities, dtype=np.float64)
    if survival.ndim != 2:
        raise ValueError(f"DSPARK_SURVIVAL_RANK expected=2 got={survival.ndim}")
    request_count = survival.shape[0]
    maximum_tokens = request_count * config.resolved_max_verify_len
    if maximum_tokens > sps_table.max_batch_tokens:
        raise ValueError(
            "DSPARK_SPS_TABLE_COVERAGE_INSUFFICIENT "
            f"required={maximum_tokens} max={sps_table.max_batch_tokens}"
        )
    candidates = survival[:, : config.resolved_max_verify_len].reshape(-1)
    candidates = np.sort(candidates[candidates >= config.survival_eps])[::-1]
    expected_outputs = request_count + np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(candidates))
    )
    batch_tokens = request_count + np.arange(expected_outputs.size, dtype=np.int64)
    step_rates = _lookup_steps_per_second(sps_table, batch_tokens)
    output_rates = expected_outputs * step_rates
    budget = int(np.argmax(output_rates))
    return VerifyBudgetDecision(
        budget=budget,
        predicted_step_seconds=1.0 / float(step_rates[budget]),
        predicted_output_tokens_per_second=float(output_rates[budget]),
    )


def schedule_verify_lengths(
    survival_probabilities: np.ndarray,
    budget: int,
    config: DSparkScheduleConfig,
) -> np.ndarray:
    config.validate()
    survival = np.asarray(survival_probabilities, dtype=np.float64)
    if survival.ndim != 2:
        raise ValueError(f"DSPARK_SURVIVAL_RANK expected=2 got={survival.ndim}")
    request_count = survival.shape[0]
    floor = max(config.min_verify_len, 1)
    maximum = config.resolved_max_verify_len
    capacity = request_count * (maximum - floor)
    if not 0 <= budget <= capacity:
        raise ValueError(f"DSPARK_VERIFY_BUDGET_INVALID budget={budget} capacity={capacity}")
    lengths = np.full(request_count, floor, dtype=np.int32)
    if budget == 0:
        return lengths
    candidate_rows = []
    for request in range(request_count):
        for verify_len in range(floor + 1, maximum + 1):
            draft_index = verify_len - 2
            probability = survival[request, draft_index]
            candidate_rows.append((-probability, request, verify_len))
    candidate_rows.sort()
    selected = {(request, verify_len) for _, request, verify_len in candidate_rows[:budget]}
    for request in range(request_count):
        while (request, int(lengths[request]) + 1) in selected:
            lengths[request] += 1
    if int(np.sum(lengths - floor)) > budget:
        raise AssertionError("DSPARK_VERIFY_BUDGET_EXCEEDED")
    return lengths
