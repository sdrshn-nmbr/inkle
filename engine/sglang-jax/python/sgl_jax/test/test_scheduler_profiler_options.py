from unittest import mock

import pytest

from sgl_jax.srt.managers.scheduler_profiler_mixing import _build_profiler_options


class FakeProfileOptions:
    def __init__(self):
        self.advanced_configuration = {}
        self.enable_hlo_proto = False
        self.host_tracer_level = 2
        self.python_tracer_level = 1
        self.raise_error_on_start_failure = False


def test_tpu_profile_options_use_strict_defaults(monkeypatch):
    monkeypatch.delenv("SGLANG_TPU_TRACE_MODE", raising=False)
    monkeypatch.delenv("SGLANG_TPU_PROFILE_CHIP_COUNT", raising=False)
    monkeypatch.delenv("SGLANG_TPU_PERF_COUNTERS", raising=False)
    monkeypatch.delenv("SGLANG_TPU_PERIODIC_COUNTERS", raising=False)
    monkeypatch.delenv("SGLANG_TPU_PROFILE_ENABLE_HLO_PROTO", raising=False)
    with mock.patch(
        "sgl_jax.srt.managers.scheduler_profiler_mixing.jax.profiler.ProfileOptions",
        FakeProfileOptions,
    ):
        options = _build_profiler_options(None, None, platform="tpu")

    assert options.raise_error_on_start_failure is True
    assert options.enable_hlo_proto is False
    assert options.advanced_configuration == {}


def test_tpu_profile_options_add_only_requested_configuration(monkeypatch):
    monkeypatch.delenv("SGLANG_TPU_PERIODIC_COUNTERS", raising=False)
    monkeypatch.setenv("SGLANG_TPU_TRACE_MODE", "TRACE_COMPUTE_AND_SYNC")
    monkeypatch.setenv("SGLANG_TPU_PROFILE_CHIP_COUNT", "4")
    monkeypatch.setenv("SGLANG_TPU_PERF_COUNTERS", "1")
    monkeypatch.setenv("SGLANG_TPU_PROFILE_ENABLE_HLO_PROTO", "1")
    with mock.patch(
        "sgl_jax.srt.managers.scheduler_profiler_mixing.jax.profiler.ProfileOptions",
        FakeProfileOptions,
    ):
        options = _build_profiler_options(0, 0, platform="tpu")

    assert options.host_tracer_level == 0
    assert options.python_tracer_level == 0
    assert options.raise_error_on_start_failure is True
    assert options.enable_hlo_proto is True
    assert options.advanced_configuration == {
        "tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC",
        "tpu_num_chips_to_profile_per_task": 4,
        "tpu_perf_counters": True,
    }


def test_cpu_profile_options_do_not_add_tpu_configuration():
    with mock.patch(
        "sgl_jax.srt.managers.scheduler_profiler_mixing.jax.profiler.ProfileOptions",
        FakeProfileOptions,
    ):
        options = _build_profiler_options(1, 0, platform="cpu")

    assert options.host_tracer_level == 1
    assert options.python_tracer_level == 0
    assert options.raise_error_on_start_failure is True
    assert options.enable_hlo_proto is False
    assert options.advanced_configuration == {}


def test_tpu_profile_options_add_periodic_counters(monkeypatch):
    monkeypatch.delenv("SGLANG_TPU_PERF_COUNTERS", raising=False)
    monkeypatch.setenv("SGLANG_TPU_PERIODIC_COUNTERS", "1")
    with mock.patch(
        "sgl_jax.srt.managers.scheduler_profiler_mixing.jax.profiler.ProfileOptions",
        FakeProfileOptions,
    ):
        options = _build_profiler_options(None, None, platform="tpu")

    assert options.advanced_configuration == {
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 "
            "indices:1 indices:3 indices:4 indices:10 indices:11 "
            "indices:31 indices:32 indices:33 indices:34 indices:35 "
            "indices:37 indices:38 indices:56 indices:57 indices:58 "
            "indices:73 indices:74 indices:75 indices:105"
        ),
        "num_tensor_cores_to_trace_per_device": 1,
    }


def test_tpu_profile_options_reject_mixed_counter_modes(monkeypatch):
    monkeypatch.setenv("SGLANG_TPU_PERF_COUNTERS", "1")
    monkeypatch.setenv("SGLANG_TPU_PERIODIC_COUNTERS", "1")
    with mock.patch(
        "sgl_jax.srt.managers.scheduler_profiler_mixing.jax.profiler.ProfileOptions",
        FakeProfileOptions,
    ), mock.patch(
        "sgl_jax.srt.managers.scheduler_profiler_mixing.jax.default_backend",
        return_value="tpu",
    ), pytest.raises(ValueError, match="not both"):
        _build_profiler_options(None, None)
