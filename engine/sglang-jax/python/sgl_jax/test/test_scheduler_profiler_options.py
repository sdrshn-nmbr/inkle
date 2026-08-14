from unittest import mock

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
