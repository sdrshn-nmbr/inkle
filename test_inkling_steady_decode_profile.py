from types import SimpleNamespace
from unittest import mock

import pytest

from inkling_steady_decode_profile import post_profile, run_profile


def test_profile_requires_tokens_after_capture_window() -> None:
    args = SimpleNamespace(captured_steps=64, output_tokens=64)

    with pytest.raises(ValueError, match="smaller than output tokens"):
        run_profile(args)


def test_profile_endpoint_accepts_plain_text_responses() -> None:
    response = mock.Mock(
        content=b"Profiling started",
        headers={"content-type": "text/plain; charset=utf-8"},
        status_code=200,
        text="Profiling started",
    )

    with mock.patch("inkling_steady_decode_profile.requests.post", return_value=response):
        result = post_profile("http://server", "start_profile", {})

    assert result == {"status_code": 200, "body": "Profiling started"}
    response.raise_for_status.assert_called_once_with()
