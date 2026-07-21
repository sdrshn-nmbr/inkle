import json
import struct
from pathlib import Path

import httpx
import ml_dtypes
import numpy as np
from checkpoint_io import HttpByteRangeSource, LocalByteRangeSource, SafetensorsRangeReader
from inkling_layout import decode_nvfp4_numpy, deinterleave_gate_up_numpy, unpack_nvfp4_numpy


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], dtype_names: dict[str, str]) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        tensor_bytes = tensor.tobytes(order="C")
        header[name] = {
            "dtype": dtype_names[name],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(tensor_bytes)],
        }
        payload.extend(tensor_bytes)
        offset += len(tensor_bytes)
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    padding = b" " * ((8 - len(encoded_header) % 8) % 8)
    encoded_header += padding
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload)


def test_safetensors_reader_reads_full_tensor_and_first_axis(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "probe.safetensors"
    bf16 = np.arange(24, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(3, 2, 4)
    packed = np.arange(12, dtype=np.uint8).reshape(3, 4)
    write_safetensors(checkpoint_path, {"bf16": bf16, "packed": packed}, {"bf16": "BF16", "packed": "U8"})

    reader = SafetensorsRangeReader(LocalByteRangeSource(checkpoint_path))

    np.testing.assert_array_equal(reader.read_tensor("bf16"), bf16)
    np.testing.assert_array_equal(reader.read_first_axis("bf16", 1), bf16[1])
    np.testing.assert_array_equal(reader.read_first_axis_slice("bf16", 1, 3), bf16[1:3])
    np.testing.assert_array_equal(reader.read_first_axis("packed", 2), packed[2])


def test_gate_up_rows_are_deinterleaved() -> None:
    raw = np.array([[10], [20], [11], [21], [12], [22]], dtype=np.float32)
    expected = np.array([[10], [11], [12], [20], [21], [22]], dtype=np.float32)

    np.testing.assert_array_equal(deinterleave_gate_up_numpy(raw), expected)


def test_nvfp4_uses_low_nibble_before_high_nibble() -> None:
    packed = np.array([[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F]], dtype=np.uint8)
    expected_codes = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]], dtype=np.uint8)

    np.testing.assert_array_equal(unpack_nvfp4_numpy(packed), expected_codes)


def test_nvfp4_applies_block_and_global_scales() -> None:
    packed = np.full((2, 8), 0x22, dtype=np.uint8)
    block_scale = np.array([[2.0], [3.0]], dtype=np.float32)
    global_scale = np.array(0.5, dtype=np.float32)

    decoded = decode_nvfp4_numpy(packed, block_scale, global_scale)

    np.testing.assert_array_equal(decoded[0], np.ones(16, dtype=np.float32))
    np.testing.assert_array_equal(decoded[1], np.full(16, 1.5, dtype=np.float32))


def test_http_range_source_refreshes_expired_url_without_consuming_retry() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/expired":
            return httpx.Response(httpx.codes.FORBIDDEN)
        return httpx.Response(httpx.codes.PARTIAL_CONTENT, content=b"2345")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpByteRangeSource(
            "https://checkpoint.test/expired",
            size=10,
            client=client,
            refresh_url=lambda: "https://checkpoint.test/fresh",
        )

        assert source.read(2, 6) == b"2345"

    assert [request.url.path for request in requests] == ["/expired", "/fresh"]
    assert requests[1].headers["Range"] == "bytes=2-5"


def test_http_range_source_survives_repeated_server_errors(monkeypatch: object) -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 4:
            return httpx.Response(httpx.codes.SERVICE_UNAVAILABLE)
        return httpx.Response(httpx.codes.PARTIAL_CONTENT, content=b"2345")

    monkeypatch.setattr("checkpoint_io.time.sleep", lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpByteRangeSource(
            "https://checkpoint.test/current",
            size=10,
            client=client,
            refresh_url=lambda: "https://checkpoint.test/refreshed",
        )

        assert source.read(2, 6) == b"2345"

    assert attempts == 5
