# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx==0.28.1",
#   "huggingface-hub==1.3.3",
#   "ml-dtypes==0.5.4",
#   "numpy==2.4.3",
# ]
# ///

import json
import logging
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import ml_dtypes
import numpy as np
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

NVFP4_REPOSITORY = "thinkingmachines/Inkling-NVFP4"
NVFP4_REVISION = "deeb2d05eaa977db4ff7727db33670a2e05938cf"
BF16_REPOSITORY = "thinkingmachines/Inkling"
BF16_REVISION = "32b58a6494356948441ceedc192f1194f06f2e23"

SAFETENSORS_DTYPES: dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "I16": np.dtype(np.int16),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
}
LOGGER = logging.getLogger(__name__)
MAX_RANGE_READ_ATTEMPTS = 12


class ByteRangeSource(Protocol):
    @property
    def size(self) -> int: ...

    def read(self, start: int, stop: int) -> bytes: ...


class LocalByteRangeSource:
    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def size(self) -> int:
        return self.path.stat().st_size

    def read(self, start: int, stop: int) -> bytes:
        with self.path.open("rb") as checkpoint_file:
            checkpoint_file.seek(start)
            data = checkpoint_file.read(stop - start)
        if len(data) != stop - start:
            raise OSError(f"INKLING_RANGE_READ_FAILED path={self.path} start={start} stop={stop} received={len(data)}")
        return data


class HttpByteRangeSource:
    def __init__(
        self,
        url: str,
        size: int,
        client: httpx.Client,
        refresh_url: Callable[[], str] | None = None,
    ) -> None:
        self.url = url
        self._size = size
        self.client = client
        self.refresh_url = refresh_url

    @property
    def size(self) -> int:
        return self._size

    def read(self, start: int, stop: int) -> bytes:
        for attempt in range(1, MAX_RANGE_READ_ATTEMPTS + 1):
            response: httpx.Response | None = None
            try:
                response = self.client.get(self.url, headers={"Range": f"bytes={start}-{stop - 1}"})
                if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN) and self.refresh_url:
                    self.url = self.refresh_url()
                    LOGGER.warning(
                        "INKLING_RANGE_URL_REFRESH attempt=%d start=%d stop=%d",
                        attempt,
                        start,
                        stop,
                    )
                    response = self.client.get(self.url, headers={"Range": f"bytes={start}-{stop - 1}"})
                if response.status_code != httpx.codes.PARTIAL_CONTENT:
                    raise OSError(
                        "INKLING_RANGE_READ_FAILED "
                        f"status={response.status_code} start={start} stop={stop} url={self.url.split('?')[0]}"
                    )
                if len(response.content) != stop - start:
                    raise OSError(
                        "INKLING_RANGE_READ_FAILED "
                        f"start={start} stop={stop} received={len(response.content)} url={self.url.split('?')[0]}"
                    )
                return response.content
            except (httpx.TransportError, OSError):
                if attempt == MAX_RANGE_READ_ATTEMPTS:
                    raise
                if (
                    response is not None
                    and response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR
                    and self.refresh_url
                    and attempt % 3 == 0
                ):
                    self.url = self.refresh_url()
                    LOGGER.warning(
                        "INKLING_RANGE_URL_REFRESH attempt=%d start=%d stop=%d",
                        attempt,
                        start,
                        stop,
                    )
                LOGGER.warning(
                    "INKLING_RANGE_READ_RETRY attempt=%d start=%d stop=%d url=%s",
                    attempt,
                    start,
                    stop,
                    self.url.split("?")[0],
                )
                time.sleep(min(2 ** (attempt - 1), 30))
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def nbytes(self) -> int:
        return self.stop - self.start


class SafetensorsRangeReader:
    def __init__(self, source: ByteRangeSource) -> None:
        self.source = source
        header_length_bytes = source.read(0, 8)
        self.header_length = struct.unpack("<Q", header_length_bytes)[0]
        header_stop = 8 + self.header_length
        if header_stop > source.size:
            raise ValueError(f"INKLING_INVALID_SAFETENSORS_HEADER header_stop={header_stop} file_size={source.size}")
        header = json.loads(source.read(8, header_stop))
        self.data_start = header_stop
        self.tensors: dict[str, TensorMetadata] = {}
        for name, raw_metadata in header.items():
            if name == "__metadata__":
                continue
            dtype_name = raw_metadata["dtype"]
            if dtype_name not in SAFETENSORS_DTYPES:
                raise ValueError(f"INKLING_UNSUPPORTED_DTYPE tensor={name} dtype={dtype_name}")
            offsets = raw_metadata["data_offsets"]
            metadata = TensorMetadata(
                name=name,
                dtype=SAFETENSORS_DTYPES[dtype_name],
                shape=tuple(raw_metadata["shape"]),
                start=self.data_start + offsets[0],
                stop=self.data_start + offsets[1],
            )
            expected_bytes = int(np.prod(metadata.shape, dtype=np.int64)) * metadata.dtype.itemsize
            if metadata.nbytes != expected_bytes:
                raise ValueError(
                    f"INKLING_INVALID_TENSOR_SIZE tensor={name} expected={expected_bytes} actual={metadata.nbytes}"
                )
            self.tensors[name] = metadata

    def metadata(self, name: str) -> TensorMetadata:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"INKLING_TENSOR_NOT_FOUND tensor={name}") from error

    def read_tensor(self, name: str) -> np.ndarray:
        metadata = self.metadata(name)
        data = self.source.read(metadata.start, metadata.stop)
        return np.frombuffer(data, dtype=metadata.dtype).reshape(metadata.shape).copy()

    def read_first_axis(self, name: str, index: int) -> np.ndarray:
        return self.read_first_axis_slice(name, index, index + 1)[0]

    def read_first_axis_slice(self, name: str, start_index: int, stop_index: int) -> np.ndarray:
        metadata = self.metadata(name)
        if not metadata.shape:
            raise ValueError(f"INKLING_SCALAR_CANNOT_BE_SLICED tensor={name}")
        if not 0 <= start_index < stop_index <= metadata.shape[0]:
            raise IndexError(
                "INKLING_SLICE_OUT_OF_RANGE "
                f"tensor={name} start={start_index} stop={stop_index} size={metadata.shape[0]}"
            )
        elements_per_slice = int(np.prod(metadata.shape[1:], dtype=np.int64))
        bytes_per_slice = elements_per_slice * metadata.dtype.itemsize
        start = metadata.start + start_index * bytes_per_slice
        data = self.source.read(start, start + (stop_index - start_index) * bytes_per_slice)
        return np.frombuffer(data, dtype=metadata.dtype).reshape(stop_index - start_index, *metadata.shape[1:]).copy()


class HuggingFaceSafetensorsRepository:
    def __init__(
        self,
        repository: str,
        revision: str,
        *,
        token: str | bool | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.repository = repository
        self.revision = revision
        self.token = token
        self.client = httpx.Client(follow_redirects=True, timeout=timeout_seconds)
        index_path = hf_hub_download(
            repository,
            "model.safetensors.index.json",
            revision=revision,
            token=token,
        )
        self.weight_map: dict[str, str] = json.loads(Path(index_path).read_text())["weight_map"]
        self._readers: dict[str, SafetensorsRangeReader] = {}

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "HuggingFaceSafetensorsRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def reader_for(self, tensor_name: str) -> SafetensorsRangeReader:
        try:
            filename = self.weight_map[tensor_name]
        except KeyError as error:
            raise KeyError(f"INKLING_TENSOR_NOT_IN_INDEX tensor={tensor_name}") from error
        if filename not in self._readers:
            url = hf_hub_url(self.repository, filename, revision=self.revision)
            file_metadata = get_hf_file_metadata(url, token=self.token)
            if file_metadata.location is None or file_metadata.size is None:
                raise RuntimeError(f"INKLING_FILE_METADATA_INCOMPLETE file={filename}")

            def refresh_url() -> str:
                refreshed_metadata = get_hf_file_metadata(url, token=self.token)
                if refreshed_metadata.location is None:
                    raise RuntimeError(f"INKLING_FILE_METADATA_INCOMPLETE file={filename}")
                return refreshed_metadata.location

            source = HttpByteRangeSource(file_metadata.location, file_metadata.size, self.client, refresh_url)
            self._readers[filename] = SafetensorsRangeReader(source)
        return self._readers[filename]

    def metadata(self, tensor_name: str) -> TensorMetadata:
        return self.reader_for(tensor_name).metadata(tensor_name)

    def read_tensor(self, tensor_name: str) -> np.ndarray:
        return self.reader_for(tensor_name).read_tensor(tensor_name)

    def read_first_axis(self, tensor_name: str, index: int) -> np.ndarray:
        return self.reader_for(tensor_name).read_first_axis(tensor_name, index)

    def read_first_axis_slice(self, tensor_name: str, start_index: int, stop_index: int) -> np.ndarray:
        return self.reader_for(tensor_name).read_first_axis_slice(tensor_name, start_index, stop_index)
