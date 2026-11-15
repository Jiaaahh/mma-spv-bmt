"""Deterministic binary writers and canonical hashes for frozen benchmark assets."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np


class DeterministicIoError(ValueError):
    """Raised when a supposedly reproducible artifact cannot be serialized."""


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-contiguous array bytes without pickle."""

    if not isinstance(array, np.ndarray):
        raise TypeError("array_sha256 expects a numpy.ndarray")
    if array.dtype.hasobject:
        raise DeterministicIoError("object arrays are forbidden in frozen assets")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return a byte-stable compressed NPZ with sorted keys and fixed ZIP metadata."""

    if not arrays:
        raise DeterministicIoError("at least one array is required")
    keys = tuple(sorted(arrays))
    if any(not key or "/" in key or "\\" in key for key in keys):
        raise DeterministicIoError("NPZ keys must be non-empty simple names")

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for key in keys:
            array = arrays[key]
            if not isinstance(array, np.ndarray):
                raise TypeError(f"NPZ value {key!r} must be a numpy.ndarray")
            if array.dtype.hasobject:
                raise DeterministicIoError(
                    f"object array {key!r} is forbidden in frozen assets"
                )
            member = io.BytesIO()
            np.save(member, np.ascontiguousarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def write_deterministic_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Atomically write a deterministic NPZ and return its SHA-256."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = deterministic_npz_bytes(arrays)
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return sha256_bytes(payload)


def tree_sha256(root: str | Path, paths: list[Path]) -> str:
    """Hash ordered relative paths plus their contents for a small frozen tree."""

    base = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda p: p.as_posix()):
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as exc:
            raise DeterministicIoError(f"path escapes tree root: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
