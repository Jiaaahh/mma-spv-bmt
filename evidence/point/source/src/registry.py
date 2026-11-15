"""Character and font registries used by the reproducible MVP dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence

from PIL import ImageFont, UnidentifiedImageError


MVP_CHARACTERS: tuple[str, ...] = tuple("人大小中国天地山水日月木火土口田目上下王")
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}
FONT_FIELDS = (
    "font_id",
    "split",
    "family_name",
    "style_name",
    "file_path",
    "file_sha256",
    "face_index",
)


class RegistryError(ValueError):
    """Raised when a character or font registry is invalid."""


@dataclass(frozen=True, slots=True)
class CharacterRecord:
    class_id: int
    character: str


@dataclass(frozen=True, slots=True)
class FontSpec:
    font_id: str
    split: str
    source_path: Path
    face_index: int = 0


@dataclass(frozen=True, slots=True)
class FontRecord:
    font_id: str
    split: str
    family_name: str
    style_name: str
    file_path: Path
    file_sha256: str
    face_index: int


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes, *, replace: bool) -> str:
    """Write atomically, returning created, unchanged, or replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "unchanged"
        if not replace:
            raise RegistryError(
                f"refusing to replace existing registry: {path}; rerun with --replace-registry"
            )
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return "replaced" if replace else "created"


def write_mvp_character_registry(
    path: str | Path,
    *,
    expected_count: int = 20,
    replace: bool = False,
) -> str:
    if len(MVP_CHARACTERS) != expected_count:
        raise RegistryError(
            f"built-in MVP list has {len(MVP_CHARACTERS)} characters, expected {expected_count}"
        )
    if len(set(MVP_CHARACTERS)) != len(MVP_CHARACTERS):
        raise RegistryError("built-in MVP character list contains duplicates")
    lines = ["# Hanzi-SPV MVP character registry v1", "# class_id\tcharacter"]
    lines.extend(f"{index}\t{character}" for index, character in enumerate(MVP_CHARACTERS))
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return _write_bytes(Path(path), payload, replace=replace)


def load_character_registry(
    path: str | Path, *, expected_count: int | None = None
) -> tuple[CharacterRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise RegistryError(f"character registry does not exist: {source}")
    records: list[CharacterRecord] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise RegistryError(f"{source}:{line_number} must be class_id<TAB>character")
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise RegistryError(f"{source}:{line_number} has an invalid class_id") from exc
        character = parts[1]
        if len(character) != 1 or character.isspace():
            raise RegistryError(
                f"{source}:{line_number} must contain exactly one non-space character"
            )
        records.append(CharacterRecord(class_id, character))

    if not records:
        raise RegistryError(f"character registry is empty: {source}")
    expected_ids = list(range(len(records)))
    actual_ids = [record.class_id for record in records]
    if actual_ids != expected_ids:
        raise RegistryError("character class IDs must be contiguous and start at 0")
    characters = [record.character for record in records]
    if len(set(characters)) != len(characters):
        raise RegistryError("character registry contains duplicate characters")
    if expected_count is not None and len(records) != expected_count:
        raise RegistryError(
            f"character registry has {len(records)} records, expected {expected_count}"
        )
    return tuple(records)


def inspect_font(path: str | Path, face_index: int = 0) -> tuple[str, str]:
    source = Path(path)
    if not source.is_file():
        raise RegistryError(f"font file does not exist: {source}")
    if source.suffix.lower() not in FONT_EXTENSIONS:
        raise RegistryError(f"unsupported font extension: {source.suffix}")
    if isinstance(face_index, bool) or not isinstance(face_index, int) or face_index < 0:
        raise RegistryError("font face_index must be a non-negative integer")
    try:
        font = ImageFont.truetype(str(source), size=32, index=face_index)
        family_name, style_name = font.getname()
    except (OSError, UnidentifiedImageError) as exc:
        raise RegistryError(
            f"Pillow cannot open font {source} at face index {face_index}: {exc}"
        ) from exc
    family_name = str(family_name).strip()
    style_name = str(style_name).strip()
    if not family_name:
        raise RegistryError(f"font has no readable family name: {source}")
    return family_name, style_name or "Regular"


def _validate_font_id(font_id: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]*", font_id) is None:
        raise RegistryError(
            f"font_id must use lowercase ASCII letters, digits, and underscores: {font_id!r}"
        )


def _copy_frozen_font(source: Path, font_dir: Path, font_id: str) -> tuple[Path, str]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise RegistryError(f"font file does not exist: {source}")
    if source.suffix.lower() not in FONT_EXTENSIONS:
        raise RegistryError(f"unsupported font extension: {source.suffix}")
    digest = sha256_file(source)
    font_dir.mkdir(parents=True, exist_ok=True)
    target = font_dir / f"{font_id}_{digest[:12]}{source.suffix.lower()}"
    if target.exists():
        if sha256_file(target) != digest:
            raise RegistryError(f"hash collision or damaged frozen font: {target}")
    elif source != target:
        shutil.copy2(source, target)
    return target.resolve(), digest


def write_font_registry(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    font_dir: str | Path,
    specs: Sequence[FontSpec],
    replace: bool = False,
) -> tuple[tuple[FontRecord, ...], str]:
    root = Path(project_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = (root / manifest).resolve()
    frozen_dir = Path(font_dir)
    if not frozen_dir.is_absolute():
        frozen_dir = (root / frozen_dir).resolve()

    if not specs:
        raise RegistryError("at least one font specification is required")
    records: list[FontRecord] = []
    for spec in specs:
        _validate_font_id(spec.font_id)
        if spec.split not in {"train", "validation", "test"}:
            raise RegistryError(f"invalid split for {spec.font_id}: {spec.split}")
        family, style = inspect_font(spec.source_path, spec.face_index)
        frozen_path, digest = _copy_frozen_font(
            Path(spec.source_path), frozen_dir, spec.font_id
        )
        try:
            frozen_path.relative_to(root)
        except ValueError as exc:
            raise RegistryError("frozen font directory must stay inside project_root") from exc
        records.append(
            FontRecord(
                font_id=spec.font_id,
                split=spec.split,
                family_name=family,
                style_name=style,
                file_path=frozen_path,
                file_sha256=digest,
                face_index=spec.face_index,
            )
        )

    font_ids = [record.font_id for record in records]
    if len(set(font_ids)) != len(font_ids):
        raise RegistryError("font specifications contain duplicate font IDs")
    identities = [(record.file_sha256, record.face_index) for record in records]
    if len(set(identities)) != len(identities):
        raise RegistryError("train and test must not use the same font file and face")
    families = [record.family_name.casefold() for record in records]
    if len(set(families)) != len(families):
        raise RegistryError("MVP fonts must come from distinct font families")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FONT_FIELDS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "font_id": record.font_id,
                "split": record.split,
                "family_name": record.family_name,
                "style_name": record.style_name,
                "file_path": record.file_path.relative_to(root).as_posix(),
                "file_sha256": record.file_sha256,
                "face_index": record.face_index,
            }
        )
    status = _write_bytes(
        manifest, buffer.getvalue().encode("utf-8-sig"), replace=replace
    )
    return tuple(records), status


def load_font_registry(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
    expected_splits: Mapping[str, str] | None = None,
    verify_hashes: bool = True,
) -> tuple[FontRecord, ...]:
    manifest = Path(manifest_path)
    root = Path(project_root).resolve()
    if not manifest.is_file():
        raise RegistryError(f"font manifest does not exist: {manifest}")

    records: list[FontRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != FONT_FIELDS:
            raise RegistryError(f"font manifest columns must be exactly {FONT_FIELDS}")
        for row_number, row in enumerate(reader, 2):
            font_id = row["font_id"].strip()
            _validate_font_id(font_id)
            split = row["split"].strip()
            if split not in {"train", "validation", "test"}:
                raise RegistryError(f"invalid split at manifest row {row_number}")
            try:
                face_index = int(row["face_index"])
            except ValueError as exc:
                raise RegistryError(f"invalid face_index at manifest row {row_number}") from exc
            relative = Path(row["file_path"])
            if relative.is_absolute():
                raise RegistryError("font manifest file_path values must be relative")
            file_path = (root / relative).resolve()
            try:
                file_path.relative_to(root)
            except ValueError as exc:
                raise RegistryError("font path escapes project_root") from exc
            if not file_path.is_file():
                raise RegistryError(f"registered font file does not exist: {file_path}")
            expected_hash = row["file_sha256"].strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
                raise RegistryError(f"invalid SHA-256 at manifest row {row_number}")
            if verify_hashes and sha256_file(file_path) != expected_hash:
                raise RegistryError(f"font hash mismatch: {file_path}")
            family, style = inspect_font(file_path, face_index)
            if family != row["family_name"].strip() or style != row["style_name"].strip():
                raise RegistryError(f"font metadata mismatch: {file_path}")
            records.append(
                FontRecord(
                    font_id=font_id,
                    split=split,
                    family_name=family,
                    style_name=style,
                    file_path=file_path,
                    file_sha256=expected_hash,
                    face_index=face_index,
                )
            )

    if not records:
        raise RegistryError("font manifest is empty")
    ids = [record.font_id for record in records]
    if len(set(ids)) != len(ids):
        raise RegistryError("font manifest contains duplicate font IDs")
    identities = {(record.file_sha256, record.face_index) for record in records}
    if len(identities) != len(records):
        raise RegistryError("font manifest reuses the same font file and face")
    families = {record.family_name.casefold() for record in records}
    if len(families) != len(records):
        raise RegistryError("font manifest must use distinct font families")
    if expected_splits is not None:
        actual = {record.font_id: record.split for record in records}
        if actual != dict(expected_splits):
            raise RegistryError(
                f"font manifest does not match config splits: actual={actual}, expected={dict(expected_splits)}"
            )
    return tuple(records)


def discover_windows_font_pair() -> tuple[tuple[Path, int], tuple[Path, int]]:
    """Find two distinct common Chinese font families on Windows."""

    if os.name != "nt":
        raise RegistryError(
            "automatic font discovery is Windows-only; provide --train-font and --test-font"
        )
    windows_dir = Path(os.environ.get("WINDIR", r"/historical/Windows"))
    font_dir = windows_dir / "Fonts"
    candidates = (
        "msyh.ttc",      # Microsoft YaHei
        "simsun.ttc",    # SimSun
        "simhei.ttf",    # SimHei
        "Deng.ttf",      # DengXian
        "simkai.ttf",    # KaiTi
        "simfang.ttf",   # FangSong
    )
    found: list[tuple[Path, int, str]] = []
    for filename in candidates:
        path = font_dir / filename
        if not path.is_file():
            continue
        try:
            family, _ = inspect_font(path, 0)
        except RegistryError:
            continue
        if family.casefold() not in {item[2].casefold() for item in found}:
            found.append((path, 0, family))
        if len(found) == 2:
            return (found[0][0], found[0][1]), (found[1][0], found[1][1])
    raise RegistryError(
        "could not discover two Chinese font families in /historical/Windows\\Fonts; "
        "provide explicit --train-font and --test-font paths"
    )
