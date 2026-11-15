"""Stage-8 registries and non-performance validation for the formal benchmark."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import ImageFont

from src.config import ExperimentConfig
from src.deterministic_io import sha256_file
from src.registry import CharacterRecord, FontRecord, inspect_font


BENCHMARK_ID = "eb_sse_hanzi180_font8_v1"
GOOGLE_FONTS_COMMIT = "352f6b7d9d6cc4fa9e242b931291d31b21a6dc84"
CHARACTER_SEQUENCE_SHA256 = (
    "570f754fc4182ef4010472c7ac13867db86a058ce031f3c6fbad7a393a294936"
)
STAGE5_5_TRUTH_TABLE_SHA256 = (
    "3559d22eba27bd195bef99a29169307c21f0898b2869b113ff1c09af9dace97e"
)
FORMAL_METHODS = (
    "raw_topk",
    "edge_topk",
    "skeleton_topk",
    "eb_topk",
    "eb_sse",
)
FORMAL_BUDGETS = (32, 64, 128)
FORMAL_MAPPING_SEEDS = (2026, 2027, 2028, 2029, 2030)
DEVELOPMENT_MAPPING_SEED = 2101

FORMAL_CHARACTER_FIELDS = (
    "char_id",
    "character",
    "unicode_codepoint",
    "source_rank",
    "chr_count",
    "chr_per_million",
    "log_chr",
    "chr_cd",
    "chr_cd_percent",
    "log_chr_cd",
    "selection_status",
    "registry_version",
)

FORMAL_FONT_FIELDS = (
    "font_id",
    "family",
    "style",
    "face_index",
    "split_role",
    "runtime_split",
    "file_name",
    "file_path",
    "file_size_bytes",
    "file_sha256",
    "license_id",
    "license_path",
    "license_sha256",
    "metadata_path",
    "metadata_sha256",
    "source_kind",
    "source_url",
    "source_commit",
    "prior_mvp_usage",
    "source_note",
)


class FormalBenchmarkError(ValueError):
    """Raised when Stage-8 assets do not match the frozen benchmark."""


@dataclass(frozen=True, slots=True)
class FormalFontSpec:
    font_id: str
    family: str
    style: str
    face_index: int
    split_role: str
    runtime_split: str
    file_path: str
    file_sha256: str
    license_path: str
    license_sha256: str
    metadata_path: str
    metadata_sha256: str
    source_kind: str
    source_url: str
    source_commit: str
    prior_mvp_usage: str
    source_note: str


def _google_raw(folder: str, filename: str) -> str:
    return (
        "http/historical/raw.githubusercontent.com/google/fonts/"
        f"{GOOGLE_FONTS_COMMIT}/ofl/{folder}/{filename}"
    )


FORMAL_FONT_SPECS: tuple[FormalFontSpec, ...] = (
    FormalFontSpec(
        "noto_sans_sc_thin",
        "Noto Sans SC Thin",
        "Regular",
        0,
        "development_train",
        "train",
        "data/raw/fonts/mvp_train_font_0f4c9c5bb1a8.ttf",
        "0f4c9c5bb1a887840c4de396279bdb5059bd20b5f58341a08542f18868aa1b1e",
        "data/raw/font_licenses/formal/noto_sans_sc_thin/OFL.txt",
        "1c05c68c34f9708415aada51f17e1b0092d2cea709bf4a94cd38114f9e73d7d9",
        "data/raw/font_licenses/formal/noto_sans_sc_thin/METADATA.pb",
        "c6c4bdb785793b2de1da177caeeffcd0c90d7680bdab2de8f88ab45007bc59bc",
        "stage1_frozen_asset",
        _google_raw("notosanssc", "METADATA.pb"),
        GOOGLE_FONTS_COMMIT,
        "mvp_train_stage1_to_stage7",
        "Exact Stage-1 binary preserved byte-for-byte; upstream family metadata and OFL are pinned separately.",
    ),
    FormalFontSpec(
        "noto_serif_sc_extralight",
        "Noto Serif SC ExtraLight",
        "Regular",
        0,
        "development_train",
        "train",
        "data/raw/fonts/mvp_test_font_de14e4adfc94.ttf",
        "de14e4adfc94592e7858170cad44171f4b44a7c46479ecbba013046f6fb038e1",
        "data/raw/font_licenses/formal/noto_serif_sc_extralight/OFL.txt",
        "5e0da210fb04058a8c0087985d2d456b931c2579811a49655721d3cf0c36b6d6",
        "data/raw/font_licenses/formal/noto_serif_sc_extralight/METADATA.pb",
        "e0c253c6f850fc3ef6e7fc398361133dd55acc408c931822a3f69cfac08554c2",
        "stage1_frozen_asset",
        _google_raw("notoserifsc", "METADATA.pb"),
        GOOGLE_FONTS_COMMIT,
        "mvp_test_stage1_to_stage7_now_development_only",
        "Exact Stage-1 binary preserved byte-for-byte; upstream family metadata and OFL are pinned separately.",
    ),
    FormalFontSpec(
        "ma_shan_zheng",
        "Ma Shan Zheng",
        "Regular",
        0,
        "development_train",
        "train",
        "data/raw/fonts/formal/ma_shan_zheng_6d2546bb189c.ttf",
        "6d2546bb189c732a8ca29af9e22457b152387d158aa459e4ac2ce1e51788b7fb",
        "data/raw/font_licenses/formal/ma_shan_zheng/OFL.txt",
        "d7bdb1cee215b689e23c2f95672a6084c790542170648267a55114103d756a08",
        "data/raw/font_licenses/formal/ma_shan_zheng/METADATA.pb",
        "7e21912b6659b58694f0b999e69ad5a95da12fdc453565059727a2467e525f84",
        "pinned_google_fonts_binary",
        _google_raw("mashanzheng", "MaShanZheng-Regular.ttf"),
        GOOGLE_FONTS_COMMIT,
        "none_before_stage8",
        "Downloaded once before any formal performance run and frozen by SHA-256.",
    ),
    FormalFontSpec(
        "zcool_qingke_huangyou",
        "ZCOOL QingKe HuangYou",
        "Regular",
        0,
        "development_train",
        "train",
        "data/raw/fonts/formal/zcool_qingke_huangyou_54f0c0df4308.ttf",
        "54f0c0df4308cd74cd0f2fd3494ae054dbc4a1fd6fa7d71f4807eb4cdd8b4136",
        "data/raw/font_licenses/formal/zcool_qingke_huangyou/OFL.txt",
        "f1bce31b817dee01c1e4ef8bc45d8ecb95f01f4abbf0a985007cb3cd0fd8123d",
        "data/raw/font_licenses/formal/zcool_qingke_huangyou/METADATA.pb",
        "254f65c558dd29fdabcec269174fb5d80fbd36350a8df729a85686163840f74f",
        "pinned_google_fonts_binary",
        _google_raw("zcoolqingkehuangyou", "ZCOOLQingKeHuangYou-Regular.ttf"),
        GOOGLE_FONTS_COMMIT,
        "none_before_stage8",
        "Downloaded once before any formal performance run and frozen by SHA-256.",
    ),
    FormalFontSpec(
        "zcool_kuaile",
        "ZCOOL KuaiLe",
        "Regular",
        0,
        "development_train",
        "train",
        "data/raw/fonts/formal/zcool_kuaile_812a6fc1fe54.ttf",
        "812a6fc1fe54b6d73a419245c32dfeba8aa33104d5be90d1cf6af082007cb71d",
        "data/raw/font_licenses/formal/zcool_kuaile/OFL.txt",
        "538078469839b4a2e7ad22bef4ebe41681a4e53749bb2a072144024f1d6d703d",
        "data/raw/font_licenses/formal/zcool_kuaile/METADATA.pb",
        "fc1f48f54521ac41b8aa59d5b44c6b3c7cb7d24b9d4d499564c9febd01f84769",
        "pinned_google_fonts_binary",
        _google_raw("zcoolkuaile", "ZCOOLKuaiLe-Regular.ttf"),
        GOOGLE_FONTS_COMMIT,
        "none_before_stage8",
        "Downloaded once before any formal performance run and frozen by SHA-256.",
    ),
    FormalFontSpec(
        "liu_jian_mao_cao",
        "Liu Jian Mao Cao",
        "Regular",
        0,
        "development_validation",
        "validation",
        "data/raw/fonts/formal/liu_jian_mao_cao_cab396b91a5b.ttf",
        "cab396b91a5b7c0b4005a35891180d06e6751f5ac261fe680aec65c1ae209033",
        "data/raw/font_licenses/formal/liu_jian_mao_cao/OFL.txt",
        "ff56684b0212481e7c3886c26d5a655de9b211c8119e5ccc756138a97b066acc",
        "data/raw/font_licenses/formal/liu_jian_mao_cao/METADATA.pb",
        "22725c680fa61f8facbedbeb668292454214c740050e16f1c89a9c66ddcb2538",
        "pinned_google_fonts_binary",
        _google_raw("liujianmaocao", "LiuJianMaoCao-Regular.ttf"),
        GOOGLE_FONTS_COMMIT,
        "none_before_stage8",
        "Validation identity fixed before alpha selection; never a locked-test font.",
    ),
    FormalFontSpec(
        "long_cang",
        "Long Cang",
        "Regular",
        0,
        "locked_test",
        "test",
        "data/raw/fonts/formal/long_cang_e5bf2c3f24ef.ttf",
        "e5bf2c3f24ef2327c6f136d8f73e2f9dfdf44896fdbeb35a9515f44777bb91bc",
        "data/raw/font_licenses/formal/long_cang/OFL.txt",
        "603546b7219a94bb59bf8294458194a5010119486354092b66a09a3fd61aeacc",
        "data/raw/font_licenses/formal/long_cang/METADATA.pb",
        "c7d6c01a886b37dcef3c1e89796424f34240647fedac0d37108309c02fa8f3a3",
        "pinned_google_fonts_binary",
        _google_raw("longcang", "LongCang-Regular.ttf"),
        GOOGLE_FONTS_COMMIT,
        "none_before_stage8_locked_unseen",
        "Locked before any encoder, alpha search, classifier, or accuracy access.",
    ),
    FormalFontSpec(
        "zhi_mang_xing",
        "Zhi Mang Xing",
        "Regular",
        0,
        "locked_test",
        "test",
        "data/raw/fonts/formal/zhi_mang_xing_644e0cae9b40.ttf",
        "644e0cae9b40f0b10ab729a01bd32032e3973bac22be3dccae01bf6ae7fde969",
        "data/raw/font_licenses/formal/zhi_mang_xing/OFL.txt",
        "10947328199e369a3e6b4a67e8e5507ed99d5bbb264a1f156415aa9b665e4d15",
        "data/raw/font_licenses/formal/zhi_mang_xing/METADATA.pb",
        "a3319526a687b010f28e79bc5fef7072005be36615224df60df37847b2ef0c78",
        "pinned_google_fonts_binary",
        _google_raw("zhimangxing", "ZhiMangXing-Regular.ttf"),
        GOOGLE_FONTS_COMMIT,
        "none_before_stage8_locked_unseen",
        "Locked before any encoder, alpha search, classifier, or accuracy access.",
    ),
)


def sequence_sha256(characters: Sequence[str]) -> str:
    return hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()


def load_formal_characters(path: str | Path) -> tuple[CharacterRecord, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != FORMAL_CHARACTER_FIELDS:
            raise FormalBenchmarkError(
                f"formal character columns must be exactly {FORMAL_CHARACTER_FIELDS}"
            )
        rows = list(reader)
    if len(rows) != 180:
        raise FormalBenchmarkError(f"formal character count is {len(rows)}, expected 180")
    records: list[CharacterRecord] = []
    for expected_id, row in enumerate(rows):
        try:
            char_id = int(row["char_id"])
            source_rank = int(row["source_rank"])
        except ValueError as exc:
            raise FormalBenchmarkError("invalid char_id or source_rank") from exc
        character = row["character"]
        if char_id != expected_id or source_rank != expected_id + 1:
            raise FormalBenchmarkError("formal character order or source rank changed")
        if len(character) != 1 or character.isspace():
            raise FormalBenchmarkError("each formal class must be one non-space character")
        if row["unicode_codepoint"] != f"U+{ord(character):04X}":
            raise FormalBenchmarkError("formal Unicode codepoint does not match character")
        if row["selection_status"] != "FROZEN_CONFIRMED":
            raise FormalBenchmarkError("formal character registry is not confirmed/frozen")
        if row["registry_version"] != "stage8_formal_v1":
            raise FormalBenchmarkError("unexpected formal character registry version")
        records.append(CharacterRecord(char_id, character))
    characters = [record.character for record in records]
    if len(set(characters)) != 180:
        raise FormalBenchmarkError("formal character registry contains duplicates")
    if sequence_sha256(characters) != CHARACTER_SEQUENCE_SHA256:
        raise FormalBenchmarkError("formal character sequence SHA-256 changed")
    return tuple(records)


def _font_mask_fingerprint(font: ImageFont.FreeTypeFont, character: str) -> str:
    mask = font.getmask(character, mode="L")
    digest = hashlib.sha256()
    digest.update(str(mask.size).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(font.getbbox(character)).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes(mask))
    return digest.hexdigest()


def check_font_coverage(
    font_path: str | Path,
    face_index: int,
    characters: Sequence[CharacterRecord],
    *,
    font_size: int = 144,
) -> dict[str, Any]:
    """Check direct FreeType coverage without OS fallback and detect tofu/duplicates."""

    font = ImageFont.truetype(str(font_path), size=font_size, index=face_index)
    missing_fingerprint = _font_mask_fingerprint(font, chr(0x10FFFF))
    fingerprints = {
        record.character: _font_mask_fingerprint(font, record.character)
        for record in characters
    }
    missing = sorted(
        character
        for character, fingerprint in fingerprints.items()
        if fingerprint == missing_fingerprint
    )
    unique_count = len(set(fingerprints.values()))
    return {
        "supported_count": len(characters) - len(missing),
        "missing_characters": missing,
        "unique_raw_glyph_fingerprints": unique_count,
        "tofu_sentinel_codepoint": "U+10FFFF",
    }


def load_formal_fonts(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[FontRecord, ...]:
    source = Path(manifest_path)
    root = Path(project_root).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != FORMAL_FONT_FIELDS:
            raise FormalBenchmarkError(
                f"formal font columns must be exactly {FORMAL_FONT_FIELDS}"
            )
        rows = list(reader)
    if len(rows) != len(FORMAL_FONT_SPECS):
        raise FormalBenchmarkError("formal font manifest must contain exactly eight rows")

    records: list[FontRecord] = []
    for row, expected in zip(rows, FORMAL_FONT_SPECS, strict=True):
        values = {
            "font_id": expected.font_id,
            "family": expected.family,
            "style": expected.style,
            "face_index": str(expected.face_index),
            "split_role": expected.split_role,
            "runtime_split": expected.runtime_split,
            "file_path": expected.file_path,
            "file_sha256": expected.file_sha256,
            "license_id": "OFL-1.1",
            "license_path": expected.license_path,
            "license_sha256": expected.license_sha256,
            "metadata_path": expected.metadata_path,
            "metadata_sha256": expected.metadata_sha256,
            "source_kind": expected.source_kind,
            "source_url": expected.source_url,
            "source_commit": expected.source_commit,
            "prior_mvp_usage": expected.prior_mvp_usage,
            "source_note": expected.source_note,
        }
        for key, expected_value in values.items():
            if row[key] != expected_value:
                raise FormalBenchmarkError(
                    f"font {expected.font_id} field {key} changed: {row[key]!r}"
                )
        file_path = (root / row["file_path"]).resolve()
        license_path = (root / row["license_path"]).resolve()
        metadata_path = (root / row["metadata_path"]).resolve()
        for path, expected_hash, label in (
            (file_path, expected.file_sha256, "font"),
            (license_path, expected.license_sha256, "license"),
            (metadata_path, expected.metadata_sha256, "metadata"),
        ):
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise FormalBenchmarkError(f"{label} path escapes project root") from exc
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise FormalBenchmarkError(
                    f"{label} file missing or hash mismatch for {expected.font_id}"
                )
        if row["file_name"] != file_path.name:
            raise FormalBenchmarkError(f"font filename mismatch for {expected.font_id}")
        if int(row["file_size_bytes"]) != file_path.stat().st_size:
            raise FormalBenchmarkError(f"font size mismatch for {expected.font_id}")
        family, style = inspect_font(file_path, expected.face_index)
        if (family, style) != (expected.family, expected.style):
            raise FormalBenchmarkError(
                f"Pillow metadata mismatch for {expected.font_id}: {(family, style)}"
            )
        records.append(
            FontRecord(
                font_id=expected.font_id,
                split=expected.runtime_split,
                family_name=expected.family,
                style_name=expected.style,
                file_path=file_path,
                file_sha256=expected.file_sha256,
                face_index=expected.face_index,
            )
        )
    if len({record.font_id for record in records}) != 8:
        raise FormalBenchmarkError("formal font IDs are not unique")
    if len({record.family_name.casefold() for record in records}) != 8:
        raise FormalBenchmarkError("formal fonts must be eight distinct families")
    if [record.split for record in records].count("train") != 5:
        raise FormalBenchmarkError("formal split must contain five training fonts")
    if [record.split for record in records].count("validation") != 1:
        raise FormalBenchmarkError("formal split must contain one validation font")
    if [record.split for record in records].count("test") != 2:
        raise FormalBenchmarkError("formal split must contain two locked-test fonts")
    return tuple(records)


def validate_formal_config(config: ExperimentConfig) -> None:
    expected_fonts = {
        "dataset.split.train_fonts": tuple(
            spec.font_id for spec in FORMAL_FONT_SPECS if spec.runtime_split == "train"
        ),
        "dataset.split.validation_fonts": tuple(
            spec.font_id
            for spec in FORMAL_FONT_SPECS
            if spec.runtime_split == "validation"
        ),
        "dataset.split.test_fonts": tuple(
            spec.font_id for spec in FORMAL_FONT_SPECS if spec.runtime_split == "test"
        ),
    }
    checks = {
        "stage": config.stage == "benchmark",
        "character_count": int(config.get("dataset.character_count")) == 180,
        "image_size": config.image_size == (96, 96),
        "grid": config.grid_shape == (16, 16),
        "methods": config.encoders == FORMAL_METHODS,
        "budgets": config.budgets == FORMAL_BUDGETS,
        "mapping_names": config.mapping_names == ("ideal", "moderate"),
        "mapping_seeds": config.mapping_seeds
        == (DEVELOPMENT_MAPPING_SEED, *FORMAL_MAPPING_SEEDS),
        "candidate_utility": config.get("protocol.candidate_utility")
        == "l1_ideal_gaussian_projection",
        "unit_amplitude": config.get("protocol.stimulation_amplitude")
        == "unit_selected_electrode",
        "topology": config.get("protocol.topology_detection")
        == "crossing_number",
        "status": config.get("protocol.result_status")
        == "stage8_benchmark_frozen_no_accuracy",
    }
    for key, expected in expected_fonts.items():
        checks[key] = tuple(config.get(key)) == expected
    changed = [name for name, value in checks.items() if not value]
    if changed:
        raise FormalBenchmarkError(f"formal benchmark config changed: {changed}")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FormalBenchmarkError(f"JSON root must be an object: {path}")
    return value
