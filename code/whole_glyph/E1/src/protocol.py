"""Read-only validation of the embedded Stage-5.5 machine truth table."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from src.contracts import (
    FROZEN_STAGE6_EB_DRIVE,
    FROZEN_STAGE7_COVERAGE,
    FROZEN_TOPOLOGY_DETECTION,
)


class ProtocolTruthError(ValueError):
    """Raised when the embedded machine truth table is absent or changed."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE5_5_TRUTH_TABLE_PATH = (
    PROJECT_ROOT / "protocols" / "STAGE5_5_CODE_TRUTH_TABLE.yaml"
)
STAGE5_5_TRUTH_TABLE_SHA256 = (
    "3559d22eba27bd195bef99a29169307c21f0898b2869b113ff1c09af9dace97e"
)


@dataclass(frozen=True, slots=True)
class ProtocolValidationEvidence:
    protocol_id: str
    sha256: str
    semantic_check_count: int


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _at(values: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = values
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ProtocolTruthError(f"truth table key is missing: {dotted_key}")
        current = current[part]
    return current


@lru_cache(maxsize=1)
def load_stage5_5_truth_table() -> Mapping[str, Any]:
    """Load the exact embedded YAML after byte and semantic verification."""

    try:
        payload = STAGE5_5_TRUTH_TABLE_PATH.read_bytes()
    except OSError as exc:
        raise ProtocolTruthError(
            f"cannot read Stage-5.5 truth table: {STAGE5_5_TRUTH_TABLE_PATH}"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != STAGE5_5_TRUTH_TABLE_SHA256:
        raise ProtocolTruthError(
            "Stage-5.5 truth table bytes changed; create a new explicit protocol "
            "version instead of silently changing Stage 7"
        )
    try:
        parsed = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ProtocolTruthError(f"invalid Stage-5.5 YAML: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ProtocolTruthError("Stage-5.5 YAML root must be a mapping")
    values = _freeze(parsed)
    _validate_semantics(values)
    return values


def _validate_semantics(values: Mapping[str, Any]) -> int:
    topology = FROZEN_TOPOLOGY_DETECTION
    eb = FROZEN_STAGE6_EB_DRIVE
    coverage = FROZEN_STAGE7_COVERAGE
    expected: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "eb_sse_stage5_5_freeze_v1",
        "image.shape_hw": (96, 96),
        "image.dtype": "float32",
        "grid.rows": 16,
        "grid.cols": 16,
        "grid.candidate_count": 256,
        "grid.index_order": "row_major",
        "grid.spacing_px": coverage.d_e_px,
        "grid.d_e_px": coverage.d_e_px,
        "grid.budgets": coverage.budgets,
        "grid.exact_budget_required": True,
        "topology.threshold": topology.binarization_threshold,
        "topology.skeletonization": topology.skeletonization,
        "topology.skeleton_pruning": "none",
        "topology.crossing_number.border_policy": topology.border_policy,
        "topology.junction_clustering.connectivity": (
            topology.junction_component_connectivity
        ),
        "normalization.common_positive_percentile_normalization": "none",
        "eb_drive_map.gaussian_diffusion.canvas": "full_96x96",
        "eb_drive_map.gaussian_diffusion.explicit_radius_truncation": "none",
        "eb_drive_map.gaussian_diffusion.sigma_s_px": eb.sigma_s_px,
        "eb_drive_map.gaussian_diffusion.sigma_t_px": eb.sigma_t_px,
        "eb_drive_map.alpha_s": eb.alpha_s,
        "candidate_projector.mapping": "ideal_only",
        "candidate_projector.mapping_seed": 0,
        "candidate_projector.center_geometry": "ideal_grid",
        "candidate_projector.sigma_px": 2.7,
        "candidate_projector.canvas": "full_96x96",
        "candidate_projector.explicit_radius_truncation": "none",
        "candidate_projector.normalization": "l1_after_canvas_boundary",
        "candidate_projector.moderate_mapping_allowed": False,
        "selection.eb_topk.drive_map": "D_EB",
        "selection.eb_topk.utility": "shared_candidate_utility",
        "selection.eb_sse.drive_map": "D_EB",
        "selection.eb_sse.utility": "identical_to_eb_topk",
        "selection.eb_sse.overlap.sigma_r_px": coverage.sigma_r_px,
        "selection.eb_sse.overlap.cutoff_px": coverage.cutoff_px,
        "selection.eb_sse.overlap.diagonal": coverage.diagonal,
        "selection.eb_sse.overlap.geometry": coverage.geometry,
        "selection.eb_sse.objective": coverage.objective,
        "selection.eb_sse.greedy_gain": coverage.greedy_gain,
        "selection.eb_sse.tie_break": coverage.tie_break,
        "selection.eb_sse.nesting": "one_greedy_order_prefixes_32_64_128",
        "selection.eb_sse.theoretical_optimality_claim": (
            coverage.theoretical_optimality_claim
        ),
        "stimulation.selected_amplitude": 1.0,
        "stimulation.amplitude_is_encoder_output": False,
        "stimulation.active_count_equals_budget": True,
        "spv_forward.output_normalization": "none",
        "spv_forward.output_clipping": "none",
        "spv_forward.moderate.affects_selection": False,
        "mappings.engineering_mvp.seed": 2101,
        "mappings.engineering_mvp.result_status": "engineering_validation_only",
    }
    failures = [
        key for key, required in expected.items() if _at(values, key) != required
    ]
    if failures:
        raise ProtocolTruthError(
            f"Stage-5.5 semantic truth differs at keys: {failures}"
        )
    return len(expected)


def validate_stage5_5_truth_table() -> ProtocolValidationEvidence:
    values = load_stage5_5_truth_table()
    return ProtocolValidationEvidence(
        protocol_id=str(_at(values, "protocol_id")),
        sha256=STAGE5_5_TRUTH_TABLE_SHA256,
        semantic_check_count=_validate_semantics(values),
    )
