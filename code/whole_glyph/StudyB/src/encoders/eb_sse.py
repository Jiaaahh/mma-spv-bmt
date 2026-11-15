"""Stage-7 EB-SSE: frozen EB representation plus coverage-greedy selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FROZEN_STAGE6_EB_DRIVE,
    FROZEN_STAGE7_COVERAGE,
    FROZEN_TOPOLOGY_DETECTION,
    FloatImage,
    FloatVector,
    SelectionResult,
)
from src.coverage import (
    CoverageGreedyTrace,
    build_ideal_overlap_matrix,
    greedy_coverage_order,
)
from src.eb_drive import EbDriveComponents
from src.eb_representation import build_shared_eb_representation
from src.encoders.base import BaseEncoder, EncoderError
from src.protocol import validate_stage5_5_truth_table


@dataclass(frozen=True, slots=True)
class EbSseOrderResult:
    """One shared EB representation and its single 128-step greedy order."""

    components: EbDriveComponents = field(repr=False)
    candidate_scores: FloatVector = field(repr=False)
    trace: CoverageGreedyTrace = field(repr=False)

    def __post_init__(self) -> None:
        if self.candidate_scores.dtype != np.float32 or self.candidate_scores.ndim != 1:
            raise TypeError("candidate_scores must be a one-dimensional float32 array")
        if self.trace.order.shape != (
            FROZEN_STAGE7_COVERAGE.max_order_length,
        ):
            raise ValueError("EB-SSE must generate exactly one 128-step order")
        if self.trace.final_residual.shape != self.candidate_scores.shape:
            raise ValueError("coverage residual and candidate utility shapes differ")


class EbSseEncoder(BaseEncoder):
    """Use the Stage-5.5 redundancy-aware greedy selector on Stage-6 D_EB."""

    name = "eb_sse"

    def encode_full_order(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        *,
        projector: CandidateUtilityProjector,
    ) -> EbSseOrderResult:
        spec = FROZEN_STAGE7_COVERAGE
        self.validate_inputs(image, grid, spec.max_order_length, projector)
        validate_stage5_5_truth_table()
        representation = build_shared_eb_representation(image, grid, projector)
        components = representation.components
        scores = representation.candidate_scores
        overlap = build_ideal_overlap_matrix(grid)
        trace = greedy_coverage_order(
            scores,
            overlap,
            max_steps=spec.max_order_length,
        )
        return EbSseOrderResult(
            components=components,
            candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
            trace=trace,
        )

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        spec = FROZEN_STAGE7_COVERAGE
        if budget not in spec.budgets:
            raise EncoderError(
                f"EB-SSE budget must be one of the frozen values {spec.budgets}"
            )
        full = self.encode_full_order(image, grid, projector=projector)
        components = full.components
        topology_spec = FROZEN_TOPOLOGY_DETECTION
        eb_spec = FROZEN_STAGE6_EB_DRIVE
        protocol = validate_stage5_5_truth_table()
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=np.ascontiguousarray(
                full.trace.order[:budget], dtype=np.int64
            ),
            candidate_scores=full.candidate_scores,
            drive_map=components.drive_map,
            metadata={
                "binarization_threshold": topology_spec.binarization_threshold,
                "skeletonization": topology_spec.skeletonization,
                "topology_detection": topology_spec.method,
                "junction_connectivity": topology_spec.junction_component_connectivity,
                "junction_representative": topology_spec.junction_representative,
                "endpoint_count": int(components.topology.endpoints.sum()),
                "junction_candidate_count": int(
                    components.topology.junction_candidates.sum()
                ),
                "junction_cluster_count": components.topology.junction_cluster_count,
                "sigma_s_px": eb_spec.sigma_s_px,
                "sigma_t_px": eb_spec.sigma_t_px,
                "alpha_s": eb_spec.alpha_s,
                "alpha_e": eb_spec.alpha_e,
                "alpha_j": eb_spec.alpha_j,
                "weight_status": eb_spec.weight_status,
                "drive_map": "eb_topology_equal_weight_mvp",
                "representation_builder": "build_shared_eb_representation",
                "drive_map_normalization": eb_spec.drive_normalization,
                "candidate_projection": "l1_ideal_gaussian",
                "selection": "redundancy_aware_coverage_greedy",
                "tie_break": spec.tie_break,
                "coverage_geometry": spec.geometry,
                "coverage_sigma_r_px": spec.sigma_r_px,
                "coverage_cutoff_px": spec.cutoff_px,
                "coverage_objective": spec.objective,
                "coverage_gain": spec.greedy_gain,
                "greedy_order_length": spec.max_order_length,
                "budget_prefix": budget,
                "objective_at_budget": float(full.trace.objective_values[budget]),
                "theoretical_optimality_claim": spec.theoretical_optimality_claim,
                "protocol_id": protocol.protocol_id,
                "truth_table_sha256": protocol.sha256,
            },
        )
