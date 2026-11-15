"""Equal-weight Stage-6 EB topology drive map with deterministic Top-K."""

from __future__ import annotations

import numpy as np

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FROZEN_STAGE6_EB_DRIVE,
    FROZEN_TOPOLOGY_DETECTION,
    FloatImage,
    SelectionResult,
)
from src.eb_representation import build_shared_eb_representation
from src.encoders.base import BaseEncoder, deterministic_topk


class EbTopKEncoder(BaseEncoder):
    """Rank shared ideal-Gaussian projections of the frozen EB drive map."""

    name = "eb_topk"

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        self.validate_inputs(image, grid, budget, projector)
        representation = build_shared_eb_representation(image, grid, projector)
        components = representation.components
        scores = representation.candidate_scores
        selected = deterministic_topk(scores, budget)
        topology_spec = FROZEN_TOPOLOGY_DETECTION
        eb_spec = FROZEN_STAGE6_EB_DRIVE
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=selected,
            candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
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
                "selection": "deterministic_topk",
                "tie_break": "candidate_index_ascending",
                "coverage": "not_implemented",
            },
        )
