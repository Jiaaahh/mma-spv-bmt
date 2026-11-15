"""Sobel edge-magnitude Top-K electrode baseline."""

from __future__ import annotations

import numpy as np
from skimage.filters import sobel

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FloatImage,
    SelectionResult,
)
from src.encoders.base import BaseEncoder, EncoderError, deterministic_topk


class EdgeTopKEncoder(BaseEncoder):
    """Rank shared candidate projections of normalized Sobel magnitude."""

    name = "edge_topk"

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        self.validate_inputs(image, grid, budget, projector)
        edge_map = np.asarray(sobel(image), dtype=np.float32)
        if edge_map.shape != image.shape or not np.isfinite(edge_map).all():
            raise EncoderError("Sobel filtering produced an invalid edge map")
        maximum = float(edge_map.max())
        if maximum <= 0.0:
            raise EncoderError("Sobel edge map is empty")
        edge_map = np.ascontiguousarray(
            edge_map / np.float32(maximum), dtype=np.float32
        )

        scores = projector.project(edge_map)
        if scores.shape != (grid.candidate_count,) or scores.dtype != np.float32:
            raise EncoderError("edge candidate scoring produced an invalid array")
        if float(scores.max()) <= 0.0:
            raise EncoderError("no candidate center received a positive edge score")
        selected = deterministic_topk(scores, budget)
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=selected,
            candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
            drive_map=edge_map,
            metadata={
                "drive_map": "normalized_sobel_magnitude",
                "candidate_projection": "l1_ideal_gaussian",
                "tie_break": "candidate_index_ascending",
            },
        )
