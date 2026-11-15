"""Raw-intensity Top-K electrode baseline."""

from __future__ import annotations

import numpy as np

from src.contracts import (
    CandidateUtilityProjector,
    ElectrodeGrid,
    FloatImage,
    SelectionResult,
)
from src.encoders.base import BaseEncoder, EncoderError, deterministic_topk


class RawTopKEncoder(BaseEncoder):
    """Rank candidates by shared ideal-Gaussian projection of the raw glyph."""

    name = "raw_topk"

    def encode(
        self,
        image: FloatImage,
        grid: ElectrodeGrid,
        budget: int,
        *,
        projector: CandidateUtilityProjector,
    ) -> SelectionResult:
        self.validate_inputs(image, grid, budget, projector)
        drive_map = np.ascontiguousarray(image, dtype=np.float32)
        scores = projector.project(drive_map)
        if scores.shape != (grid.candidate_count,) or scores.dtype != np.float32:
            raise EncoderError("raw candidate scoring produced an invalid array")

        selected = deterministic_topk(scores, budget)
        return SelectionResult(
            encoder_name=self.name,
            budget=budget,
            candidate_count=grid.candidate_count,
            selected_indices=selected,
            candidate_scores=np.ascontiguousarray(scores, dtype=np.float32),
            drive_map=drive_map,
            metadata={
                "drive_map": "raw_intensity",
                "candidate_projection": "l1_ideal_gaussian",
                "tie_break": "candidate_index_ascending",
            },
        )
