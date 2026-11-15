"""Encoder plug-in factory for the frozen fair comparison contract."""

from __future__ import annotations

from src.encoders.base import BaseEncoder, EncoderError
from src.encoders.eb_sse import EbSseEncoder, EbSseOrderResult
from src.encoders.eb_topk import EbTopKEncoder
from src.encoders.edge_topk import EdgeTopKEncoder
from src.encoders.raw_topk import RawTopKEncoder
from src.encoders.skeleton_topk import SkeletonTopKEncoder


def build_encoder(name: str) -> BaseEncoder:
    """Construct one supported encoder by its frozen configuration name."""

    normalized = name.strip().lower()
    if normalized == RawTopKEncoder.name:
        return RawTopKEncoder()
    if normalized == EdgeTopKEncoder.name:
        return EdgeTopKEncoder()
    if normalized == SkeletonTopKEncoder.name:
        return SkeletonTopKEncoder()
    if normalized == EbTopKEncoder.name:
        return EbTopKEncoder()
    if normalized == EbSseEncoder.name:
        return EbSseEncoder()
    raise EncoderError(
        f"unsupported encoder {name!r}; available: "
        "'raw_topk', 'edge_topk', 'skeleton_topk', 'eb_topk', 'eb_sse'"
    )


__all__ = [
    "BaseEncoder",
    "EbSseEncoder",
    "EbSseOrderResult",
    "EbTopKEncoder",
    "EdgeTopKEncoder",
    "EncoderError",
    "RawTopKEncoder",
    "SkeletonTopKEncoder",
    "build_encoder",
]
