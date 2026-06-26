from typing import Mapping, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .visibility_mask import VisibilityConfig, build_layerwise_visibility_mask

logger = get_logger(__name__)


class FastWAM3DMask(FastWAM):
    """FastWAM variant with configurable layer-wise visibility masks."""

    def __init__(self, *args, visibility_config: Optional[Mapping[str, object]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visibility_config = VisibilityConfig.from_mapping(visibility_config)

    @classmethod
    def from_wan22_pretrained(
        cls,
        visibility_config: Optional[Mapping[str, object]] = None,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.visibility_config = VisibilityConfig.from_mapping(visibility_config)
        logger.info("Initialized FastWAM3DMask with visibility_config=%s", model.visibility_config)
        return model

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        video_grid_size: Optional[tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        base_mask = super()._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            video_grid_size=video_grid_size,
        )
        return build_layerwise_visibility_mask(
            base_mask=base_mask,
            num_layers=int(self.mot.num_layers),
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            video_grid_size=video_grid_size,
            cfg=self.visibility_config,
            training=bool(self.mot.training),
        )
