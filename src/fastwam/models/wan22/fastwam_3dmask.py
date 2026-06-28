from typing import Any, Mapping, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .fastwam_joint import FastWAMJoint
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
        if self.visibility_config.mode == "fastwam":
            return base_mask
        if self.visibility_config.mode == "joint":
            joint_mask = base_mask.clone()
            joint_mask[video_seq_len:, :video_seq_len] = True
            return joint_mask
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

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if self.visibility_config.mode == "joint":
            if test_action_with_infer_action:
                logger.warning(
                    "`FastWAM3DMask(mode='joint').infer_joint` matches FastWAMJoint and "
                    "always runs with `test_action_with_infer_action=False`."
                )
            return super().infer_joint(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                action=action,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                test_action_with_infer_action=False,
            )
        return super().infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=test_action_with_infer_action,
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        if self.visibility_config.mode == "joint":
            if num_video_frames is None:
                raise ValueError(
                    "`FastWAM3DMask(mode='joint').infer_action` requires `num_video_frames`, "
                    "matching FastWAMJoint."
                )
            return FastWAMJoint.infer_action(
                self,
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )
        del num_video_frames
        return super().infer_action(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
