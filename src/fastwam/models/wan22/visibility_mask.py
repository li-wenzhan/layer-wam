from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch


PHASES = ("p0", "p1", "p2", "p3")
SUPPORTED_MODES = {
    "fastwam",
    "joint",
    "early_matched",
    "sandwich_dense",
    "late_matched",
    "clcf",
    "clcf_wo_causal",
    "clcf_wo_temporal_local",
    "clcf_wo_spatial_c2f",
}
SCHEDULED_MODES = SUPPORTED_MODES - {"fastwam", "joint"}
SUPPORTED_TEMPORAL_RULES = {
    "none",
    "all",
    "coarse_causal",
    "local_causal",
    "local_noncausal",
    "causal_prefix",
}
SUPPORTED_SPATIAL_RULES = {"dense", "sparse"}


@dataclass(frozen=True)
class VisibilityRule:
    temporal: str = "none"
    spatial: str = "dense"


@dataclass(frozen=True)
class VisibilityConfig:
    mode: str = "clcf"
    phase_ratios: tuple[float, float, float] = (0.2, 0.4, 0.6)
    coarse_lookahead: int = 1
    coarse_spatial_stride: int = 4
    local_window_start: int = 2
    local_window_end: int = 0
    future_mask_dropout: float = 0.0

    def __post_init__(self) -> None:
        mode = str(self.mode)
        object.__setattr__(self, "mode", mode)
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported visibility mode: {mode}. Supported modes: {sorted(SUPPORTED_MODES)}")

        ratios = tuple(float(v) for v in self.phase_ratios)
        if len(ratios) != 3:
            raise ValueError(f"`phase_ratios` must have length 3, got {ratios}")
        r0, r1, r2 = ratios
        if not (0.0 < r0 < r1 < r2 < 1.0):
            raise ValueError(f"`phase_ratios` must satisfy 0 < r0 < r1 < r2 < 1, got {ratios}")
        object.__setattr__(self, "phase_ratios", ratios)

        if int(self.coarse_lookahead) < 0:
            raise ValueError(f"`coarse_lookahead` must be non-negative, got {self.coarse_lookahead}")
        if int(self.coarse_spatial_stride) <= 0:
            raise ValueError(f"`coarse_spatial_stride` must be positive, got {self.coarse_spatial_stride}")
        if int(self.local_window_start) < 0 or int(self.local_window_end) < 0:
            raise ValueError(
                "`local_window_start` and `local_window_end` must be non-negative, "
                f"got {self.local_window_start} and {self.local_window_end}"
            )
        dropout = float(self.future_mask_dropout)
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"`future_mask_dropout` must be in [0, 1], got {dropout}")
        object.__setattr__(self, "future_mask_dropout", dropout)

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, object]]) -> "VisibilityConfig":
        if mapping is None:
            return cls()
        return cls(**dict(mapping))


def get_phase(layer_idx: int, num_layers: int, phase_ratios: tuple[float, float, float] = (0.2, 0.4, 0.6)) -> str:
    if num_layers <= 0:
        raise ValueError(f"`num_layers` must be positive, got {num_layers}")
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"`layer_idx` must be in [0, {num_layers}), got {layer_idx}")
    r = (float(layer_idx) + 0.5) / float(num_layers)
    if r < phase_ratios[0]:
        return "p0"
    if r < phase_ratios[1]:
        return "p1"
    if r < phase_ratios[2]:
        return "p2"
    return "p3"


def compute_local_window(idx_in_phase: int, phase_len: int, start: int = 2, end: int = 0) -> int:
    if phase_len <= 0:
        raise ValueError(f"`phase_len` must be positive, got {phase_len}")
    if idx_in_phase < 0 or idx_in_phase >= phase_len:
        raise ValueError(f"`idx_in_phase` must be in [0, {phase_len}), got {idx_in_phase}")
    if phase_len <= 1:
        return int(end)
    alpha = idx_in_phase / float(phase_len - 1)
    return int(round(float(start) * (1.0 - alpha) + float(end) * alpha))


def compute_action_latent_alignment(
    action_seq_len: int,
    num_video_latent_frames: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if action_seq_len <= 0:
        raise ValueError(f"`action_seq_len` must be positive, got {action_seq_len}")
    if num_video_latent_frames <= 1:
        raise ValueError(
            "`compute_action_latent_alignment` requires at least one future latent frame, "
            f"got num_video_latent_frames={num_video_latent_frames}"
        )
    num_future_latent_frames = num_video_latent_frames - 1
    if action_seq_len % num_future_latent_frames != 0:
        raise ValueError(
            "`action_seq_len` must be divisible by actual future latent frames for latent-level alignment: "
            f"action_seq_len={action_seq_len}, future_latent_frames={num_future_latent_frames}"
        )
    action_latent_freq_ratio = action_seq_len // num_future_latent_frames
    action_idx = torch.arange(action_seq_len, device=device)
    aligned_frame = 1 + torch.div(action_idx, action_latent_freq_ratio, rounding_mode="floor")
    return aligned_frame.to(dtype=torch.long), int(action_latent_freq_ratio)


def build_spatial_visible(
    video_grid_size: tuple[int, int, int],
    spatial_rule: str,
    *,
    coarse_spatial_stride: int,
    device: torch.device,
) -> torch.Tensor:
    if spatial_rule not in SUPPORTED_SPATIAL_RULES:
        raise ValueError(f"Unsupported spatial rule: {spatial_rule}. Supported rules: {sorted(SUPPORTED_SPATIAL_RULES)}")
    _, h, w = _validate_video_grid_size(video_grid_size)
    if spatial_rule == "dense":
        return torch.ones((h * w,), dtype=torch.bool, device=device)

    stride = int(coarse_spatial_stride)
    if stride <= 0:
        raise ValueError(f"`coarse_spatial_stride` must be positive, got {coarse_spatial_stride}")
    y = torch.arange(h, device=device).view(h, 1).expand(h, w)
    x = torch.arange(w, device=device).view(1, w).expand(h, w)
    return ((y % stride == 0) & (x % stride == 0)).reshape(h * w)


def build_temporal_visible(
    action_seq_len: int,
    num_video_latent_frames: int,
    temporal_rule: str,
    *,
    coarse_lookahead: int,
    local_window: int,
    device: torch.device,
) -> torch.Tensor:
    if temporal_rule not in SUPPORTED_TEMPORAL_RULES:
        raise ValueError(
            f"Unsupported temporal rule: {temporal_rule}. Supported rules: {sorted(SUPPORTED_TEMPORAL_RULES)}"
        )
    visible = torch.zeros((action_seq_len, num_video_latent_frames), dtype=torch.bool, device=device)
    if temporal_rule == "none" or num_video_latent_frames <= 1:
        return visible

    aligned_frame, _ = compute_action_latent_alignment(
        action_seq_len=action_seq_len,
        num_video_latent_frames=num_video_latent_frames,
        device=device,
    )
    tau = torch.arange(num_video_latent_frames, device=device).view(1, num_video_latent_frames)
    future = tau >= 1
    aligned = aligned_frame.view(action_seq_len, 1)

    if temporal_rule == "all":
        return future.expand(action_seq_len, -1).clone()
    if temporal_rule == "coarse_causal":
        return future & (tau <= aligned + int(coarse_lookahead))
    if temporal_rule == "local_causal":
        return future & (tau >= aligned - int(local_window)) & (tau <= aligned)
    if temporal_rule == "local_noncausal":
        return future & ((tau - aligned).abs() <= int(local_window))
    if temporal_rule == "causal_prefix":
        return future & (tau <= aligned)
    raise AssertionError(f"Unhandled temporal rule: {temporal_rule}")


def resolve_rule(mode: str, phase: str) -> VisibilityRule:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported visibility mode: {mode}. Supported modes: {sorted(SUPPORTED_MODES)}")
    if phase not in PHASES:
        raise ValueError(f"Unsupported phase: {phase}. Supported phases: {PHASES}")

    if mode == "fastwam":
        return VisibilityRule("none", "dense")
    if mode == "joint":
        return VisibilityRule("all", "dense")
    if mode == "early_matched":
        return VisibilityRule("all", "dense") if phase in {"p0", "p1"} else VisibilityRule("none", "dense")
    if mode == "sandwich_dense":
        return VisibilityRule("all", "dense") if phase in {"p1", "p2"} else VisibilityRule("none", "dense")
    if mode == "late_matched":
        return VisibilityRule("all", "dense") if phase == "p3" else VisibilityRule("none", "dense")
    if mode == "clcf":
        if phase == "p1":
            return VisibilityRule("coarse_causal", "sparse")
        if phase == "p2":
            return VisibilityRule("local_causal", "dense")
        return VisibilityRule("none", "dense")
    if mode == "clcf_wo_causal":
        if phase == "p1":
            return VisibilityRule("all", "sparse")
        if phase == "p2":
            return VisibilityRule("local_noncausal", "dense")
        return VisibilityRule("none", "dense")
    if mode == "clcf_wo_temporal_local":
        if phase == "p1":
            return VisibilityRule("coarse_causal", "sparse")
        if phase == "p2":
            return VisibilityRule("causal_prefix", "dense")
        return VisibilityRule("none", "dense")
    if mode == "clcf_wo_spatial_c2f":
        if phase == "p1":
            return VisibilityRule("coarse_causal", "dense")
        if phase == "p2":
            return VisibilityRule("local_causal", "dense")
        return VisibilityRule("none", "dense")
    raise AssertionError(f"Unhandled visibility mode: {mode}")


@torch.no_grad()
def build_layerwise_visibility_mask(
    base_mask: torch.Tensor,
    *,
    num_layers: int,
    video_seq_len: int,
    action_seq_len: int,
    video_tokens_per_frame: int,
    video_grid_size: Optional[tuple[int, int, int]],
    cfg: Optional[VisibilityConfig] = None,
    training: bool = False,
) -> torch.Tensor:
    """Build a layer-wise direct action-to-future-video visibility mask.

    The only changed region is action query -> future video key. Video self-attention,
    action self-attention, action -> first-frame video, and video -> action are copied
    from `base_mask`.
    """
    cfg = VisibilityConfig() if cfg is None else cfg
    _validate_base_mask(
        base_mask=base_mask,
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
    )
    if num_layers <= 0:
        raise ValueError(f"`num_layers` must be positive, got {num_layers}")
    if video_tokens_per_frame <= 0:
        raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")
    if video_seq_len % video_tokens_per_frame != 0:
        raise ValueError(
            "`video_seq_len` must be divisible by `video_tokens_per_frame`: "
            f"video_seq_len={video_seq_len}, video_tokens_per_frame={video_tokens_per_frame}"
        )

    num_video_latent_frames = video_seq_len // video_tokens_per_frame
    if num_video_latent_frames <= 1:
        return base_mask

    if video_grid_size is None:
        raise ValueError("`video_grid_size` is required when future latent frames exist.")
    f, h, w = _validate_video_grid_size(video_grid_size)
    if f != num_video_latent_frames:
        raise ValueError(
            "`video_grid_size[0]` must match actual video latent frames derived from tokens: "
            f"grid_f={f}, derived_f={num_video_latent_frames}"
        )
    if h * w != video_tokens_per_frame:
        raise ValueError(
            "`video_grid_size` spatial size must match `video_tokens_per_frame`: "
            f"h*w={h * w}, tokens_per_frame={video_tokens_per_frame}"
        )

    compute_action_latent_alignment(
        action_seq_len=action_seq_len,
        num_video_latent_frames=num_video_latent_frames,
        device=base_mask.device,
    )

    dropout_enabled = training and cfg.mode in SCHEDULED_MODES and cfg.future_mask_dropout > 0.0
    if dropout_enabled:
        drop_sample = torch.rand((), device=base_mask.device)
        if bool(drop_sample < cfg.future_mask_dropout):
            return base_mask.unsqueeze(0).repeat(num_layers, 1, 1)

    layer_masks = base_mask.unsqueeze(0).repeat(num_layers, 1, 1)
    phases = [get_phase(layer_idx, num_layers, cfg.phase_ratios) for layer_idx in range(num_layers)]
    phase_positions = _phase_positions(phases)
    frame_ids = torch.arange(num_video_latent_frames, device=base_mask.device).repeat_interleave(video_tokens_per_frame)
    spatial_ids = torch.arange(video_tokens_per_frame, device=base_mask.device).repeat(num_video_latent_frames)

    for layer_idx in range(num_layers):
        phase = phases[layer_idx]
        rule = resolve_rule(cfg.mode, phase)
        if rule.temporal == "none":
            continue

        idx_in_phase, phase_len = phase_positions[layer_idx]
        local_window = compute_local_window(
            idx_in_phase=idx_in_phase,
            phase_len=phase_len,
            start=int(cfg.local_window_start),
            end=int(cfg.local_window_end),
        )
        temporal_visible = build_temporal_visible(
            action_seq_len=action_seq_len,
            num_video_latent_frames=num_video_latent_frames,
            temporal_rule=rule.temporal,
            coarse_lookahead=int(cfg.coarse_lookahead),
            local_window=local_window,
            device=base_mask.device,
        )
        spatial_visible = build_spatial_visible(
            video_grid_size=(f, h, w),
            spatial_rule=rule.spatial,
            coarse_spatial_stride=int(cfg.coarse_spatial_stride),
            device=base_mask.device,
        )
        visible_video_tokens = temporal_visible[:, frame_ids] & spatial_visible[spatial_ids].view(1, -1)
        layer_masks[layer_idx, video_seq_len:, :video_seq_len] |= visible_video_tokens

    return layer_masks


def _validate_base_mask(base_mask: torch.Tensor, video_seq_len: int, action_seq_len: int) -> None:
    if base_mask.dtype != torch.bool:
        raise ValueError(f"`base_mask` must be bool, got {base_mask.dtype}")
    if base_mask.ndim != 2:
        raise ValueError(f"`base_mask` must be 2D [S,S], got shape {tuple(base_mask.shape)}")
    total_seq_len = int(video_seq_len) + int(action_seq_len)
    if base_mask.shape != (total_seq_len, total_seq_len):
        raise ValueError(
            "`base_mask` shape mismatch: "
            f"mask={tuple(base_mask.shape)} vs expected={(total_seq_len, total_seq_len)}"
        )


def _validate_video_grid_size(video_grid_size: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(video_grid_size) != 3:
        raise ValueError(f"`video_grid_size` must be a 3-tuple (f,h,w), got {video_grid_size}")
    f, h, w = (int(video_grid_size[0]), int(video_grid_size[1]), int(video_grid_size[2]))
    if f <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"`video_grid_size` values must be positive, got {video_grid_size}")
    return f, h, w


def _phase_positions(phases: list[str]) -> dict[int, tuple[int, int]]:
    phase_to_indices: dict[str, list[int]] = {phase: [] for phase in PHASES}
    for layer_idx, phase in enumerate(phases):
        phase_to_indices[phase].append(layer_idx)

    positions: dict[int, tuple[int, int]] = {}
    for indices in phase_to_indices.values():
        phase_len = len(indices)
        for idx_in_phase, layer_idx in enumerate(indices):
            positions[layer_idx] = (idx_in_phase, phase_len)
    return positions
