import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.wan22.visibility_mask import (
    SCHEDULED_MODES,
    SUPPORTED_MODES,
    VisibilityConfig,
    build_layerwise_visibility_mask,
    get_phase,
)


def make_base_mask(video_frames: int, h: int, w: int, action_seq_len: int):
    video_tokens_per_frame = h * w
    video_seq_len = video_frames * video_tokens_per_frame
    total_seq_len = video_seq_len + action_seq_len
    base = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool)
    base[:video_seq_len, :video_seq_len] = True
    base[video_seq_len:, video_seq_len:] = True
    base[video_seq_len:, :video_tokens_per_frame] = True
    return base, video_seq_len, action_seq_len, video_tokens_per_frame, (video_frames, h, w)


def count_future_visible_layers(
    mask: torch.Tensor,
    video_seq_len: int,
    video_tokens_per_frame: int,
    num_layers: int,
) -> int:
    if mask.ndim == 2:
        future_visible = bool(mask[video_seq_len:, video_tokens_per_frame:video_seq_len].any().item())
        return int(num_layers) if future_visible else 0
    future_block = mask[:, video_seq_len:, video_tokens_per_frame:video_seq_len]
    return int(future_block.any(dim=(1, 2)).sum().item())


def main():
    parser = argparse.ArgumentParser(description="Inspect direct action-to-future-video visibility masks.")
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), default="clcf")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--video-frames", type=int, default=3)
    parser.add_argument("--height-tokens", type=int, default=14)
    parser.add_argument("--width-tokens", type=int, default=28)
    parser.add_argument("--action-seq-len", type=int, default=32)
    parser.add_argument("--future-mask-dropout", type=float, default=0.0)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    base, video_seq_len, action_seq_len, tpf, grid = make_base_mask(
        video_frames=args.video_frames,
        h=args.height_tokens,
        w=args.width_tokens,
        action_seq_len=args.action_seq_len,
    )
    cfg = VisibilityConfig(
        mode=args.mode,
        future_mask_dropout=args.future_mask_dropout,
    )
    mask = build_layerwise_visibility_mask(
        base_mask=base,
        num_layers=args.num_layers,
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=tpf,
        video_grid_size=grid,
        cfg=cfg,
        training=args.training,
    )
    reference_cfg = VisibilityConfig(mode=args.mode, future_mask_dropout=0.0)
    reference_mask = build_layerwise_visibility_mask(
        base_mask=base,
        num_layers=args.num_layers,
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=tpf,
        video_grid_size=grid,
        cfg=reference_cfg,
        training=False,
    )

    visible_layers = count_future_visible_layers(mask, video_seq_len, tpf, args.num_layers)
    reference_visible_layers = count_future_visible_layers(reference_mask, video_seq_len, tpf, args.num_layers)
    actual_dropout = (
        args.training
        and args.mode in SCHEDULED_MODES
        and args.future_mask_dropout > 0.0
        and reference_visible_layers > 0
        and visible_layers == 0
    )

    layerwise = mask if mask.ndim == 3 else mask.unsqueeze(0).expand(args.num_layers, -1, -1)
    p3_layers = [
        layer_idx
        for layer_idx in range(layerwise.shape[0])
        if get_phase(layer_idx, args.num_layers, cfg.phase_ratios) == "p3"
    ]
    p3_future_open = bool(layerwise[p3_layers, video_seq_len:, tpf:video_seq_len].any().item()) if p3_layers else False

    payload = {
        "mode": args.mode,
        "mask_shape": list(mask.shape),
        "video_grid_size": list(grid),
        "video_seq_len": video_seq_len,
        "video_tokens_per_frame": tpf,
        "action_seq_len": action_seq_len,
        "future_mask_dropout": args.future_mask_dropout,
        "training": bool(args.training),
        "actual_dropout_applied": actual_dropout,
        "direct_action_to_future_visible_layers": visible_layers,
        "reference_visible_layers_without_dropout": reference_visible_layers,
        "p3_future_open": p3_future_open,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
