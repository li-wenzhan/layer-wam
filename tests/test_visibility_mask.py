import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.visibility_mask import (
    VisibilityConfig,
    build_layerwise_visibility_mask,
    build_spatial_visible,
    compute_action_latent_alignment,
    compute_local_window,
)


def make_base_mask(video_frames=3, h=2, w=3, action_seq_len=32):
    video_tokens_per_frame = h * w
    video_seq_len = video_frames * video_tokens_per_frame
    total_seq_len = video_seq_len + action_seq_len
    base = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool)
    base[:video_seq_len, :video_seq_len] = True
    base[video_seq_len:, video_seq_len:] = True
    base[video_seq_len:, :video_tokens_per_frame] = True
    return base, video_seq_len, action_seq_len, video_tokens_per_frame, (video_frames, h, w)


def build_mask(mode):
    base, video_seq_len, action_seq_len, tpf, grid = make_base_mask()
    return build_layerwise_visibility_mask(
        base_mask=base,
        num_layers=30,
        video_seq_len=video_seq_len,
        action_seq_len=action_seq_len,
        video_tokens_per_frame=tpf,
        video_grid_size=grid,
        cfg=VisibilityConfig(mode=mode),
        training=False,
    )


def future_visible_by_layer(mask, video_seq_len, video_tokens_per_frame):
    return mask[:, video_seq_len:, video_tokens_per_frame:video_seq_len].any(dim=(1, 2))


def as_layerwise(mask):
    return mask.unsqueeze(0).expand(30, -1, -1) if mask.ndim == 2 else mask


class VisibilityMaskTest(unittest.TestCase):
    def test_latent_level_action_alignment(self):
        aligned, ratio = compute_action_latent_alignment(
            action_seq_len=32,
            num_video_latent_frames=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(ratio, 16)
        self.assertTrue(torch.equal(aligned[:16], torch.ones(16, dtype=torch.long)))
        self.assertTrue(torch.equal(aligned[16:], torch.full((16,), 2, dtype=torch.long)))

    def test_only_first_frame_returns_base_mask(self):
        base, video_seq_len, action_seq_len, tpf, grid = make_base_mask(video_frames=1)
        mask = build_layerwise_visibility_mask(
            base_mask=base,
            num_layers=30,
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=tpf,
            video_grid_size=grid,
            cfg=VisibilityConfig(mode="clcf"),
            training=False,
        )
        self.assertEqual(mask.ndim, 2)
        self.assertTrue(torch.equal(mask, base))

    def test_p2_window_interpolation(self):
        windows = [compute_local_window(i, 6, start=2, end=0) for i in range(6)]
        self.assertEqual(windows, [2, 2, 1, 1, 0, 0])

    def test_visible_budget_for_matched_modes(self):
        for mode in ("early_matched", "sandwich_dense", "late_matched", "clcf"):
            with self.subTest(mode=mode):
                mask = build_mask(mode)
                _, video_seq_len, _, tpf, _ = make_base_mask()
                self.assertEqual(int(future_visible_by_layer(mask, video_seq_len, tpf).sum().item()), 12)

    def test_endpoint_modes_match_original_2d_masks(self):
        base, video_seq_len, _, _, _ = make_base_mask()

        fastwam_mask = build_mask("fastwam")
        self.assertEqual(fastwam_mask.ndim, 2)
        self.assertTrue(torch.equal(fastwam_mask, base))

        joint_expected = base.clone()
        joint_expected[video_seq_len:, :video_seq_len] = True
        joint_mask = build_mask("joint")
        self.assertEqual(joint_mask.ndim, 2)
        self.assertTrue(torch.equal(joint_mask, joint_expected))

    def test_clcf_p3_closes_future_video(self):
        mask = build_mask("clcf")
        _, video_seq_len, _, tpf, _ = make_base_mask()
        self.assertFalse(mask[18:30, video_seq_len:, tpf:video_seq_len].any().item())

    def test_ablations_are_not_sandwich_duplicates(self):
        sandwich = build_mask("sandwich_dense")
        self.assertFalse(torch.equal(build_mask("clcf_wo_temporal_local"), sandwich))
        self.assertFalse(torch.equal(build_mask("clcf_wo_spatial_c2f"), sandwich))

    def test_first_frame_visibility_is_preserved(self):
        base, video_seq_len, action_seq_len, tpf, _ = make_base_mask()
        for mode in (
            "fastwam",
            "joint",
            "early_matched",
            "sandwich_dense",
            "late_matched",
            "clcf",
            "clcf_wo_causal",
            "clcf_wo_temporal_local",
            "clcf_wo_spatial_c2f",
        ):
            with self.subTest(mode=mode):
                mask = build_mask(mode)
                layerwise_mask = as_layerwise(mask)
                self.assertTrue(layerwise_mask[:, video_seq_len:, :tpf].all().item())
                self.assertTrue(torch.equal(layerwise_mask[:, :video_seq_len, :video_seq_len], base[:video_seq_len, :video_seq_len].expand(30, -1, -1)))
                self.assertFalse(layerwise_mask[:, :video_seq_len, video_seq_len:video_seq_len + action_seq_len].any().item())

    def test_sparse_spatial_rule_uses_2d_grid(self):
        visible = build_spatial_visible(
            video_grid_size=(3, 3, 5),
            spatial_rule="sparse",
            coarse_spatial_stride=2,
            device=torch.device("cpu"),
        )
        expected = torch.zeros((15,), dtype=torch.bool)
        for y in (0, 2):
            for x in (0, 2, 4):
                expected[y * 5 + x] = True
        self.assertTrue(torch.equal(visible, expected))

    def test_video_self_mask_from_3d_joint_mask(self):
        base, video_seq_len, _, _, _ = make_base_mask()
        layerwise = base.unsqueeze(0).repeat(30, 1, 1)
        self_mask = FastWAM._video_self_mask_from_joint_mask(layerwise, video_seq_len)
        self.assertEqual(tuple(self_mask.shape), (video_seq_len, video_seq_len))
        self.assertTrue(torch.equal(self_mask, base[:video_seq_len, :video_seq_len]))

    def test_mot_layer_mask_selector_accepts_2d_and_3d(self):
        mask_2d = torch.zeros((4, 4), dtype=torch.bool)
        mask_3d = torch.zeros((3, 4, 4), dtype=torch.bool)
        mask_3d[2] = True
        self.assertIs(MoT._select_layer_attention_mask(mask_2d, layer_idx=1, num_layers=3), mask_2d)
        self.assertTrue(MoT._select_layer_attention_mask(mask_3d, layer_idx=2, num_layers=3).all().item())


if __name__ == "__main__":
    unittest.main()
