# CLCF-WAM / LayerWAM 实验运行指南

本文档用于快速运行本仓库中基于 Fast-WAM 改造的 CLCF-WAM / LayerWAM 实验。原始 Fast-WAM 的环境、数据、权重准备流程仍然适用；本文重点说明新增的 layer-wise 3D visibility mask 实验如何启动、调试、训练和评估。

默认资源假设为单机 `4 * H100 80GB`。如果 GPU 数量不同，请相应修改命令中的进程数、batch size 或 Deepspeed 配置。

## 1. 核心改动速览

新增模型配置：

```text
configs/model/fastwam_3dmask.yaml
```

新增核心代码：

```text
src/fastwam/models/wan22/visibility_mask.py
src/fastwam/models/wan22/fastwam_3dmask.py
```

新增实验工具：

```text
scripts/debug_visibility_mask.py
scripts/run_clcf_ablation.sh
tests/test_visibility_mask.py
```

主要实验不再通过新增多个模型类完成，而是统一使用：

```bash
model=fastwam_3dmask
model.visibility.mode=<mode_name>
```

支持的 `mode_name`：

```text
fastwam
joint
early_matched
sandwich_dense
late_matched
clcf
clcf_wo_causal
clcf_wo_temporal_local
clcf_wo_spatial_c2f
```

其中 `fastwam` / `joint` 是两个端点，`future_mask_dropout=0`；其余 scheduled variants 默认使用 `future_mask_dropout=0.3`。

## 2. 环境安装

如果已经按原 Fast-WAM README 配好环境，可以跳过本节。

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

建议确认当前环境可以导入核心依赖：

```bash
python - <<'PY'
import torch
import fastwam
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

## 3. 模型准备

设置 Wan / Fast-WAM 权重目录：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

预生成 ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

CLCF-WAM 使用同一个 ActionDiT backbone，不需要额外生成新的 backbone。

## 4. 数据准备

沿用原 Fast-WAM 数据格式。

LIBERO 数据目录应类似：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

RoboTwin 数据目录应类似：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

首次训练新任务时，如果没有可用的统计文件，请先把对应 `configs/data/*.yaml` 中的 `pretrained_norm_stats` 设为 `null`。第一次训练会在 run 目录生成 `dataset_stats.json`，后续训练建议复用该文件。

## 5. 预计算文本 Embedding

训练前建议先预计算 T5 embedding cache：

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

4 卡并行版本：

```bash
torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

## 6. 先检查 Visibility Mask

正式训练前，建议先用 debug 脚本检查 mask 是否符合预期：

```bash
python scripts/debug_visibility_mask.py --mode clcf
```

默认模拟当前 LIBERO 训练设置下的 latent-level token 结构：

```text
video latent frames = 3
future latent frames = 2
action_seq_len = 32
action_latent_freq_ratio = 16
```

检查某个消融：

```bash
python scripts/debug_visibility_mask.py --mode clcf_wo_temporal_local
python scripts/debug_visibility_mask.py --mode clcf_wo_spatial_c2f
```

检查训练态 dropout 是否可能触发：

```bash
python scripts/debug_visibility_mask.py \
  --mode clcf \
  --training \
  --future-mask-dropout 0.3 \
  --seed 0
```

输出中的关键字段：

```text
direct_action_to_future_visible_layers
reference_visible_layers_without_dropout
p3_future_open
actual_dropout_applied
```

对 `clcf` 而言，30 层模型中 future-visible layers 应为 `12`，且 `p3_future_open=false`。

## 7. 单独训练某个 Mode

推荐 4 卡 H100 使用 Deepspeed ZeRO-2：

```bash
bash scripts/train_zero2.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_3dmask \
  model.visibility.mode=clcf \
  model.visibility.future_mask_dropout=0.3
```

训练 FastWAM 端点：

```bash
bash scripts/train_zero2.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_3dmask \
  model.visibility.mode=fastwam \
  model.visibility.future_mask_dropout=0.0
```

训练 Joint 端点：

```bash
bash scripts/train_zero2.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_3dmask \
  model.visibility.mode=joint \
  model.visibility.future_mask_dropout=0.0
```

训练其它主表变体：

```bash
# Early, budget matched
bash scripts/train_zero2.sh 4 task=libero_uncond_2cam224_1e-4 model=fastwam_3dmask model.visibility.mode=early_matched model.visibility.future_mask_dropout=0.3

# Sandwich dense, budget matched
bash scripts/train_zero2.sh 4 task=libero_uncond_2cam224_1e-4 model=fastwam_3dmask model.visibility.mode=sandwich_dense model.visibility.future_mask_dropout=0.3

# Late, budget matched
bash scripts/train_zero2.sh 4 task=libero_uncond_2cam224_1e-4 model=fastwam_3dmask model.visibility.mode=late_matched model.visibility.future_mask_dropout=0.3
```

训练 CLCF 内部消融：

```bash
# 去掉 direct causal 约束
bash scripts/train_zero2.sh 4 task=libero_uncond_2cam224_1e-4 model=fastwam_3dmask model.visibility.mode=clcf_wo_causal model.visibility.future_mask_dropout=0.3

# 去掉 P2 temporal locality
bash scripts/train_zero2.sh 4 task=libero_uncond_2cam224_1e-4 model=fastwam_3dmask model.visibility.mode=clcf_wo_temporal_local model.visibility.future_mask_dropout=0.3

# 去掉 P1 spatial coarse-to-fine sparse anchors
bash scripts/train_zero2.sh 4 task=libero_uncond_2cam224_1e-4 model=fastwam_3dmask model.visibility.mode=clcf_wo_spatial_c2f model.visibility.future_mask_dropout=0.3
```

RoboTwin 任务只需替换 task：

```bash
bash scripts/train_zero2.sh 4 \
  task=robotwin_uncond_3cam_384_1e-4 \
  model=fastwam_3dmask \
  model.visibility.mode=clcf \
  model.visibility.future_mask_dropout=0.3
```

## 8. 一键跑完整 Ablation

脚本会按顺序运行全部第一版 mode：

```bash
bash scripts/run_clcf_ablation.sh task=libero_uncond_2cam224_1e-4
```

默认使用：

```text
NPROC_PER_NODE=4
FUTURE_MASK_DROPOUT=0.3
```

可以显式指定：

```bash
NPROC_PER_NODE=4 \
FUTURE_MASK_DROPOUT=0.3 \
RUN_ID_PREFIX=libero_clcf_v1 \
bash scripts/run_clcf_ablation.sh task=libero_uncond_2cam224_1e-4
```

继续追加 Hydra overrides：

```bash
bash scripts/run_clcf_ablation.sh \
  task=libero_uncond_2cam224_1e-4 \
  batch_size=8 \
  eval_every=500 \
  save_every=2000
```

注意：`RUN_ID_PREFIX` 建议每轮实验设为不同值，避免输出目录名冲突。

## 9. Validation 指标解释

本仓库已将训练期 validation 拆成两条路径：

```text
infer_action()  -> 主 action 指标
infer_joint()   -> video rollout diagnostic
```

主 action 指标：

```text
eval/action_only_l1
eval/action_only_l2
```

video diagnostic：

```text
eval/psnr_rg
eval/ssim_rg
eval/psnr_rd
eval/ssim_rd
eval/psnr_dg
eval/ssim_dg
```

joint action 只作为诊断项：

```text
eval/joint_action_l1_diagnostic
eval/joint_action_l2_diagnostic
```

论文主表或成功率统计应优先使用 `infer_action()` 对应的 action-only 路径，不要把 joint rollout 的 action 当成主 action 指标。

## 10. 评估训练好的 Checkpoint

LIBERO：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/<task>/<run_id>/checkpoints/weights/<step>.pt \
  MULTIRUN.num_gpus=4
```

如果使用自定义 dataset stats：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/<task>/<run_id>/checkpoints/weights/<step>.pt \
  EVALUATION.dataset_stats_path=./runs/<task>/<run_id>/dataset_stats.json \
  MULTIRUN.num_gpus=4
```

RoboTwin：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./runs/<task>/<run_id>/checkpoints/weights/<step>.pt \
  MULTIRUN.num_gpus=4
```

## 11. 推荐实验顺序

建议第一轮按以下顺序运行：

```text
1. fastwam
2. joint
3. early_matched
4. sandwich_dense
5. late_matched
6. clcf
7. clcf_wo_causal
8. clcf_wo_temporal_local
9. clcf_wo_spatial_c2f
```

对应的主要对比关系：

```text
fastwam vs joint:
  future video direct visibility 的两个端点

early_matched / sandwich_dense / late_matched:
  相同 visible layer budget 下的位置对比

clcf vs sandwich_dense:
  中层可见相同预算下，验证 causal/local/spatial 结构是否有效

clcf vs clcf_wo_causal:
  验证 direct-edge causal 约束

clcf vs clcf_wo_temporal_local:
  验证 P2 temporal locality

clcf vs clcf_wo_spatial_c2f:
  验证 P1 spatial coarse-to-fine sparse anchors
```

## 12. 常见问题

### 12.1 为什么 CLCF 的时间对齐不是 `action_video_freq_ratio=4`？

CLCF mask 工作在 MoT latent token 层级。当前数据流是：

```text
33 raw frames
-> 每 4 帧采样
-> 9 sampled video frames
-> Wan VAE temporal downsample x4
-> 3 latent video frames
```

因此训练时实际是：

```text
32 actions / 2 future latent frames = 16 actions per future latent
```

也就是：

```text
a0-a15  -> future latent frame 1
a16-a31 -> future latent frame 2
```

### 12.2 CLCF 是否实现严格因果信息隔离？

不是。本项目第一版只控制：

```text
direct action-to-future-video visibility
```

即 action query 直接读取哪些 future video key。它是 direct-edge causal / local，不是 strict causal information flow。我们不修改 video-to-video mask。

### 12.3 `infer_action()` 中为什么没有 future video？

这是 Fast-WAM 的核心推理路径：推理时只给 current image / first-frame latent，直接 denoise action，不显式生成 future video。对于 CLCF-WAM，只有 first-frame latent 时，3D visibility mask 会自动退化为 base no-future mask。

### 12.4 跑单元测试报 `ModuleNotFoundError: No module named 'torch'` 怎么办？

说明当前 Python 环境不是项目环境。请先：

```bash
conda activate fastwam
pip install -e .
python -m unittest tests.test_visibility_mask
```

### 12.5 4 张 H100 显存不够怎么办？

优先尝试：

```bash
batch_size=8
model.mot_checkpoint_mixed_attn=true
gradient_accumulation_steps=2
```

例如：

```bash
bash scripts/train_zero2.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  model=fastwam_3dmask \
  model.visibility.mode=clcf \
  batch_size=8 \
  gradient_accumulation_steps=2 \
  model.mot_checkpoint_mixed_attn=true
```

