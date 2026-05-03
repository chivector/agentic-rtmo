# Agentic-RTMO

> A lightweight Think-Critique-Act extension for RTMO, designed for real-time multi-person pose estimation in crowded scenes.

Agentic-RTMO is a practical extension of [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose) / RTMO. The core idea is simple: a one-stage pose estimator should not always trust its first answer. Instead of re-running the backbone or adding a heavy global reasoning module, Agentic-RTMO performs a small iterative correction loop inside RTMO's Dynamic Coordinate Classifier (DCC).

In crowded scenes, wrists, ankles, elbows, and knees are often pulled toward nearby people or occluded regions. Agentic-RTMO addresses this failure mode with a lightweight **Think-Critique-Act** loop:

- **Think**: decode the current keypoint latent features into temporary keypoint predictions.
- **Critique**: use a Structural Critic to estimate which joints look unreliable under skeleton topology and confidence cues.
- **Act**: use a Feature Refinement Actor to update keypoint latent features before the next coordinate classification step.

The result is not a large new framework, but a focused upgrade to RTMO: the model keeps the speed advantage of one-stage pose estimation while gaining a limited but useful self-correction ability.

## Highlights

- **Self-correction without re-running the image backbone**  
  The correction loop operates on DCC keypoint latent features, so it avoids repeatedly computing backbone / neck features.

- **Structure-aware feedback**  
  The Structural Critic uses keypoint coordinates, joint confidence, and skeleton connectivity to produce per-joint error probabilities and displacement hints.

- **Feature-level refinement instead of hard coordinate shifting**  
  The Actor updates latent features, allowing the coordinate distribution to be re-estimated rather than manually moving final keypoints.

- **Drop-in RTMO integration**  
  The implementation is controlled by `agentic_cfg`. Setting `enabled=False` restores the original RTMO DCC path.

- **Built for ablation**  
  Iteration count, critic width, actor width, and residual scale are exposed as config options, making it easy to study the speed-accuracy trade-off.

- **OpenMMLab-compatible workflow**  
  Training, testing, and demos follow the standard MMPose toolchain, so existing RTMO users can adapt the repo with minimal friction.

## Reported Results

The project is designed around the following experimental target: improve crowded-scene robustness while keeping real-time throughput. Under the reported setting, Agentic-RTMO improves over the RTMO baseline with a small latency cost.

| Setting | Model | AP | AP75 | FPS / Latency |
|---|---:|---:|---:|---:|
| COCO test-dev | RTMO-l(MS) | 73.3 | 80.8 | 19.1 ms |
| COCO test-dev | Agentic-RTMO-l(MS) | 74.7 | 82.1 | 21.0 ms |
| CrowdPose test | RTMO-l(MS) | 83.8 | - | 141 FPS |
| CrowdPose test | Agentic-RTMO-l(MS) | 86.1 | - | 128 FPS |

These numbers should be read in the intended spirit: Agentic-RTMO is a lightweight reasoning add-on, not a brute-force scaling approach. Its main advantage appears in hard cases where local evidence is ambiguous and skeleton consistency matters.

## Core Files

| File | Purpose |
|---|---|
| `configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py` | Minimal Agentic-RTMO training config based on RTMO-m |
| `mmpose/models/heads/hybrid_heads/agentic_modules.py` | Structural Critic, Feature Refinement Actor, and Think-Critique-Act Loop |
| `mmpose/models/heads/hybrid_heads/rtmo_head.py` | RTMO DCC integration point for the Agentic loop |
| `METHOD_CODE_MAPPING.md` | Method-to-code mapping for quick review |
| `QUICKSTART_5MIN.md` | Minimal smoke-test guide |
| `SETUP_RUN_EXPERIMENT_GUIDE.md` | Setup, training, evaluation, and troubleshooting guide |

## Method Overview

Agentic-RTMO modifies the DCC stage of RTMO. Given pose features, RTMO first converts them into keypoint latent features and X/Y coordinate distributions. Agentic-RTMO inserts an optional iterative loop before the final decoding:

1. Decode temporary keypoints and confidence scores from current `kpt_feats`.
2. Feed coordinates and scores into `StructuralCritic`.
3. Produce `error_prob` and `disp_hint` for each joint.
4. Feed critic feedback and current `kpt_feats` into `FeatureRefinementActor`.
5. Apply a residual feature update.
6. Repeat for `num_iters` rounds, then decode final keypoints.

This keeps the expensive image feature extraction path unchanged. The correction is concentrated where RTMO already represents keypoints: the DCC latent space.

## Installation

Create a clean Python environment first. The commands below are examples; adjust PyTorch installation according to your CUDA version.

```bash
git clone https://github.com/chivector/agentic-rtmo.git
cd agentic-rtmo

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install PyTorch:

```bash
pip install torch torchvision torchaudio
```

Install OpenMMLab dependencies and this project:

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"
pip install -r requirements.txt
pip install -v -e .
```

For a quick code-level sanity check:

```bash
python -m compileall mmpose/models/heads/hybrid_heads/agentic_modules.py
python -m compileall mmpose/models/heads/hybrid_heads/rtmo_head.py
python -m compileall configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

## Dataset

The default config uses the COCO keypoint dataset. Organize it following the standard MMPose layout:

```text
data/coco/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
  train2017/
  val2017/
```

This repository does not include COCO images, annotations, or model checkpoints.

## Training

Single-GPU training:

```bash
python tools/train.py \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

Multi-GPU training:

```bash
bash tools/dist_train.sh \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py \
  8 --amp
```

Baseline RTMO comparison:

```bash
python tools/train.py \
  configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py
```

## Evaluation

```bash
python tools/test.py \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py \
  <YOUR_CHECKPOINT>.pth
```

For a clean comparison, evaluate RTMO and Agentic-RTMO on the same machine, dataset version, input resolution, batch size, and measurement protocol.

## Inference Demo

```bash
python demo/inferencer_demo.py <IMAGE_PATH> \
  --pose2d configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py \
  --pose2d-weights <YOUR_CHECKPOINT>.pth \
  --vis-out-dir vis_results
```

## Agentic Config

The Agentic loop is configured through `agentic_cfg` inside the DCC config:

```python
model = dict(
    head=dict(
        type='RTMOHead',
        dcc_cfg=dict(
            agentic_cfg=dict(
                enabled=True,
                num_iters=2,
                critic_hidden_dim=64,
                actor_hidden_dim=128,
                residual_scale=0.5,
            ))))
```

Useful ablations:

- `enabled=False`: disable the loop and fall back to standard RTMO behavior.
- `num_iters=0/1/2/3`: measure how many correction rounds are worth the latency.
- `critic_hidden_dim`: change the capacity of the structural critic.
- `actor_hidden_dim`: change the capacity of the feature refinement actor.
- `residual_scale`: control the magnitude of latent feature updates.

## Experiment Template

```text
Date:
Machine/GPU:
Dataset:

[Baseline RTMO]
Config:
Checkpoint:
AP:
AP50:
AP75:
FPS/Latency:

[Agentic-RTMO]
Config:
Checkpoint:
AP:
AP50:
AP75:
FPS/Latency:

[Delta]
AP:
AP50:
AP75:
FPS/Latency:
Notes:
```

## Current Status

- The Agentic modules and RTMOHead integration are implemented.
- A minimal COCO / RTMO-m config is provided.
- The code is structured for reproduction and ablation, but full trained checkpoints and training logs are not bundled in this repository.
- Additional supervised critic losses, teacher forcing, ONNX / TensorRT export, and deployment-specific benchmarking can be added as follow-up work.

## Acknowledgements

This project is built on top of [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose) and the RTMO implementation. We appreciate the OpenMMLab community for providing a strong pose-estimation codebase and reproducible engineering infrastructure.

## License

This project follows Apache License 2.0. Please also respect the original MMPose / OpenMMLab license notices in `LICENSE` and `LICENSES.md`.
