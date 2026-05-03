# Agentic-RTMO

Agentic-RTMO 是一个基于 OpenMMLab MMPose / RTMO 的实时多人姿态估计实验项目。项目在 RTMO 的 DCC keypoint latent feature 上加入轻量级的 **Think-Critique-Act** 迭代修正机制，用 Structural Critic 评估当前关节预测，再由 Feature Refinement Actor 对关键点隐特征做残差式更新。

这个仓库保留 MMPose 的工程结构，便于直接复用原有训练、测试、推理和数据集配置；新增内容集中在 Agentic 模块、RTMOHead 接入逻辑和最小复现实验配置。

## Highlights

- **轻量增量改造**：不重跑 backbone / neck，只在 DCC latent feature 上做迭代修正。
- **可开关设计**：通过 `agentic_cfg.enabled` 控制，关闭后可回退到原 RTMO 路径。
- **结构感知反馈**：Structural Critic 基于关键点坐标、关节置信度和骨架邻接关系估计关节错误概率与位移提示。
- **最小复现配置**：提供 COCO 上基于 `rtmo-m` 的 Agentic 配置入口。
- **兼容原 MMPose 工具链**：继续使用 `tools/train.py`、`tools/test.py` 和 demo 推理脚本。

## Core Files

| 文件 | 作用 |
|---|---|
| `configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py` | Agentic-RTMO 最小训练配置 |
| `mmpose/models/heads/hybrid_heads/agentic_modules.py` | Structural Critic、Feature Refinement Actor、Think-Critique-Act Loop |
| `mmpose/models/heads/hybrid_heads/rtmo_head.py` | 在 RTMO DCC 前向过程中接入 Agentic loop |
| `METHOD_CODE_MAPPING.md` | 方法模块与代码位置映射 |
| `QUICKSTART_5MIN.md` | 5 分钟快速跑通检查 |
| `SETUP_RUN_EXPERIMENT_GUIDE.md` | 环境、训练、评估和排查说明 |

## Method Overview

Agentic-RTMO 的核心流程发生在 RTMOHead 的 DCC 模块内部：

1. **Think**：根据当前 keypoint latent feature 生成关键点预测和置信度。
2. **Critique**：Structural Critic 结合预测坐标、置信度和骨架拓扑，输出每个关节的错误概率 `error_prob` 与位移提示 `disp_hint`。
3. **Act**：Feature Refinement Actor 将 critic feedback 融入 keypoint latent feature，生成残差更新后的特征。
4. **Iterate**：默认迭代 `T=2`，由 `agentic_cfg.num_iters` 控制。

当前实现优先保证工程可运行和最小侵入，暂未强制加入额外结构监督损失；后续可在 `RTMOHead.loss` 中扩展 `Lerr` / `Ldisp` 等监督项。

## Installation

建议使用独立 Python 环境。以下命令以 Linux / macOS 为例，Windows 可使用 Conda 或 PowerShell 虚拟环境。

```bash
git clone https://github.com/chivector/agentic-rtmo.git
cd agentic-rtmo

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

按你的 CUDA 版本安装 PyTorch，例如：

```bash
pip install torch torchvision torchaudio
```

安装 OpenMMLab 依赖和本项目：

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"
pip install -r requirements.txt
pip install -v -e .
```

如果只想先确认核心文件能被 Python 编译：

```bash
python -m compileall mmpose/models/heads/hybrid_heads/agentic_modules.py
python -m compileall mmpose/models/heads/hybrid_heads/rtmo_head.py
python -m compileall configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

## Dataset

默认配置使用 COCO keypoint 数据集。请按 MMPose 的数据集组织方式将 COCO 放到 `data/coco`：

```text
data/coco/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
  train2017/
  val2017/
```

本仓库不包含 COCO 数据集和训练权重。

## Training

单卡训练：

```bash
python tools/train.py \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

多卡训练：

```bash
bash tools/dist_train.sh \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py \
  8 --amp
```

Baseline 对比配置：

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

建议在同一机器、同一数据、同一 batch size 下记录 baseline 和 Agentic-RTMO 的 AP、AP50、AP75、FPS / latency。

## Inference Demo

```bash
python demo/inferencer_demo.py <IMAGE_PATH> \
  --pose2d rtmo \
  --pose2d-weights <YOUR_CHECKPOINT>.pth \
  --vis-out-dir vis_results
```

## Agentic Config

Agentic loop 的主要开关位于：

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

常用消融项：

- `enabled=False`：关闭 Agentic loop，回退到原 RTMO DCC 路径。
- `num_iters=0/1/2/3`：比较迭代次数和时延的折中。
- `critic_hidden_dim` / `actor_hidden_dim`：控制 critic / actor 的轻量 MLP 宽度。
- `residual_scale`：控制 actor 残差更新幅度。

## Experiment Template

```text
Date:
Machine/GPU:
Dataset:

[Baseline RTMO]
Config: configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py
Checkpoint:
AP:
AP50:
AP75:
FPS/Latency:

[Agentic-RTMO]
Config: configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
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

- 已完成 Agentic 模块和 RTMOHead 的最小工程接入。
- 已提供 COCO / RTMO-M 的最小复现配置。
- 当前仓库未附带完整训练日志、模型权重或最终论文级指标。
- 额外结构监督、teacher forcing、ONNX / TensorRT 导出分支仍可继续扩展。

## Acknowledgements

本项目基于 [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose) 和 RTMO 代码结构进行二次开发。感谢 OpenMMLab 社区提供的开源工具链、模型实现和文档。

## License

本项目沿用 Apache License 2.0。请保留原始 MMPose / OpenMMLab 版权声明，并遵守 `LICENSE` 与 `LICENSES.md` 中的条款。
