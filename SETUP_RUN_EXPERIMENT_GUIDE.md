# Agentic-RTMO 工程配置与实验指南

本文档面向接手者，提供从环境到运行、再到实验复核与排查的完整流程。

## 1. 目录与关键文件

- 新增方法说明：`METHOD_CODE_MAPPING.md`
- 快速复现：`QUICKSTART_5MIN.md`
- Agentic 配置：`configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py`
- 核心改造代码：
  - `mmpose/models/heads/hybrid_heads/agentic_modules.py`
  - `mmpose/models/heads/hybrid_heads/rtmo_head.py`

## 2. 环境增量配置（建议）

> 以下步骤是“从零到可运行”的推荐顺序；如果你已有 OpenMMLab 环境，可按需跳过。

### 2.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### 2.2 安装 PyTorch（按你的 CUDA 版本选择）

示例（请按官网替换为你的 CUDA 对应命令）：

```bash
pip install torch torchvision torchaudio
```

### 2.3 安装 OpenMMLab 依赖与本项目

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"
pip install -r requirements.txt
pip install -v -e .
```

### 2.4 可选：快速健康检查

```bash
python -m compileall mmpose/models/heads/hybrid_heads/agentic_modules.py
python -m compileall mmpose/models/heads/hybrid_heads/rtmo_head.py
python -m compileall configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

## 3. 运行方式

## 3.1 单卡训练（最直接）

```bash
python tools/train.py \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

## 3.2 多卡训练（与 baseline 一致）

```bash
bash tools/dist_train.sh \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py \
  8 --amp
```

## 3.3 测试评估

```bash
python tools/test.py \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py \
  <YOUR_CHECKPOINT>.pth
```

## 3.4 推理（demo）

```bash
python demo/inferencer_demo.py <IMAGE_PATH> \
  --pose2d rtmo \
  --pose2d-weights <YOUR_CHECKPOINT>.pth \
  --vis-out-dir vis_results
```

## 4. 实验建议流程（最小可复核）

1. **先跑 baseline**：`rtmo-m_16xb16-600e_coco-640x640.py`。  
2. **再跑 agentic**：`agentic-rtmo-m_16xb16-600e_coco-640x640.py`。  
3. 记录同一环境、同一数据下的 AP/FPS 对比，重点看 crowded 场景变化。  
4. 做一次 `T` 消融（`0/1/2/3`）验证迭代收益与时延折中。  

## 5. 常见问题与排查

### 5.1 `ImportError` / `No module named mmcv|mmdet|mmengine`

- 原因：环境依赖未完整安装。  
- 处理：重新执行 2.3，确保 `pip install -v -e .` 成功。  

### 5.2 配置能加载但训练报 shape mismatch

- 检查 `head.dcc_cfg` 中 `in_channels/feat_channels/num_bins` 是否与当前 head 输出一致。  
- 若你改了 backbone/neck 宽度，请同步检查 `head_module_cfg.pose_vec_channels`。  

### 5.3 推理速度下降明显

- 优先检查 `num_iters` 是否 > 2。  
- 关闭可视化、减少日志和保存频率，避免 I/O 干扰。  

### 5.4 导出部署异常

- 当前 agentic 模式保留了原前向路径优先稳定运行。  
- 如需 ONNX/TensorRT 强一致导出，建议先固定 `num_iters` 并单独加导出分支。  

## 6. 交接验收清单

- [ ] baseline 训练/推理可跑通  
- [ ] agentic 配置可跑通  
- [ ] 同机同卡完成 baseline vs agentic 对比记录  
- [ ] 关键日志保留（命令、配置、ckpt、评估结果）  
- [ ] 复核 `METHOD_CODE_MAPPING.md` 与代码一致  
