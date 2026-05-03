# QUICKSTART 5MIN：最快跑通 Agentic-RTMO

目标：用最少命令先跑通，再进入深入复核。

## 1. 最少命令

在 `mmpose-main` 根目录执行：

```bash
# 1) 激活你的环境（示例）
source .venv/bin/activate

# 2) 用 Agentic 配置启动训练
python tools/train.py \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

> 如果你先只想验证配置可加载，可先执行：

```bash
python -m compileall \
  configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
```

## 2. 关键验收点（5 分钟内可检查）

1. 日志中成功构建 `RTMOHead`，且未出现模块导入错误。  
2. 训练开始后无首轮 shape 报错。  
3. 使用的配置文件为：`agentic-rtmo-m_16xb16-600e_coco-640x640.py`。  
4. 代码层面已启用 `agentic_cfg.enabled=True` 且 `num_iters=2`。  

## 3. baseline 对比模板（直接复制）

```text
实验日期:
机器/GPU:
数据集:

[Baseline]
Config: configs/body_2d_keypoint/rtmo/coco/rtmo-m_16xb16-600e_coco-640x640.py
CKPT:
AP:
AP50:
AP75:
FPS/Latency:

[Agentic-RTMO]
Config: configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py
CKPT:
AP:
AP50:
AP75:
FPS/Latency:

[Delta]
AP:
AP50:
AP75:
FPS/Latency:
备注:
```

## 4. 跑通后下一步

- 读 `METHOD_CODE_MAPPING.md`：先理解创新点与代码映射。  
- 读 `SETUP_RUN_EXPERIMENT_GUIDE.md`：按完整流程做实验复核与排查。  
