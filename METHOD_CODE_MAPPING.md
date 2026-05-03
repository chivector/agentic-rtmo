# Agentic-RTMO 方法与代码映射说明

本文档用于说明：论文中的创新点，在当前代码中具体映射到哪里，以及核心改造思路是什么。

## 1. 论文创新点与代码映射

| 论文模块/概念 | 代码落点 | 关键符号/函数 | 说明 |
|---|---|---|---|
| Structural Critic | `mmpose/models/heads/hybrid_heads/agentic_modules.py` | `class StructuralCritic` | 输入当前关键点坐标和关节置信度，输出每关节错误概率 `error_prob` 与位移提示 `disp_hint`。 |
| Feature Refinement Actor | `mmpose/models/heads/hybrid_heads/agentic_modules.py` | `class FeatureRefinementActor` | 接收 `kpt_feats + error_prob + disp_hint`，做残差式 latent feature 更新。 |
| Think-Critique-Act Loop | `mmpose/models/heads/hybrid_heads/agentic_modules.py` | `class ThinkCritiqueActLoop` | 封装一次迭代：Think(当前预测) -> Critique(结构评估) -> Act(特征更新)。 |
| 在 DCC latent 上迭代（不重跑 backbone） | `mmpose/models/heads/hybrid_heads/rtmo_head.py` | `DCC.forward_train` / `DCC.forward_test` | 在 keypoint latent feature (`kpt_feats`) 上循环更新；backbone/neck 输出不重复计算。 |
| 迭代次数 T（默认 2） | `mmpose/models/heads/hybrid_heads/rtmo_head.py` + 配置文件 | `agentic_cfg.num_iters` | 通过配置控制迭代轮数，默认采用论文建议值 `T=2`。 |
| 最小复现配置入口 | `configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py` | `model.head.dcc_cfg.agentic_cfg` | 在 baseline `rtmo-m` 上最小增量启用 Agentic 模块。 |

## 2. 核心改造思路（最小、稳健）

1. **不改 backbone/neck 主干**：只在 `DCC` 内部新增可开关迭代逻辑，降低对原有训练/推理流程影响。  
2. **增量接入**：通过 `agentic_cfg.enabled` 控制开关，关闭时可回退到原 RTMO 行为。  
3. **API 优先**：将 Critic/Actor/Loop 抽为独立模块，便于后续替换实现细节，不绑定某一版实验参数。  
4. **保持原损失主路径**：当前先复用 RTMO 主损失，确保训练稳定和可落地；后续可再扩展结构监督。  

## 3. 关键代码路径导读

- `mmpose/models/heads/hybrid_heads/agentic_modules.py`
  - `StructuralCritic.forward(pose_preds, joint_scores)`
  - `FeatureRefinementActor.forward(kpt_feats, error_prob, disp_hint)`
  - `ThinkCritiqueActLoop.forward(...)`

- `mmpose/models/heads/hybrid_heads/rtmo_head.py`
  - `DCC.__init__(..., agentic_cfg=None, ...)`
  - `DCC._pose_feats_to_kpt_feats(...)`
  - `DCC._kpt_feats_to_heatmaps(...)`
  - `DCC.forward_train(...)`（含迭代）
  - `DCC.forward_test(...)`（含迭代）

- `configs/body_2d_keypoint/rtmo/coco/agentic-rtmo-m_16xb16-600e_coco-640x640.py`
  - 开启 `agentic_cfg`，并设定默认 `T=2`。

## 4. 当前实现与论文的对应边界

- **已对齐**：模块命名、迭代机制、在 DCC latent 上修正、`T=2` 默认。  
- **简化实现**：Critic 用轻量 MLP + 图邻接传播近似；Actor 为轻量残差 MLP。  
- **暂未全量落地**：`Lerr/Ldisp`、课程权重与 teacher forcing 目前未强制接入主训练损失。  

## 5. 后续扩展建议（供接手者）

1. 在 `RTMOHead.loss` 中增加可开关结构监督分支（默认关闭）。  
2. 将 `disp_hint` 的监督信号与 GT 关键点偏移对齐，统一与 `vis_targets` 联动。  
3. 增加 ablation 配置（`T=0/1/2/3`）和统一日志字段，便于复核论文结论。  
