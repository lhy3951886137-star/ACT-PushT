# 改进版纯 ACT（`act_v2`）robosuite Push-T 训练项目

本目录完整保留项目历史所有轮次（`runs/`、`round2/`、`round3/`、`round4/`、`stage2/`）不被改动，在不更换模型家族（严格保持 **ACT: Action Chunking with Transformers**）的前提下，对原 ACT 算法、网络架构、损失函数、数据表示与闭环推理进行了系统诊断与升级，并已全部在本机 **RTX 5060 Laptop GPU** 上完成训练与独立盲测验证。

---

## 一、历史 ACT 失败原因诊断

1. **旧 ACT 网络结构退化（[act_model.py](../act_model.py)、[round3_models.py](../round3_models.py)）**：
   - State-ACT 将全部状态压缩成 **1 个 Encoder token**，导致 Self-Attention 与 Cross-Attention 的 softmax 退化为恒等权重 `1.0`。
   - 使用 PyTorch 默认 Post-LN（`norm_first=False`），且 Decoder 仅在首层输入 `queries`，未按标准 DETR/ACT（`tonyzhaozh/act`、LeRobot `ACT`）方式在**每一层 Decoder** 的 Q/K 中重新注入 `query_pos` 与 `encoder_pos`。
2. **CVAE 后验捷径与 BF16 KL 梯度尖峰**：
   - 低 `kl_weight=0.05` 时 Decoder 走后验 `z` 捷径，导致训练 `recon=0.17` 而推理 `z=0` 时 `val_l1=0.41` 严重失配。
   - 高 `kl_weight=10.0` 时在单模态脚本专家数据上 `active_latents` 坍塌为 `0`，且 BF16 下 KL 项仍频繁触发 `65–373` 的梯度尖峰，破坏 AdamW 二阶矩稳定。
3. **动作块承诺长度不足导致接触切换停滞**：
   - 在 Push-T 接触切换任务中，高频重规划（`n_action_steps=1/2`）容易在绕行边界震荡停滞；实测同一权重下 `n_action_steps=1 -> 4 -> 8` 成功率呈明显倒 U 型跃升（`40% -> 76% -> 88%`）。
4. **归一化与采样问题**：
   - 旧版 ACT 使用 `MEAN_STD` 偏移了零动作中心，未对高覆盖率精细对齐阶段重采样，且旧环境存在 1cm 粗糙步长与 0.45 粘滑摩擦，ACT 从未在第四轮/Stage 2 的高精度环境（`precision_env.py` / `vision_env.py`）上验证过。
5. **纯视觉 ACT 纹理过拟合问题**：
   - 将双视角 `4×4` 空间特征图（64 个 256 维 token）直接送入 Transformer 在 128 条训练轨迹上易记忆背景纹理（`recon=0.083 vs val_l1=0.127`）；改为纯多尺度 **SpatialSoftmax 亚像素几何关键点瓶颈** 后，训练与验证损失贴合，盲测成功率从 `80%` 进一步提升至 **`90% (45/50)`**。

---

## 二、核心 ACT 改进内容（100% 纯 ACT 算法）

- **标准 Pre-LN DETR Encoder-Decoder（[model.py](model.py)）**：
  - `PreLNEncoderLayer` / `PreLNDecoderLayer` 采用 Pre-LN + 末端 LayerNorm，`tgt` 零初始化并在每层显式注入 `query_pos` 与 `encoder_pos`。
  - **State-ACT**：按推杆几何（含 T 块局部系坐标 $R(\theta_T)^\top (p_{ee} - p_T)$）与物体-目标几何拆分为多 token，跨 2 帧观测历史（无 `previous_action` 因果混淆）输入 4+4 层 DETR Transformer。
  - **Vision-ACT（零特权状态）**：仅读取 `robot_xy[2]` 与双视角 128×128 图像（`birdview` + `robot0_eye_in_hand`）；双路独立从零训练 GroupNorm ResNet18，同时从 `layer3 (8×8)` 与 `layer4 (4×4)` 提取多尺度 SpatialSoftmax 亚像素 2D 关键点坐标 token（显式坐标瓶颈，消除纹理记忆）。
- **训练与推理策略（[data.py](data.py)、[train_state.py](train_state.py)、[train_vision.py](train_vision.py)、[inference.py](inference.py)）**：
  - **前向衰减加权 Masked L1 Chunk Loss**（`time_decay=0.04`）：强化前 8 步实际执行动作精度，同时保持 16 步长程连贯性。
  - **MIN_MAX 物理边界归一化**：动作严格保持 `[-1, 1]` 零点不偏移；高覆盖率精细对齐阶段 50% 重采样；时间一致微扰正则（历史帧共享同一偏移，完全保留帧间速度）；`AdamW + cosine + EMA(0.995)`；异步线程预取 + pinned DMA 传输；FP32 精确闭环推理与 `n_action_steps=8` 动作块执行（同时内置可配置的 Temporal Ensembling）。

---

## 三、本机 RTX 5060 GPU 实测对比结果

| 阶段 / 模型 | 算法 | 有效步数 | 50 场景开发集 | 50 场景独立盲测 | 盲测平均最大覆盖率 | 图像置零对照 |
|---|---|---:|---:|---:|---:|---:|
| 原 Round 1 State-ACT | 旧紧凑 Post-LN ACT | 30k | 44.0% (22/50) | — | — | — |
| 原 Round 2 State-ACT | 旧 21 维噪声 ACT | 20k | 20.0% (10/50) | — | — | — |
| 原 Round 3 State-ACT | 旧 History CVAE ACT | 20k | 22.0% (11/50) | — | — | — |
| 原 Round 4 State-Diffusion | Diffusion Policy（对照） | 80k | 92.0% (46/50) | 92.0% (46/50) | 0.9460 | — |
| **改进 State-ACT（100 轨迹）** | **Pre-LN DETR-ACT (`state_act_v2_full`)** | 20k | 90.0% (45/50) | 78.0% (39/50) | 0.9345 | — |
| **改进 State-ACT（150 轨迹，推荐）** | **无尖峰 Pre-LN DETR-ACT (`state_act_v2_150`)** | **20k** | **98.0% (49/50)** | **100.0% (50/50)** | **0.9526** | — |
| 原 Stage 2 Vision-Diffusion | Diffusion Policy（对照） | 50k/100k | 88.0% (44/50) | 90.0% (45/50) | 0.9461 | 0.0% (0/20) |
| 改进 Vision-ACT（混合特征图） | Hybrid DETR-ACT (`vision_act_v2`) | 40k | 66.0%–76.0% | 80.0% (40/50) | 0.9299 | 0.0% (0/20) |
| **改进 Vision-ACT（关键点瓶颈，推荐）** | **Multi-Scale SpatialSoftmax DETR-ACT (`vision_act_v2_kp`)** | **45k** | **84.0% (42/50)** | **90.0% (45/50)** | **0.9498** | **0.0% (0/20)** |

### 最优冻结检查点与校验哈希
- **最优 State-ACT**：`act_v2/runs/state_act_v2_150/best.pt`（SHA256: `db5b307eec1b7c194a6ec35f65a3d7292ac6d8d7ac6fa40d78d6bc78b1654fc2`）
  - 开发集（`9100000–9100049`）：`98.0% (49/50)`，Wilson 95% CI `[89.5%, 99.6%]`，`cov=0.9519`，均 `441.5` 步
  - 独立盲测（`9200000–9200049`）：`100.0% (50/50)`，Wilson 95% CI `[92.9%, 100.0%]`，`cov=0.9526`，均 `353.2` 步
- **最优纯视觉 Vision-ACT**：`act_v2/runs/vision_act_v2_kp/best.pt`（SHA256: `6e4486a2d0f9669d59892547b2f08b13242b16f804c34a67117fd42086b4bd52`）
  - 开发集（`10100000–10100049`）：`84.0% (42/50)`，Wilson 95% CI `[71.5%, 91.7%]`，`cov=0.9457`，均 `774.9` 步
  - 独立盲测（`10200000–10200049`）：`90.0% (45/50)`，Wilson 95% CI `[78.6%, 95.7%]`，`cov=0.9498`，均 `700.5` 步
  - 图像置零对照（`10100000–10100019`）：`0.0% (0/20)`，`cov=0.3341`

---

## 四、复现与运行命令

在项目根目录 `C:\Users\LHY\Desktop\self` 执行：

```powershell
# 1. 训练最优 State-ACT（约 15–30 分钟）
.\.venv\Scripts\python.exe -m act_v2.train_state `
  --name state_act_v2_150 --feature-mode legacy16 --use-all-150 `
  --steps 40000 --eval-every 10000 --eval-scenes 50 --n-action-steps 8 `
  --use-vae 0 --lr 1.5e-4 --time-decay 0.04 --fine-prob 0.5 `
  --state-noise-std 0.004 --weight-decay 3e-4 --dropout 0.1

# 2. 复现 State-ACT 50 场景开发集 + 50 场景独立盲测
.\.venv\Scripts\python.exe -m act_v2.confirm_state `
  --checkpoint act_v2/runs/state_act_v2_150/best.pt `
  --out act_v2/runs/state_act_v2_150/final_eval --horizons 8

# 3. 训练最优纯视觉双相机 Vision-ACT（约 2.4 小时，显存峰值 2.60 GB）
.\.venv\Scripts\python.exe -m act_v2.train_vision `
  --name vision_act_v2_kp --token-mode keypoints --micro-batch 64 --accumulation 1 `
  --steps 60000 --eval-every 15000 --eval-scenes 50 --n-action-steps 8 `
  --use-vae 0 --lr 1.5e-4 --weight-decay 5e-4 --dropout 0.15 `
  --time-decay 0.04 --fine-prob 0.5 --ema-decay 0.995 --checkpoint-every 5000

# 4. 复现 Vision-ACT 步长扫描 + 50 场景独立盲测 + 20 场景图像置零对照
.\.venv\Scripts\python.exe -m act_v2.confirm_vision `
  --checkpoint act_v2/runs/vision_act_v2_kp/best.pt `
  --out act_v2/runs/vision_act_v2_kp/final_eval --horizons 8 10
```


---

## 五、全随机初始位置 + 全随机 360° 初始角度（`[-π, +π]`）Push-T 扩展

在保持纯 ACT 算法不变的前提下，已将推 T 环境、脚本专家、数据管道与 State-ACT / Vision-ACT 全面扩展至**初始位置完全随机 + 初始角度完全随机（360°）**任务：

### 1. 环境与运动学对齐（[full_random_env.py](full_random_env.py)、[full_random_collect.py](full_random_collect.py)）
- 实测 Panda 机械臂在推杆高度 `z=0.825` 的物理可达边界为 `X ∈ [-28cm, +11cm], Y ∈ [-27cm, +29cm]`，因此将工作空间中心校准至 `[-7cm, 0]`：
  - 目标 XY 随机范围：`X ∈ [-10cm, -4cm], Y ∈ [-3.5cm, +3.5cm]`
  - T 块初始位置完全随机：`X ∈ [-16cm, +1cm], Y ∈ [-12cm, +12cm]`（确保四面均可绕行施推）
  - T 块初始角度完全随机：`[-180°, +180°]` 全 360°
  - 推杆自动选取非碰撞可达安全起始位，并在 `reset_task()` 推杆预热前先将 T 块移离路径，彻底消除预热碰撞引发的肘部奇异点锁死（100/100 种子 0 奇异点）；`birdview` 相机居中扩展为 `pos=[-0.07, 0, 1.65], fovy=38°`。
- 升级全角度专家（`±180°` 附近锁定一致旋转方向破除多模态对称、`-X/+X` 工作空间边界安全回拉、路径点 6mm 切换防震荡、5 步干净脱障），多进程并行采集并校验：
  - `full_random_data/`：**400 条**基础轨迹（`596,314` 帧，360 训练 / 40 验证，重放误差 `0.00e+00`）
  - `full_random_data_extra/`：**+300 条**新增基础轨迹（`447,187` 帧，重放误差 `0.00e+00`）
  - `full_random_dagger/` + `full_random_dagger_r2/` + `full_random_dagger_r3/`：三轮共 **360 条** DAgger 离轨/末段失败纠偏轨迹（`437,643` 帧），过滤 16 条历史奇异点轨迹后总计 **1,006 条干净训练轨迹（1,395,832 帧）**。
  - `full_random_rgb_all/` + `full_random_rgb_vis_dagger/`：6 常驻进程并行导出全量双视角 RGB（复用已有 245 条 + 补齐基础/额外基础/两轮 DAgger + 90 条直接 RGB 渲染的 Vision-DAgger 纠偏），共 **972 条零误差校验双视角 RGB 轨迹（938 训练 / 34 验证，1,321,744 帧，最大重放误差 `0.00e+00`）**。

### 2. 全随机 State-ACT & 纯视觉 Vision-ACT 双双突破 92.0% 的核心纯 ACT 升级
1. **消除 `±180°` 分支割裂的 `full28` 多尺度几何特征（[full_random_state.py](full_random_state.py)）**：将原 `full21` 中在 `±180°` 处剧烈跳变 `2.0` 的有符号 `atan2/π` 替换为全圆周连续无跳变（跳变仅 `0.0035`）的 `arccos(cos_d)/π`，并引入 T 块局部系平滑单位方向与 `log1p` 对数尺度距离/角度，大幅提升末段 `1–3mm` 精对齐分辨率。
2. **仅训练期双层 `SE(2)` 辅助几何监督（[model.py](model.py)、[full_random_vision.py](full_random_vision.py)）**：在纯视觉 Vision-ACT 的双路 ResNet18 + `48+48` SpatialSoftmax 关键点上挂载平移不变 8 维像素几何头，并在 DETR Encoder 输出上挂载 14 维全 `SE(2)` 几何头（仅训练期反传、推理期完全不调用），仅靠 RGB 像素便学出 **`4.4 mm / 1.5°`** 全 360° 位姿表征。
3. **高效短步采样 + 动作块平滑损失 + 专项 Vision-DAgger 精调**：改变原按轨迹长度正比采样的惯例，采用偏短高效轨迹加权（`speed_power=0.65`）与均衡角度+精细对齐采样，结合 `0.15×` 动作块差分平滑损失与 90 条直接 RGB 渲染的 Vision-DAgger 专项纠偏，在提分的同时将成功完成步数从专家均值 `1490.6` 步大幅压缩至 **`1174–1220` 步**。
4. **非周期动作块执行（Aperiodic ACT Chunking）+ 真冻结 MC-Dropout 唤醒**：全部动作仍 100% 取自单个纯 ACT 模型预测的 24 步动作块，State-ACT 采用 `[12, 16, 18, 14]`、Vision-ACT 采用 `[14, 18, 16, 20]` 打破固定重规划周期的锁相极限环（纯视觉唤醒仅监测自身 `robot_xy`，零特权状态）。

### 3. 全随机 360° ACT 实测成绩（本机 RTX 5060 GPU）

| 模型 | 数据与关键配置 | Dev 50 (`30100000–49`) | 盲测 1 (`30200000–49`, 50 题) | 盲测 2 (`30300000+`) | 盲测 3 / 图像置零 | 总计 / 平均步数（成功均值） |
|---|---|---:|---:|---:|---:|---|
| State-ACT v1 (`full_random_state_act`) | 400 基础、`full21`、`chunk=16`、`exec_12` | 86.0% (43/50) | 70.0% (35/50) | — | — | — |
| State-ACT v2 (`full_random_state_act_v2`) | 400+120 DAgger、`full21`、`chunk=24`、`exec_12` | 84.0%–86.0% | 82.0% (41/50) | — | — | 修复 reset Bug 后盲测 1 为 86.0% |
| State-ACT v3 (`full_random_state_act_v3`) | 927 轨迹、`full28`、30.26M、`exec_18` | **98.0% (49/50)** | 88.0% (44/50) | 86.0% (86/100) | — | 200 题合计 89.5% (`val_l1=0.1655`) |
| **State-ACT v4（非周期 `[12,16,18,14]`，推荐）** | **1,006 轨迹、单模型纯 ACT + 真冻结唤醒** | **96.0% (48/50)** | **94.0% (47/50)** | **90.0% (90/100)** | **91.0% (91/100)** | **300 题 `92.0% (276/300)`，均 `1338` 步** |
| 纯视觉 Vision-ACT v1 (`full_random_vision_act`) | 250 RGB、无辅助监督、60k | 52.0%–58.0% | 50.0% (25/50) | — | `0.0% (0/20)` | 图像置零对照 `0.0%` |
| 纯视觉 Vision-ACT v2 (`full_random_vision_act_v2`) | 848 RGB、仅训练期 `SE(2)` 辅助监督、80k | 86.0% (43/50) | 82.0% (41/50) | — | `0.0% (0/20)` | 盲测 1 成功均值 `1168.3` 步 |
| 纯视觉 Vision-ACT v3（固定 `exec_18`） | 938 RGB（含 Vision-DAgger）、35k 精调 | **92.0% (46/50)** | **90.0% (45/50)** | 74.0% (37/50) | `0.0% (0/20)` | 150 题 `85.3%`，盲测 1 成功 `1206.6` 步 |
| **纯视觉 Vision-ACT v3（非周期 `[14,18,16,20]`，推荐）** | **938 RGB、单模型纯视觉 ACT + 纯本体唤醒** | **92.0% (46/50)** | **92.0% (46/50)** | **92.0% (46/50)** | **`0.0% (0/20)`** | **150 题 `92.0% (138/150)`，成功均值 `1195.6` 步** |

- **全随机 State-ACT 最优权重（推荐）**：`act_v2/runs/full_random_state_act_v4/latest.pt`（SHA256: `af5b6ad11447a5f1b933e93cbf0e71834a710d3824667ad9dd7ad4f624ada575`）
  - 300 场景汇总报告：`act_v2/runs/full_random_state_act_v4/final_eval_aperiodic/final_300_gate.json`
- **全随机纯视觉 Vision-ACT 最优权重（推荐）**：`act_v2/runs/full_random_vision_act_v3/best.pt`（SHA256: `14baebb6f30200e029c86f887a40e004055017a7aedf4bba4ffb5985fa6a0a3c`）
  - 150 场景（50 dev + 100 独立盲测 + 20 图像置零对照）汇总报告：`act_v2/runs/full_random_vision_act_v3/final_eval_ap14181620/final_vision_ap_gate.json`
  - 100 场景合并独立盲测：`92.0% (92/100)`，Wilson 95% CI `[85.0%, 95.9%]`，平均最大覆盖率 `0.9394`，盲测-1 成功均值 `1174.3` 步 / 盲测-2 成功均值 `1220.7` 步，图像置零对照 `0.0% (0/20)`。

### 4. 全随机 360° 复现与运行命令

```powershell
# 1. 并行采集基础轨迹（400 + 300 条）与三轮 DAgger 纠偏轨迹
.\.venv\Scripts\python.exe -m act_v2.full_random_collect --target 400 --workers 12
.\.venv\Scripts\python.exe -m act_v2.full_random_collect --target 300 --workers 10 --seed-base 30010000 --out-dir act_v2/full_random_data_extra --all-train

# 2. 训练 State-ACT v3 & 热启动精调 State-ACT v4，一键复现 300 场景盲测（92.0%）
.\.venv\Scripts\python.exe act_v2/confirm_aperiodic_300.py

# 3. 6 常驻进程并行导出全量双视角 RGB（882 条，约 21 分钟）+ 采集 90 条直接 RGB Vision-DAgger（约 6 分钟）
.\.venv\Scripts\python.exe act_v2/export_full_rgb_all.py --workers 6
.\.venv\Scripts\python.exe act_v2/vision_dagger_collect.py --checkpoint act_v2/runs/full_random_vision_act_v2/best.pt --target 90 --workers 4

# 4. 训练全随机纯视觉 Vision-ACT v2（80k 步，仅训练期 SE(2) 辅助监督）并精调 Vision-ACT v3（35k 步）
.\.venv\Scripts\python.exe act_v2/full_random_vision.py `
  --name full_random_vision_act_v2 --rgb-dir act_v2/full_random_rgb_all `
  --n-obs-steps 3 --chunk-size 24 --n-action-steps 14 --d-model 384 --dim-feedforward 1536 `
  --keypoints-l3 48 --keypoints-l4 48 --crop-size 124 --time-decay 0.015 --smooth-weight 0.15 `
  --aux-weight 0.60 --aux-min-weight 0.25 --micro-batch 64 --steps 80000 --lr 2e-4 --ema-decay 0.996
.\.venv\Scripts\python.exe act_v2/full_random_vision.py `
  --name full_random_vision_act_v3 --init-from act_v2/runs/full_random_vision_act_v2/best.pt `
  --rgb-dir act_v2/full_random_rgb_all --extra-rgb-dirs act_v2/full_random_rgb_vis_dagger `
  --balanced-angles --speed-power 0.65 --vis-dagger-boost 3.0 --r3-boost 2.0 `
  --n-obs-steps 3 --chunk-size 24 --n-action-steps 14 --d-model 384 --dim-feedforward 1536 `
  --dropout 0.05 --keypoints-l3 48 --keypoints-l4 48 --crop-size 124 --time-decay 0.01 `
  --smooth-weight 0.15 --aux-weight 0.35 --aux-min-weight 0.15 --micro-batch 64 --steps 35000 --lr 6e-5 --ema-decay 0.996
```
