# ACT-推T：四个训练完成的 ACT 模型

本目录归档用户指定的四个 robosuite / MuJoCo Push-T 模型。四者均为 **ACT（Action Chunking with Transformers）**，采用 Pre-LN DETR Encoder-Decoder；这四个检查点均关闭 CVAE（`use_vae=false`）。视觉模型的两路 GroupNorm ResNet18 均从零训练，无需另行下载 ImageNet 骨干权重。

## 模型与使用场景

| 模型目录 | 使用场景与策略输入 | 算法与执行参数 | 历史评测结果 |
|---|---|---|---|
| [全随机 Vision-ACT v3](act_v2/runs/full_random_vision_act_v3/README.md) | 全随机初始位置 + 全 360° 角度；双相机 RGB + 机器人 XY；推理不读取 T 块或目标真值 | 双路 ResNet18 + 多尺度 SpatialSoftmax 关键点 + DETR-ACT；训练期 SE(2) 辅助监督；3 帧历史 / 24 步预测；非周期 `[14,18,16,20]` + 冻结唤醒 | Dev 92%；独立盲测 92/100；图像置零 0/20；含Dev的150场景成功平均 1195.6 步 |
| [全随机 State-ACT v4](act_v2/runs/full_random_state_act_v4/README.md) | 全随机初始位置 + 全 360° 角度；适合仿真或可可靠获得机器人、物体和目标状态的场景 | full28 连续 SE(2) 几何特征 + DETR-ACT；3 帧历史 / 24 步预测；非周期 `[12,16,18,14]` + 冻结唤醒 | Dev 48/50；三组独立测试 47/50、90/100、91/100；含 Dev 的 300 场景总计 276/300（92%） |
| [小范围 State-ACT](act_v2/runs/state_act_v2_150/README.md) | 小范围平移、接近目标角度的高精度 Push-T；可获得完整状态 | legacy16 几何特征 + DETR-ACT；2 帧历史 / 16 步预测；每次执行 8 步 | Dev 49/50（98%）；独立盲测 50/50（100%） |
| [小范围 Vision-ACT](act_v2/runs/vision_act_v2_kp/README.md) | 小范围高精度 Push-T；双相机 RGB + 机器人 XY | 双路 ResNet18 + 多尺度 SpatialSoftmax 关键点 + DETR-ACT；2 帧历史 / 16 步预测；每次执行 8 步 | Dev 42/50（84%）；独立盲测 45/50（90%）；图像置零 0/20 |

原始评测 JSON、逐种子结果、训练配置、归一化统计和训练指标均在相应模型目录内。以上为历史冻结权重的结果，非本次上传重新跑出的全量分数。全随机 State 的“300 场景”包含 50 个 Dev 场景，独立测试部分共 250 场景。小范围视觉模型的原 `vision_gate.json` 为 `passed=false`，因为原门槛同时要求 Dev≥85%，其 Dev 为84%；独立盲测90%的记录保持原样。

## 安装

在克隆的仓库中进入本目录（`ACT-推T`），以下命令均以此目录为工作目录。已验证环境为 Windows 11、Python 3.12.10、RTX 5060 Laptop、CUDA 12.8；运行脚本目前使用 CUDA。物理引擎或 GPU 软件版本变化可能影响闭环轨迹，环境与参数见 [environment.json](environment.json)。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux 可改用 `.venv/bin/python`；离屏渲染需要可用的 MuJoCo OpenGL 后端。跨平台完整分数尚未验证。

## 下载权重与数据

仓库为私有仓库，下载时需要已登录且拥有读取权限的 GitHub 账号。脚本优先使用 `GH_TOKEN` / `GITHUB_TOKEN`，其次使用 `gh auth token` 或 Git Credential Manager 中已有的 GitHub 凭据；凭据不写入本仓库。

```powershell
# 仅下载四个指定模型（约1.75GB），可以直接开始闭环评测
python download_assets.py --preset primary

# 本次上传的全部材料：所有 best/latest、两个 soup 实验检查点、训练热启动、状态/视觉元数据与评测轨迹
python download_assets.py --preset all

# 按模型下载已上传的训练元数据和热启动；视觉模型另需本地原始RGB
python download_assets.py --preset train-full-vision
python download_assets.py --preset train-full-state
python download_assets.py --preset train-small-vision
python download_assets.py --preset train-small-state

# 下载前列出体积与文件，不访问网络
python download_assets.py --preset all --list

# 完成恢复后，离线校验已上传文件的SHA256
python download_assets.py --preset all --verify-local
```

大文件均以原始 `.pt` 或无损 `.tar.zst` 保存在 [act-pusht-v1 Release](https://github.com/lhy3951886137-star/ACT-PushT/releases/tag/act-pusht-v1)。本次共13个检查点、7个数据/轨迹附件，约5.24GB。脚本校验每个附件的 SHA256、解包路径和文件大小后恢复原相对路径。`--keep-cache` 可保留压缩附件，默认成功解包后移除本次下载的压缩缓存。

**按用户最终选择，原始 RGB 训练图像不上传 GitHub，仍保留在本地原训练目录。** 上传材料包含状态轨迹、训练/验证划分、视觉动作/辅助状态元数据、原相机配置和采集/重放代码。四个冻结模型均可直接运行和评测，无需原始训练图像；重新训练视觉模型时需要另行提供这些原始 RGB 数组。

## 评测与推理

```powershell
python reproduce.py eval full-vision --split blind
python reproduce.py eval full-vision --split blind2
python reproduce.py eval full-state --split dev
python reproduce.py eval full-state --split blind3
python reproduce.py eval small-state --split blind
python reproduce.py eval small-vision --split blind
python reproduce.py eval small-vision --split zero

# 快速运行一个场景；完整复现时省略 --count
python reproduce.py eval full-vision --split blind --count 1
```

统一入口固定 EMA 权重和上表的执行参数，并创建新的评测输出目录，避免复用原历史评测缓存。种子、输入定义和单模型加载示例见 [models.json](models.json) 与各模型卡。所有动作为二维 `[-1,1]`，通过环境 `move()` 执行；`precision_env.STEP=0.004` 米，控制频率20Hz。请保留随附的物理环境与相机配置，不能直接将动作当作其他机械臂的原始关节控制量。

## 重新训练

```powershell
python reproduce.py train small-state
python reproduce.py train full-state

# 以下视觉训练命令需要先从本地补齐RGB训练图像
python reproduce.py train small-vision
python reproduce.py train full-vision
```

命令读取原 `config.json`，使用新运行名称（`*_retrain`）训练，避免原目录的 `complete.json` 使训练提前返回。全随机 State v4 从随附 v3 的 `latest.pt` 热启动；全随机 Vision v3 从随附 v2 的 `best.pt` 热启动，并补入原 README 记录的角度均衡、轨迹速度加权和 Vision-DAgger 采样参数。可用 `--dry-run` 查看完整命令。原始配置未保存每个采样器选项、硬件非确定性也未完全消除，因此重新训练不承诺逐位相同；原冻结检查点原样保留，可直接复现其推理和评测。

状态模型重新训练所需数据已上传。视觉模型重新训练前，将本地 `stage2/data/*_rgb.npy`、`act_v2/full_random_rgb/*_rgb.npy`、`act_v2/full_random_rgb_all/*_rgb.npy` 和 `act_v2/full_random_rgb_vis_dagger/*_rgb.npy` 按原相对路径补入本目录；各目录中的清单与元数据随附件提供。保留采集与 RGB 重放源码作为参考。历史开发过程与旧命令见 [README-original.md](act_v2/README-original.md)，其中的历史路径和其他模型不属于本次四个推荐入口；以本页和 `reproduce.py` 为准。

## 文件布局与校验

```text
ACT-推T/
  act_v2/runs/<四个模型>/  README、config、stats、训练指标与原评测
  act_v2/*.py             ACT 网络、数据加载、训练、推理、全随机环境
  assets/                推杆和T块 MuJoCo XML
  precision_env.py        小范围高精度物理环境
  vision_env.py           双相机配置
  download_assets.py      私有Release下载、SHA256校验与安全解压
  reproduce.py            四个模型的统一训练与评测入口
  models.json            场景、算法、种子、执行参数、主要权重哈希
  assets-manifest.json    每个权重/数据分卷的大小、路径和SHA256
```

下载后权重回到 `act_v2/runs/<模型>/best.pt` 或 `latest.pt`。本次归档的实际验证结果见 [VALIDATION.md](VALIDATION.md)。只对分发副本中的 `act_v2/data.py` 做了路径可迁移修正，兼容数据清单中原 Windows 绝对路径；原始训练目录未改动，权重和数据内容未改动。没有额外添加开源许可。
