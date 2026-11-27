# 小范围高精度 · 纯视觉双相机 Vision-ACT

**使用场景：** 小范围高精度Push-T；双相机RGB与机器人XY；不读取T块、目标真值作为策略输入。

**使用算法：** ACT: dual from-scratch GroupNorm ResNet18 + multi-scale SpatialSoftmax keypoint bottleneck + Pre-LN DETR; weighted masked L1; EMA。此检查点 `use_vae=false`，全部动作来自单个ACT模型的动作块。

**历史结果：** Dev 42/50=84%；独立盲测45/50=90%；图像置零0/20。原始gate要求Dev和盲测均≥85%，因此passed=false，原报告保持原样。

## 权重与配置

- 指定冻结权重：`best.pt`；SHA256：`6e4486a2d0f9669d59892547b2f08b13242b16f804c34a67117fd42086b4bd52`。
- [原始权重下载](https://github.com/lhy3951886137-star/ACT-PushT/releases/download/act-pusht-v1/vision_act_v2_kp--best.pt)（公开附件，可直接下载）。
- [原训练配置](config.json)、[归一化统计](stats.json)、[推理执行参数](inference-settings.json)、[原始主评测报告](final_eval/vision_gate.json)。
- 本次上传也保留该运行的其他best/latest权重与实验检查点，见根目录附件清单；推荐结果仅对应上述指定权重与执行参数。

## 下载与运行

在 `ACT-推T` 根目录执行：

```powershell
python download_assets.py --preset primary
python reproduce.py eval small-vision --split blind

# 下载已上传元数据/热启动后，需从本地补齐原始RGB图像，再使用新名称重新训练
python download_assets.py --preset train-small-vision
python reproduce.py train small-vision --dry-run
python reproduce.py train small-vision
```

推理参数：

```json
{
  "weight_kind": "ema",
  "n_action_steps": 8
}
```

## 单模型加载

```python
from pathlib import Path
from act_v2.inference import VisionACTPolicy as Policy
policy = Policy(checkpoint_path=Path("act_v2/runs/vision_act_v2_kp/best.pt"), weight_kind="ema", **{'n_action_steps': 8})
policy.reset(seed=0)
# obs = observe_full(env)（全随机）或 vision_env.observe(env)（小范围）
action = policy.action(obs["robot_xy"], obs["images"])
env.move(action)
```

模型加载、前向推理与评测需要CUDA。完整安装、相机/物理环境和数据说明见 [总说明](../../../README.md)。

原始RGB训练图像按用户要求保留本地，不随GitHub附件提供。直接运行冻结模型不需要这些训练图像。
