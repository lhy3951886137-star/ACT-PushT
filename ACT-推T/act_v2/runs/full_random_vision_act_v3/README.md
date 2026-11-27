# 全随机 360° · 纯视觉双相机 Vision-ACT v3

**使用场景：** 全随机初始位置 + 全随机360°角度；策略仅输入双相机RGB与机器人XY。

**使用算法：** ACT: dual from-scratch GroupNorm ResNet18 + multi-scale SpatialSoftmax keypoints + Pre-LN DETR; training-only SE(2) auxiliary geometry; aperiodic chunking with MC-dropout freeze wake。此检查点 `use_vae=false`，全部动作来自单个ACT模型的动作块。

**历史结果：** Dev 46/50=92%；盲测1 46/50=92%；盲测2 46/50=92%；盲测合并92/100；图像置零0/20；盲测成功平均1195.6步。

## 权重与配置

- 指定冻结权重：`best.pt`；SHA256：`14baebb6f30200e029c86f887a40e004055017a7aedf4bba4ffb5985fa6a0a3c`。
- [原始权重下载](https://github.com/lhy3951886137-star/ACT-PushT/releases/download/act-pusht-v1/full_random_vision_act_v3--best.pt)（私有附件，需要账号读取权限）。
- [原训练配置](config.json)、[归一化统计](stats.json)、[推理执行参数](inference-settings.json)、[原始主评测报告](final_eval_ap14181620/final_vision_ap_gate.json)。
- 本次上传也保留该运行的其他best/latest权重与实验检查点，见根目录附件清单；推荐结果仅对应上述指定权重与执行参数。

## 下载与运行

在 `ACT-推T` 根目录执行：

```powershell
python download_assets.py --preset primary
python reproduce.py eval full-vision --split blind

# 下载已上传元数据/热启动后，需从本地补齐原始RGB图像，再使用新名称重新训练
python download_assets.py --preset train-full-vision
python reproduce.py train full-vision --dry-run
python reproduce.py train full-vision
```

推理参数：

```json
{
  "weight_kind": "ema",
  "n_action_steps": 14,
  "action_schedule": [
    14,
    18,
    16,
    20
  ],
  "mc_dropout_on_freeze": true
}
```

## 单模型加载

```python
from pathlib import Path
from act_v2.full_random_vision import FullVisionACTPolicy as Policy
policy = Policy(checkpoint_path=Path("act_v2/runs/full_random_vision_act_v3/best.pt"), weight_kind="ema", **{'n_action_steps': 14, 'action_schedule': [14, 18, 16, 20], 'mc_dropout_on_freeze': True})
policy.reset(seed=0)
# obs = observe_full(env)（全随机）或 vision_env.observe(env)（小范围）
action = policy.action(obs["robot_xy"], obs["images"])
env.move(action)
```

模型加载、前向推理与评测需要CUDA。完整安装、相机/物理环境和数据说明见 [总说明](../../../README.md)。

原始RGB训练图像按用户要求保留本地，不随GitHub附件提供。直接运行冻结模型不需要这些训练图像。
