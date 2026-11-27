# 小范围高精度 · State-ACT（150轨迹）

**使用场景：** 小范围平移、近目标角度的高精度Push-T；需要机器人、T块、目标状态；不适用于未经验证的全360°初始条件。

**使用算法：** ACT: legacy16 geometry with two observation frames + Pre-LN DETR; weighted masked L1; MIN_MAX normalization; EMA。此检查点 `use_vae=false`，全部动作来自单个ACT模型的动作块。

**历史结果：** Dev 49/50=98%；独立盲测50/50=100%。

## 权重与配置

- 指定冻结权重：`best.pt`；SHA256：`db5b307eec1b7c194a6ec35f65a3d7292ac6d8d7ac6fa40d78d6bc78b1654fc2`。
- [原始权重下载](https://github.com/lhy3951886137-star/ACT-PushT/releases/download/act-pusht-v1/state_act_v2_150--best.pt)（私有附件，需要账号读取权限）。
- [原训练配置](config.json)、[归一化统计](stats.json)、[推理执行参数](inference-settings.json)、[原始主评测报告](final_eval/state_gate.json)。
- 本次上传也保留该运行的其他best/latest权重与实验检查点，见根目录附件清单；推荐结果仅对应上述指定权重与执行参数。

## 下载与运行

在 `ACT-推T` 根目录执行：

```powershell
python download_assets.py --preset primary
python reproduce.py eval small-state --split blind

# 先下载对应训练数据，再使用新名称重新训练
python download_assets.py --preset train-small-state
python reproduce.py train small-state --dry-run
python reproduce.py train small-state
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
from act_v2.inference import StateACTPolicy as Policy
policy = Policy(checkpoint_path=Path("act_v2/runs/state_act_v2_150/best.pt"), weight_kind="ema", **{'n_action_steps': 8})
policy.reset(seed=0)
action = policy.action(env.state_vector())
env.move(action)
```

模型加载、前向推理与评测需要CUDA。完整安装、相机/物理环境和数据说明见 [总说明](../../../README.md)。
