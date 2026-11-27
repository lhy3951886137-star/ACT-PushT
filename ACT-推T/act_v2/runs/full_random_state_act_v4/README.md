# 全随机 360° · State-ACT v4

**使用场景：** 全随机初始位置 + 全随机360°角度；适合可获取推杆、T块与目标状态的仿真或状态感知系统。

**使用算法：** ACT: full28 continuous branch-cut-free multi-scale SE(2) geometric features + Pre-LN DETR; aperiodic chunking with MC-dropout freeze wake。此检查点 `use_vae=false`，全部动作来自单个ACT模型的动作块。

**历史结果：** Dev 48/50=96%；独立测试1 47/50=94%；独立测试2 90/100=90%；独立测试3 91/100=91%；含50个Dev的300场景总计276/300=92%；独立测试250场景合计228/250=91.2%。

## 权重与配置

- 指定冻结权重：`latest.pt`；SHA256：`af5b6ad11447a5f1b933e93cbf0e71834a710d3824667ad9dd7ad4f624ada575`。
- [原始权重下载](https://github.com/lhy3951886137-star/ACT-PushT/releases/download/act-pusht-v1/full_random_state_act_v4--latest.pt)（公开附件，可直接下载）。
- [原训练配置](config.json)、[归一化统计](stats.json)、[推理执行参数](inference-settings.json)、[原始主评测报告](final_eval_aperiodic/final_300_gate.json)。
- 本次上传也保留该运行的其他best/latest权重与实验检查点，见根目录附件清单；推荐结果仅对应上述指定权重与执行参数。

## 下载与运行

在 `ACT-推T` 根目录执行：

```powershell
python download_assets.py --preset primary
python reproduce.py eval full-state --split blind

# 先下载对应训练数据，再使用新名称重新训练
python download_assets.py --preset train-full-state
python reproduce.py train full-state --dry-run
python reproduce.py train full-state
```

推理参数：

```json
{
  "weight_kind": "ema",
  "n_action_steps": 12,
  "action_schedule": [
    12,
    16,
    18,
    14
  ],
  "mc_dropout_on_freeze": true
}
```

## 单模型加载

```python
from pathlib import Path
from act_v2.full_random_state import FullRandomStateACTPolicy as Policy
policy = Policy(checkpoint_path=Path("act_v2/runs/full_random_state_act_v4/latest.pt"), weight_kind="ema", **{'n_action_steps': 12, 'action_schedule': [12, 16, 18, 14], 'mc_dropout_on_freeze': True})
policy.reset(seed=0)
action = policy.action(env.state_vector())
env.move(action)
```

模型加载、前向推理与评测需要CUDA。完整安装、相机/物理环境和数据说明见 [总说明](../../../README.md)。
