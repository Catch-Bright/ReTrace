# MiGPO → WiGPO 重命名记录

## 日期
2025-07-04

## 修改内容
仅进行了名称重命名，**代码逻辑完全未改动**。

## 重命名的文件

### 核心代码
- `gigpo/core_migpo.py` → `gigpo/core_wigpo.py`
- `gigpo/core_migpo_webshop.py` → `gigpo/core_wigpo_webshop.py`

### 训练脚本
- `examples/gigpo_trainer/run_webshop_migpo.sh` → `examples/gigpo_trainer/run_webshop_wigpo.sh`
- `examples/gigpo_trainer/run_alfworld_migpo.sh` → `examples/gigpo_trainer/run_alfworld_wigpo.sh`

## 代码中的修改

### 函数名
- `compute_migpo_outcome_advantage()` → `compute_wigpo_outcome_advantage()`

### 枚举值
```python
# Before: AdvantageEstimator.MiGPO = 'migpo'
# After:  AdvantageEstimator.WiGPO = 'wigpo'
```

### 环境变量
- `MIGPO_MILESTONE_PATH` → `WIGPO_MILESTONE_PATH`

## 配置变更

运行训练时需要使用新名称：
```bash
# 旧的配置
algorithm.adv_estimator=migpo

# 新的配置
algorithm.adv_estimator=wigpo
```

## 验证结果
- ✅ 所有 MiGPO/migpo 引用已替换为 WiGPO/wigpo
- ✅ 代码逻辑完全未修改
- ✅ 共47处引用已更新
