# MiGPO WebShop 实现总结

## 概述

基于 ALFWorld 的 MiGPO 实现，成功扩展到 WebShop 任务。与 ALFWorld 的静态 action 锚点匹配不同，WebShop 使用**动态 milestone 检测**方法来划分任务阶段。

## WebShop 任务阶段划分方法

BEACON 的 WebShop milestone 检测定义了 **4 个顺序里程碑阶段**：

1. **SEARCH 阶段** (segment_id=0)
   - 触发条件：搜索结果页面中至少包含一个满足所有目标约束的产品
   - 检测逻辑：解析搜索结果中的 ASIN，检查产品属性、选项可用性、价格是否符合目标

2. **DETAIL 阶段** (segment_id=1)
   - 触发条件：智能体点击进入了一个符合目标约束的产品详情页
   - 检测逻辑：验证 `click[ASIN]` 动作的产品是否满足目标

3. **OPTIONS 阶段** (segment_id=2)
   - 触发条件：所有目标要求的选项（如颜色、尺寸）都已正确选择
   - 检测逻辑：跟踪 `click[option_value]` 动作，检查是否所有必需选项都已选择

4. **PURCHASE 阶段** (segment_id=3)
   - 触发条件：智能体执行 `click[buy now]` 购买了符合目标的产品
   - 检测逻辑：验证购买时产品和选项是否完全满足目标约束

5. **COMPLETE 阶段** (segment_id=4)
   - 任务完成后的状态

## 关键区别：ALFWorld vs WebShop

| 维度 | ALFWorld | WebShop |
|------|----------|---------|
| **Milestone 定义** | 静态 action 锚点列表（JSON） | 动态状态转换检测 |
| **匹配方法** | 字符串相似度匹配 | 环境状态验证 + 目标约束检查 |
| **数据依赖** | actions, trial_id | actions, states, next_states, goals, infos |
| **复杂度** | 简单（字符串匹配） | 复杂（需要产品数据库、目标解析） |
| **实现模块** | `core_migpo.py` | `core_migpo_webshop.py` + `webshop_milestone_detector.py` |

## 实现文件清单

### 新增文件

1. **gigpo/webshop_milestone_detector.py**
   - 从 BEACON 项目复制
   - 实现动态 milestone 检测逻辑
   - 包含 `MilestoneDetector` 类，处理状态转换和目标验证

2. **gigpo/core_migpo_webshop.py**
   - WebShop 专用的 segment_id 计算函数
   - `compute_webshop_segment_ids_from_info()`: 单条轨迹的 segment 计算
   - `compute_batch_segment_ids_webshop()`: 批量轨迹的 segment 计算

3. **examples/gigpo_trainer/run_webshop_migpo.sh**
   - WebShop 的 MiGPO 训练启动脚本
   - 配置 `algorithm.adv_estimator=migpo`
   - 使用与 ALFWorld 相同的超参数（ablation 模式：只用 GRPO + subgroup 优势）

### 修改文件

1. **agent_system/environments/env_package/webshop/envs.py**
   - 修改 `WebshopWorker.reset()` 和 `WebshopWorker.step()`
   - 在 info dict 中添加：
     - `obs`: 原始观察字符串
     - `goal`: 目标约束字典

2. **agent_system/environments/env_manager.py**
   - 修改 `WebshopEnvironmentManager.step()`
   - 收集并传递给 rollout：
     - `action`: 动作字符串
     - `state_before`: 动作前的观察
     - `state_after`: 动作后的观察

3. **agent_system/multi_turn_rollout/rollout_loop.py**
   - 在 batch 收集时添加 WebShop 特定字段：
     - `states`: 动作前观察
     - `next_states`: 动作后观察
     - `goals`: 目标字典
     - `infos`: 完整的 info 字典

4. **verl/trainer/ppo/ray_trainer.py**
   - 在 `compute_batch_segment_ids` 调用处添加环境检测
   - 如果是 WebShop：
     - 加载产品数据库（首次调用时缓存）
     - 调用 `compute_batch_segment_ids_webshop()`
   - 如果是 ALFWorld：
     - 调用原有的 `compute_batch_segment_ids()`

## 数据流

### 1. 环境数据收集

```
WebshopWorker.step()
  ↓ 返回 info['obs'], info['goal']
WebshopEnvironmentManager.step()
  ↓ 添加 info['action'], info['state_before'], info['state_after']
TrajectoryCollector
  ↓ 收集到 batch.non_tensor_batch['states'], ['next_states'], ['goals'], ['infos']
```

### 2. Segment ID 计算

```
ray_trainer.py (训练循环)
  ↓ 检测到 adv_estimator=migpo 且 env_name=Webshop
加载产品数据库 (product_lookup, price_lookup)
  ↓
compute_batch_segment_ids_webshop()
  ↓ 对每条轨迹
compute_webshop_segment_ids_from_info()
  ↓ 创建 MilestoneDetector
  ↓ 逐 step 调用 detector.process()
  ↓ 根据 milestone phase 分配 segment_id
返回 segment_ids 数组
  ↓ 存入 batch.non_tensor_batch['segment_ids']
```

### 3. 优势计算

```
compute_migpo_outcome_advantage()
  ↓ 使用 segment_ids
build_subgroup_uids()
  ↓ (obs-cluster ∩ segment) 构建子分组
step_norm_reward()
  ↓ 在子分组内归一化
adv = grpo_adv + step_advantage_w * gigpo_adv + subgroup_advantage_w * subgroup_adv
```

## 配置参数

在训练脚本中设置：

```bash
algorithm.adv_estimator=migpo
algorithm.gigpo.step_advantage_w=0          # GiGPO obs-group 优势权重
algorithm.gigpo.subgroup_advantage_w=1.0    # MiGPO subgroup 优势权重
algorithm.gigpo.mode=mean_std_norm          # 归一化方式
```

## 运行训练

```bash
cd /itdd-pfs/wangyuhao/long_horizon/verl-agent
bash examples/gigpo_trainer/run_webshop_migpo.sh
```

## 与 ALFWorld 实现的兼容性

- 两种实现完全独立，互不影响
- `ray_trainer.py` 通过环境名称自动选择正确的 segment 计算方法
- ALFWorld 继续使用 `core_migpo.compute_batch_segment_ids()`
- WebShop 使用 `core_migpo_webshop.compute_batch_segment_ids_webshop()`

## 关键设计决策

1. **为什么不复用 ALFWorld 的实现？**
   - WebShop 的 milestone 检测逻辑完全不同（状态验证 vs 字符串匹配）
   - 需要访问产品数据库和目标约束
   - 分离实现避免代码耦合，便于维护

2. **为什么在 ray_trainer 中缓存产品数据库？**
   - 产品数据库加载开销大
   - 在训练过程中保持不变
   - 首次加载后缓存可显著提升性能

3. **为什么需要 state_before 和 state_after？**
   - MilestoneDetector.process() 需要状态转换信息
   - state_before: 判断当前所处阶段
   - state_after: 检测 milestone 是否达成

## 验证清单

在运行训练前，确保：

- [ ] `gigpo/webshop_milestone_detector.py` 已复制
- [ ] `gigpo/core_migpo_webshop.py` 已创建
- [ ] WebShop 环境返回 `goal` 信息
- [ ] rollout 收集 `states`, `next_states`, `goals`, `infos`
- [ ] ray_trainer 能正确检测 WebShop 环境
- [ ] 产品数据库路径正确（DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH）

## 预期输出

训练启动时应看到：

```
[MiGPO-WebShop] Loading product catalog for milestone detection...
[MiGPO-WebShop] Loaded XXXX products.
```

每个 epoch 的统计信息会显示 segment 划分情况。

## 调试提示

如果遇到问题：

1. **缺少 goal 信息**
   - 检查 `WebshopWorker` 是否正确添加 `info['goal']`
   - 验证环境是否有 `env.goal` 属性

2. **segment_ids 全为 0**
   - 检查 `states`/`goals` 是否正确传递
   - 验证 MilestoneDetector 是否正常工作
   - 查看日志文件 `logs/milestone_debug/`

3. **产品数据库加载失败**
   - 确认 WebShop 数据文件存在
   - 检查路径配置是否正确

## 未来改进

1. **性能优化**
   - 考虑预计算常见 goal 的产品匹配结果
   - 并行化 milestone 检测

2. **日志增强**
   - 添加详细的 segment 统计
   - 可视化不同阶段的成功率

3. **泛化扩展**
   - 统一 ALFWorld 和 WebShop 的接口
   - 支持更多基于状态转换的任务
