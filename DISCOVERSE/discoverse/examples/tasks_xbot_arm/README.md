# XBot机械臂在DISCOVERSE框架中的使用说明

本文档介绍如何在DISCOVERSE框架中使用XBot机械臂进行仿真和训练。

## 环境结构

XBot机械臂在DISCOVERSE框架中的实现包含以下核心组件：

1. **基础环境** (`robots_env/xbot_arm_base.py`)
   - 实现机械臂的基本物理仿真
   - 提供关节控制、状态观测等功能

2. **任务基础环境** (`task_base/xbot_arm_task_base.py`)
   - 扩展基础环境，添加任务相关功能
   - 实现领域随机化、成功检测等

3. **抓取放置任务** (`examples/tasks_xbot_arm/xbot_arm_pick_place.py`)
   - 特定任务的实现
   - 定义奖励函数、终止条件等

4. **训练配置** (`policies/dp/configs/xbot_arm_config.py`)
   - 扩散策略训练的配置参数

5. **训练脚本** (`policies/dp/train_xbot_arm.py`)
   - 实现模型训练逻辑

6. **测试脚本** (`examples/test_xbot_arm_env.py`)
   - 用于测试环境功能

7. **启动脚本** (`scripts/run_xbot_arm_training.py`)
   - 简化训练过程的启动

## 快速开始

### 1. 测试环境

首先，可以运行测试脚本验证环境是否正常工作：

```bash
# 进入DISCOVERSE目录
cd DISCOVERSE

# 运行测试脚本
python examples/test_xbot_arm_env.py
```

测试脚本将依次测试：
- 基础环境功能（关节控制、TCP位置获取）
- 任务基础环境（领域随机化）
- 抓取放置任务（完整交互测试）

### 2. 运行训练

使用启动脚本运行训练过程：

```bash
# 进入DISCOVERSE目录
cd DISCOVERSE

# 运行训练（使用默认配置）
python scripts/run_xbot_arm_training.py

# 或使用自定义配置
python scripts/run_xbot_arm_training.py --config policies/dp/configs/xbot_arm_config.py --output outputs/my_xbot_training

# 在无头模式下运行（服务器环境）
python scripts/run_xbot_arm_training.py --headless
```

## 环境使用详解

### 基础环境 (`XbotArmBase`)

```python
from robots_env.xbot_arm_base import XbotArmBase

# 创建环境实例
env = XbotArmBase(headless=False)

# 重置环境
obs = env.reset()

# 执行动作
action = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0])  # 6个关节的位置
obs, pri_obs, rew, ter, info = env.step(action)

# 获取TCP位置和姿态
tcp_pos = env.get_tcp_pos()
tcp_quat = env.get_tcp_quat()

# 关闭环境
env.close()
```

### 抓取放置任务 (`XbotArmPickPlace`)

```python
from examples.tasks_xbot_arm.xbot_arm_pick_place import XbotArmPickPlace

# 创建任务实例
env = XbotArmPickPlace(headless=False)

# 重置环境
env.reset()

# 获取物体和目标位置
object_pos = env.get_object_pos()
goal_pos = env.get_goal_pos()

# 执行动作
action = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0])
obs, pri_obs, rew, ter, info = env.step(action)

# 检查是否成功
success = info.get('success', False)

# 关闭环境
env.close()
```

## 自定义配置

### 修改训练配置

编辑 `policies/dp/configs/xbot_arm_config.py` 文件可以自定义训练参数：

- **环境配置**: 修改环境模块、类名和参数
- **数据集配置**: 调整数据收集参数
- **模型配置**: 更改网络结构和参数
- **训练配置**: 设置学习率、批次大小、训练轮数等

### 添加新任务

要添加新任务，可以继承 `XbotArmTaskBase` 类：

```python
class MyCustomTask(XbotArmTaskBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化自定义任务参数
    
    def resetState(self):
        # 重置环境到初始状态
        super().resetState()
        # 添加自定义重置逻辑
    
    def compute_reward(self, obs, action):
        # 计算奖励
        # ...
        return reward
    
    def check_success(self):
        # 检查任务是否成功
        # ...
        return success
```

## 键盘交互控制

在图形界面模式下，可以使用以下键盘控制：

- **WASD**: 移动相机
- **上下左右箭头**: 旋转相机
- **R**: 重置环境
- **ESC**: 退出程序

## 常见问题

1. **环境无法启动**
   - 确保已安装所有依赖
   - 检查MUJOCO_PATH环境变量是否正确设置

2. **训练速度慢**
   - 尝试在GPU上运行 (`--device cuda`)
   - 减小批次大小或图像分辨率

3. **仿真不稳定**
   - 检查关节限制是否合理
   - 尝试调整时间步长

## 依赖要求

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Mujoco
- tqdm

## 注意事项

- 首次运行时，环境会自动生成和加载所需的MJCF文件
- 训练过程中会自动收集数据并保存到指定的输出目录
- 训练日志和检查点会定期保存，可用于恢复训练或评估