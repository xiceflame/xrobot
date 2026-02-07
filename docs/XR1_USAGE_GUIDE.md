# XR-1 模型使用指南

> **文档版本**: v1.0  
> **适用对象**: 天工Pro机器人 (Tiankung Pro)  
> **最后更新**: 2026-02-02

---

## 📋 目录

1. [XR-1 简介](#一-xr-1-简介)
2. [快速开始](#二-快速开始)
3. [模型推理](#三-模型推理)
4. [模型微调](#四-模型微调)
5. [真实机器人部署](#五-真实机器人部署)
6. [故障排查](#六-故障排查)

---

## 一、XR-1 简介

### 1.1 什么是XR-1？

XR-1是**北京人形机器人创新中心**开发的官方VLA（Vision-Language-Action）大模型，专为具身智能设计。

**核心能力**:
- 🎯 将视觉观测 + 语言指令 → 转化为物理动作
- 🦾 支持双臂协调操作
- 🔄 跨机器人本体泛化（天工、Franka、UR5等）

### 1.2 技术架构

```
输入: [图像] + [语言指令] + [机器人状态]
        ↓
   UVMC编码器 (统一视动表征)
        ↓
   自回归Transformer
        ↓
输出: [双臂关节动作]
```

**三阶段训练**:

| 阶段 | 名称 | 功能 | 数据量 | 用途 |
|------|------|------|--------|------|
| Stage 1 | UVMC学习 | 视觉-动作统一表征 | EGO4D + 机器人数据 | 预训练 |
| Stage 2 | 预训练 | 通用操作知识 | RoboMIND 2.0 (30万+轨迹) | 通用技能 |
| Stage 3 | 任务微调 | 特定任务适配 | 50-100条任务轨迹 | **推荐使用** |

### 1.3 输入输出格式

**输入**:
```python
{
    "task": "拿起红色的杯子",           # 语言指令
    "observation.images.image_0": tensor,  # 头部相机图像 [1, 3, H, W]
    "observation.images.image_1": tensor,  # 左腕相机图像 [1, 3, H, W]
    "observation.images.image_2": tensor,  # 右腕相机图像 [1, 3, H, W]
    "observation.state.arm_joint_position": tensor  # 关节状态 [1, 12]
}
```

**输出**:
```python
action_queue  # 动作序列 [action_horizon, 12]
              # 12 = 左臂6关节 + 右臂6关节
```

---

## 二、快速开始

### 2.1 环境准备

```bash
# SSH到DGX Spark
ssh spark

# 激活XR-1环境
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate xr1

# 验证环境
python -c "from lerobot.common.policies.xr1.modeling_xr1_stage2 import Xr1Stage2Policy; print('✅ 环境正常')"
```

### 2.2 检查预训练模型

```bash
# 查看已下载的模型
ls -lh ~/projects/XR-1/pretrained/

# 预期输出:
# XR-1-Stage1-UVMC/  (3.9GB)
# XR-1-Stage2/       (16GB)
```

---

## 三、模型推理

### 3.1 基础推理示例

```python
import torch
import cv2
import numpy as np
from lerobot.common.policies.xr1.modeling_xr1_stage2 import Xr1Stage2Policy

# 1. 加载模型
model_path = "/home/leo/projects/XR-1/pretrained/XR-1-Stage2"
device = "cuda"

policy = Xr1Stage2Policy.from_pretrained(model_path, map_location=device)
policy.eval()

# 2. 准备输入数据
task_name = "拿起红色的杯子"

# 图像输入 (示例使用随机图像，实际应使用相机捕获)
images = {
    "image_0": torch.randn(1, 3, 480, 640).to(device),  # 头部相机
    "image_1": torch.randn(1, 3, 480, 640).to(device),  # 左腕相机
    "image_2": torch.randn(1, 3, 480, 640).to(device),  # 右腕相机
}

# 机器人状态 (12维: 左臂6关节 + 右臂6关节)
state = torch.randn(1, 12).to(device)

# 3. 构建观测字典
observation = {
    "task": [task_name],
    "observation.images.image_0": images["image_0"],
    "observation.images.image_1": images["image_1"],
    "observation.images.image_2": images["image_2"],
    "observation.state.arm_joint_position": state,
}

# 4. 推理
action_horizon = 50  # 预测未来50步动作
with torch.no_grad():
    actions = policy.select_action(observation, action_horizon=action_horizon)

print(f"输出动作形状: {actions.shape}")  # [50, 12]
print(f"动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
```

### 3.2 使用XR1_Evaluation类（推荐）

```python
# 文件: ~/projects/XR-1/deploy/real_robot/xr1_deploy.py

from xr1_deploy import XR1_Evaluation

# 初始化评估器
evaluator = XR1_Evaluation(
    model_path="/home/leo/projects/XR-1/pretrained/XR-1-Stage2",
    robot_type="tiankung",  # 或 "franka"
    action_horizon=50,
    exp_weight=0.05,
    ensemble=True,
)

# 准备观测数据
obs = {
    "images": {
        "head": encoded_head_image,      # 头部相机JPEG编码
        "left_wrist": encoded_left_img,  # 左腕相机JPEG编码
        "right_wrist": encoded_right_img,# 右腕相机JPEG编码
    },
    "arm_joints": {
        "left": left_arm_joints,   # numpy数组 [6]
        "right": right_arm_joints, # numpy数组 [6]
    }
}

# 推理
task_name = "拿起红色的杯子"
actions = evaluator.Inference_Dual_Arm_Tien_Kung2(obs, task_name)

# actions: [action_horizon, 12] 的动作序列
```

### 3.3 图像预处理

```python
def preprocess_image(image_np, target_size=(640, 480)):
    """
    图像预处理流程
    
    Args:
        image_np: numpy数组 [H, W, 3] (BGR格式，OpenCV读取)
        target_size: (width, height)
    
    Returns:
        tensor: [1, 3, H, W] 归一化后的tensor
    """
    # 1. Resize
    image_resized = cv2.resize(image_np, target_size)
    
    # 2. 转换为RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # 3. 归一化到[0, 1]
    image_norm = image_rgb.astype(np.float32) / 255.0
    
    # 4. 转换为tensor [H, W, 3] -> [3, H, W]
    image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)
    
    # 5. 添加batch维度 [1, 3, H, W]
    image_tensor = image_tensor.unsqueeze(0).to("cuda")
    
    return image_tensor
```

---

## 四、模型微调

### 4.1 准备数据集

**数据格式**: LeRobot Dataset v2.1

```bash
# 目录结构
~/projects/XR-1/data/lerobot_tiankung/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.json
├── videos/
│   ├── observation.images.image_0/
│   ├── observation.images.image_1/
│   └── observation.images.image_2/
└── data/
    ├── chunk-000/...
```

### 4.2 执行微调（Stage 3）

```bash
# 进入项目目录
cd ~/projects/XR-1

# 激活环境
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate xr1

# 运行微调脚本
bash scripts/xr1_stage3_finetune.sh \
    --dataset lerobot_tiankung \
    --real
```

**关键参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.stage2_pretrained_path` | `../pretrained/XR-1-Stage2` | Stage2预训练模型路径 |
| `--policy.action_chunk_size` | 50 | 动作预测长度 |
| `--policy.freeze_vision_encoder` | true | 是否冻结视觉编码器 |
| `--policy.freeze_language_encoder` | true | 是否冻结语言编码器 |
| `--policy.optimizer_lr` | 5e-5 | 学习率 |
| `--batch_size` | 20 | 批次大小 |
| `--steps` | 50_000 | 训练步数 |

### 4.3 调试模式

```bash
# 快速调试（不保存模型，用于验证代码）
bash scripts/xr1_stage3_finetune.sh \
    --dataset lerobot_tiankung \
    --debug
```

### 4.4 监控训练

```bash
# 方法1: TensorBoard
tensorboard --logdir=~/projects/XR-1/save_xr1/xr1_stage3/

# 方法2: Weights & Biases (已配置)
# 训练日志会自动上传到 https://wandb.ai
```

---

## 五、真实机器人部署

### 5.1 网络配置

```bash
# 1. 配置DGX Spark网络接口
sudo ip addr add 192.168.41.100/24 dev enp1s0f0np0
sudo ip link set enp1s0f0np0 up

# 2. 测试连接
ping 192.168.41.1  # 机器人IP

# 3. SSH登录机器人
ssh ubuntu@192.168.41.1
```

### 5.2 启动机器人驱动

在**机器人**上执行：

```bash
# 终端1: 启动本体驱动
ros2 launch body_control body.launch.py

# 终端2: 启动运控
ros2 launch motion_control motion.py
```

### 5.3 运行部署脚本

在**DGX Spark**上执行：

```bash
# 激活环境
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate xr1

# 运行部署
cd ~/projects/XR-1
python deploy/real_robot/xr1_deploy.py \
    --config configs/tiankung/tiankung_pro.yaml \
    --checkpoint pretrained/XR-1-Stage2 \
    --robot_type tiankung
```

### 5.4 完整部署代码示例

```python
#!/usr/bin/env python3
"""
XR-1 天工Pro部署示例
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge

from deploy.real_robot.xr1_deploy import XR1_Evaluation

class XR1DeploymentNode(Node):
    def __init__(self):
        super().__init__('xr1_deployment')
        
        # 初始化XR-1模型
        self.evaluator = XR1_Evaluation(
            model_path="/home/leo/projects/XR-1/pretrained/XR-1-Stage2",
            robot_type="tiankung",
            action_horizon=50,
            exp_weight=0.05,
            ensemble=True,
        )
        
        # CV桥接
        self.bridge = CvBridge()
        
        # 订阅相机话题
        self.sub_head = self.create_subscription(
            Image, '/camera/nav/color/image_raw', 
            self.head_callback, 10)
        self.sub_left = self.create_subscription(
            Image, '/camera/left_wrist/color/image_raw',
            self.left_callback, 10)
        self.sub_right = self.create_subscription(
            Image, '/camera/right_wrist/color/image_raw',
            self.right_callback, 10)
        
        # 订阅关节状态
        self.sub_joints = self.create_subscription(
            JointState, '/human_arm_state_left',
            self.joints_callback, 10)
        
        # 发布控制指令
        self.pub_left = self.create_publisher(
            Float64MultiArray, '/human_arm_ctrl_left', 10)
        self.pub_right = self.create_publisher(
            Float64MultiArray, '/human_arm_ctrl_right', 10)
        
        # 存储最新数据
        self.images = {}
        self.joints = None
        self.task_name = "拿起红色的杯子"  # 当前任务
        
        # 控制循环
        self.timer = self.create_timer(0.05, self.control_loop)  # 20Hz
        
    def head_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        _, encoded = cv2.imencode('.jpg', cv_image)
        self.images['head'] = encoded
        
    def left_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        _, encoded = cv2.imencode('.jpg', cv_image)
        self.images['left_wrist'] = encoded
        
    def right_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        _, encoded = cv2.imencode('.jpg', cv_image)
        self.images['right_wrist'] = encoded
        
    def joints_callback(self, msg):
        self.joints = np.array(msg.position)
        
    def control_loop(self):
        # 检查数据是否就绪
        if len(self.images) < 3 or self.joints is None:
            return
            
        # 构建观测
        obs = {
            "images": self.images,
            "arm_joints": {
                "left": self.joints[:6],
                "right": self.joints[6:12],
            }
        }
        
        # 推理
        actions = self.evaluator.Inference_Dual_Arm_Tien_Kung2(
            obs, self.task_name)
        
        # 发布控制指令 (取第一个动作)
        left_cmd = Float64MultiArray()
        left_cmd.data = actions[0, :6].tolist()
        right_cmd = Float64MultiArray()
        right_cmd.data = actions[0, 6:12].tolist()
        
        self.pub_left.publish(left_cmd)
        self.pub_right.publish(right_cmd)
        
        self.get_logger().info(f'Published actions: {actions[0]}')

def main(args=None):
    rclpy.init(args=args)
    node = XR1DeploymentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 六、故障排查

### 6.1 模型加载失败

```bash
# 错误: ModuleNotFoundError: No module named 'lerobot'

# 解决: 确保在xr1环境中
source /home/leo/miniconda3/etc/profile.d/conda.sh
conda activate xr1

# 验证安装
python -c "import lerobot; print('✅ LeRobot已安装')"
```

### 6.2 CUDA内存不足

```python
# 错误: RuntimeError: CUDA out of memory

# 解决1: 减少batch_size
--batch_size=8  # 默认20，减小到8

# 解决2: 使用混合精度训练
--mixed_precision=fp16

# 解决3: 减少action_horizon
--policy.action_chunk_size=30  # 默认50
```

### 6.3 图像预处理错误

```python
# 错误: 图像尺寸不匹配

# 解决: 检查图像预处理流程
# 1. 确保图像resize到 (640, 480)
# 2. 确保归一化到 [0, 1]
# 3. 确保维度为 [1, 3, H, W]
```

### 6.4 ROS2连接失败

```bash
# 错误: 无法连接到机器人

# 检查1: 网络配置
ping 192.168.41.1
ip addr show enp1s0f0np0

# 检查2: ROS2环境
source ~/activate_ros2.sh
ros2 topic list

# 检查3: 机器人是否启动
ssh ubuntu@192.168.41.1
ros2 node list
```

---

## 📚 参考资源

- **XR-1 GitHub**: https://github.com/Open-X-Humanoid/XR-1
- **RoboMIND数据集**: https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND
- **LeRobot文档**: https://github.com/huggingface/lerobot
- **ROS2 Humble**: https://docs.ros.org/en/humble/

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-02-02 | v1.0 | 初始版本，包含推理、微调、部署完整流程 |

---

**文档维护**: DGX Spark开发环境  
**问题反馈**: 请在GitHub Issues中提交
