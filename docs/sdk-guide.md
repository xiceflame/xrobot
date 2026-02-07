# 天工Pro SDK开发指南

## 概述

天工Pro SDK是由北京人形机器人创新中心开发的完整开发工具包，提供丰富的接口用于控制天工Pro人形机器人的各个部件，包括头部、臂部、手部、腰部和腿部的电机控制，以及六维力传感器、IMU和相机的数据获取。

## 系统架构

### 硬件规格
- **自由度**: 42 DOF
- **身高**: 1630mm
- **体重**: 56kg
- **构型**: 双足人形机器人

### 计算单元
- **运控单元**: Intel x86架构
- **开发单元**: NVIDIA Orin AGX ×2
  - 导航Orin: 处理相机和导航任务
  - 大模型Orin: 处理AI和决策任务

### 软件架构
- **操作系统**: Ubuntu 20.04/22.04 LTS
- **中间件**: ROS 1 (Noetic)
- **通信机制**: ROS Topics 和 Services

## 环境依赖

### 系统要求
| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| ROS版本 | ROS Melodic | ROS Noetic |
| 编译器 | GCC 7+ | GCC 9+ |
| CMake | 3.5+ | 3.8+ |
| 内存 | 8GB | 16GB+ |
| 存储 | 50GB | 100GB+ |

### 依赖包
```bash
# ROS基础包
sudo apt-get install ros-noetic-desktop-full

# 常用工具
sudo apt-get install ros-noetic-catkin
sudo apt-get install ros-noetic-std-msgs
sudo apt-get install ros-noetic-sensor-msgs
sudo apt-get install ros-noetic-geometry-msgs
sudo apt-get install ros-noetic-control-msgs

# 开发工具
sudo apt-get install cmake git build-essential
sudo apt-get install libeigen3-dev
```

## 开发流程

### 1. 创建工作空间
```bash
mkdir -p ~/tiankung_ws/src
cd ~/tiankung_ws/src
catkin_init_workspace
cd ..
catkin_make
source devel/setup.bash
```

### 2. 获取SDK代码
```bash
cd ~/tiankung_ws/src
git clone https://github.com/x-humanoid-robomind/tienkung_ros.git
```

### 3. 编译工作空间
```bash
cd ~/tiankung_ws
catkin_make
source devel/setup.bash
```

### 4. 启动机器人驱动
```bash
# 终端1: 启动本体驱动
roslaunch body_control motion_evt.launch

# 终端2: 启动手臂驱动
roslaunch aubo_dev_plugin aubo_dev_all.launch

# 终端3: 启动运控驱动
roslaunch motion_control motion_control.launch
```

## 快速开始示例

### 示例1: 读取IMU数据
```cpp
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>

void imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    ROS_INFO("IMU Orientation: x=%.2f, y=%.2f, z=%.2f, w=%.2f",
             msg->orientation.x, msg->orientation.y,
             msg->orientation.z, msg->orientation.w);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "imu_reader");
    ros::NodeHandle nh;
    
    ros::Subscriber sub = nh.subscribe("/BodyControl/imu", 10, imuCallback);
    ros::spin();
    return 0;
}
```

### 示例2: 控制头部电机
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import Float64

rospy.init_node('head_controller')
pub = rospy.Publisher('/BodyControl/ey/set_pos', Float64, queue_size=10)

# 设置头部位置
msg = Float64()
msg.data = 0.5  # 位置值（弧度）
pub.publish(msg)
rospy.sleep(1)
```

## 接口分类

### 状态获取接口 (Topics)
用于获取机器人各部件的实时状态数据：
- **IMU**: `/BodyControl/imu`
- **头部状态**: `/BodyControl/ey/status`
- **腰部状态**: `/BodyControl/ze/status`
- **臂部状态**: `/human_arm_state_left`, `/human_arm_state_right`
- **六维力**: `/human_arm_6dof_left`, `/human_arm_6dof_right`
- **相机**: 多个相机Topic

### 控制接口 (Topics & Services)
用于发送控制指令：
- **头部控制**: `/BodyControl/ey/set_pos`
- **腰部控制**: `/BodyControl/ze/set_pos`
- **臂部控制**: `/human_arm_ctrl_left`, `/human_arm_ctrl_right`
- **手部控制**: `/inspire_hand/set_angle/left_hand` (Service)
- **腿部控制**: `/BodyControl/motor_ctrl`

## 安全须知

### 操作前检查
1. 确保机器人周围无障碍物
2. 检查紧急停止按钮功能正常
3. 确认所有关节处于安全位置
4. 验证电源和通信连接稳定

### 运行时注意事项
1. 始终保持对机器人的视觉监控
2. 准备随时触发紧急停止
3. 避免超出关节限位的运动指令
4. 注意六维力传感器的过载保护

### 紧急停止
```bash
# 发布紧急停止指令
rostopic pub /emergency_stop std_msgs/Bool "data: true"
```

## 常见问题

### Q1: 无法连接到机器人
**A**: 检查以下几点：
- 确认机器人已开机并进入就绪状态
- 检查网络连接（ping机器人IP）
- 验证ROS_MASTER_URI设置正确
- 确认防火墙未阻断ROS端口

### Q2: 话题没有数据
**A**: 
- 确认驱动程序已启动：`roslaunch body_control motion_evt.launch`
- 检查话题是否存在：`rostopic list | grep BodyControl`
- 验证话题类型：`rostopic type /BodyControl/imu`

### Q3: 控制指令无响应
**A**:
- 确认机器人处于控制模式
- 检查指令值是否在有效范围内
- 验证话题名称拼写正确
- 查看是否有错误日志：`rostopic echo /rosout`

### Q4: 编译错误
**A**:
- 确保所有依赖包已安装
- 清理并重新编译：`catkin_make clean && catkin_make`
- 检查CMakeLists.txt配置
- 验证ROS环境变量：`echo $ROS_PACKAGE_PATH`

## 调试技巧

### 使用rqt_graph查看节点关系
```bash
rosrun rqt_graph rqt_graph
```

### 使用rostopic监控数据
```bash
# 查看话题列表
rostopic list

# 查看话题数据
rostopic echo /BodyControl/imu

# 查看话题频率
rostopic hz /BodyControl/imu
```

### 使用rviz可视化
```bash
rosrun rviz rviz -d ~/tiankung_ws/src/tienkung_ros/config/tiankung.rviz
```

## 进阶开发

### 自定义控制算法
1. 创建新的ROS包
2. 订阅传感器数据Topic
3. 实现控制算法
4. 发布控制指令

### 集成第三方库
- **MoveIt**: 运动规划和碰撞检测
- **Gazebo**: 仿真环境测试
- **OpenCV**: 视觉处理
- **PCL**: 点云处理

## 相关资源

- **官方文档**: https://x-humanoid.com/download/
- **GitHub**: https://github.com/Open-X-Humanoid
- **开源社区**: https://opensource.x-humanoid-cloud.com/
- **ROS Wiki**: http://wiki.ros.org/

## 技术支持

如有问题，请联系：
- 技术支持邮箱: support@x-humanoid.com
- 开源社区论坛: https://opensource.x-humanoid-cloud.com/

---

**最后更新**: 2025-01-30
