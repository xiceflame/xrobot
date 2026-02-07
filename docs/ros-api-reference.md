# ROS接口参考手册

## 概述

本手册详细列出天工Pro机器人所有ROS接口，包括Topics（话题）和Services（服务），用于状态获取和控制指令发送。

## 接口总览

| 类别 | 接口类型 | 数量 | 说明 |
|------|---------|------|------|
| IMU | Topic | 1 | 惯性测量单元数据 |
| 头部/腰部 | Topic | 4 | 状态获取和控制 |
| 腿部 | Topic | 3 | 电机控制（力位混合、速度、位置） |
| 臂部 | Topic | 4 | 状态获取和插补控制 |
| 手部 | Service | 6 | 角度、力矩、速度控制 |
| 六维力 | Topic | 2 | 臂端力传感器 |
| 相机 | Topic | 多个 | 导航和大模型相机 |

---

## IMU接口

### 状态获取

**Topic**: `/BodyControl/imu`

**消息类型**: `sensor_msgs/Imu`

**发布频率**: 100Hz

**数据内容**:
```yaml
orientation:        # 四元数姿态
  x: float64
  y: float64
  z: float64
  w: float64
orientation_covariance: float64[9]

angular_velocity:   # 角速度 (rad/s)
  x: float64
  y: float64
  z: float64
angular_velocity_covariance: float64[9]

linear_acceleration: # 线加速度 (m/s²)
  x: float64
  y: float64
  z: float64
linear_acceleration_covariance: float64[9]
```

**使用示例**:
```bash
# 查看IMU数据
rostopic echo /BodyControl/imu

# 查看发布频率
rostopic hz /BodyControl/imu
```

---

## 头部/腰部电机接口

### 头部电机

#### 状态获取

**Topic**: `/BodyControl/ey/status`

**消息类型**: `std_msgs/Float64MultiArray`

**数据内容**: 头部各关节角度（弧度）

#### 位置控制

**Topic**: `/BodyControl/ey/set_pos`

**消息类型**: `std_msgs/Float64`

**说明**: 设置头部目标位置

**有效范围**: [-1.57, 1.57] rad

**使用示例**:
```cpp
// C++示例
std_msgs::Float64 msg;
msg.data = 0.5;  // 目标位置（弧度）
pub.publish(msg);
```

### 腰部电机

#### 状态获取

**Topic**: `/BodyControl/ze/status`

**消息类型**: `std_msgs/Float64MultiArray`

#### 位置控制

**Topic**: `/BodyControl/ze/set_pos`

**消息类型**: `std_msgs/Float64`

**有效范围**: [-0.78, 0.78] rad

---

## 腿部电机接口

### 力位混合控制

**Topic**: `/BodyControl/motor_ctrl`

**消息类型**: `std_msgs/Float64MultiArray`

**数据格式**:
```yaml
# 数组内容（按顺序）:
# [左腿髋关节, 左腿膝关节, 左腿踝关节, 
#  右腿髋关节, 右腿膝关节, 右腿踝关节]
data: float64[]
```

### 速度控制

**Topic**: `/BodyControl/motor_vel_ctrl`

**消息类型**: `std_msgs/Float64MultiArray`

### 位置控制

**Topic**: `/BodyControl/motor_pos_ctrl`

**消息类型**: `std_msgs/Float64MultiArray`

---

## 臂部电机接口

### 左臂状态获取

**Topic**: `/human_arm_state_left`

**消息类型**: `sensor_msgs/JointState`

**数据内容**:
```yaml
header:
  seq: uint32
  stamp: time
  frame_id: string

name: string[]        # 关节名称数组
position: float64[]   # 关节位置（弧度）
velocity: float64[]   # 关节速度（rad/s）
effort: float64[]     # 关节力矩（Nm）
```

**关节名称**（按顺序）:
1. `left_arm_joint1` - 肩关节1
2. `left_arm_joint2` - 肩关节2
3. `left_arm_joint3` - 肩关节3
4. `left_arm_joint4` - 肘关节
5. `left_arm_joint5` - 腕关节1
6. `left_arm_joint6` - 腕关节2

### 右臂状态获取

**Topic**: `/human_arm_state_right`

**消息类型**: `sensor_msgs/JointState`

**关节名称**（按顺序）:
1. `right_arm_joint1` - 肩关节1
2. `right_arm_joint2` - 肩关节2
3. `right_arm_joint3` - 肩关节3
4. `right_arm_joint4` - 肘关节
5. `right_arm_joint5` - 腕关节1
6. `right_arm_joint6` - 腕关节2

### 左臂插补控制

**Topic**: `/human_arm_ctrl_left`

**消息类型**: `sensor_msgs/JointState`

**说明**: 发送目标关节位置进行插补运动

**使用示例**:
```cpp
sensor_msgs::JointState msg;
msg.name = {"left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
            "left_arm_joint4", "left_arm_joint5", "left_arm_joint6"};
msg.position = {0.5, 0.3, 0.0, -0.5, 0.0, 0.0};  // 目标位置
pub.publish(msg);
```

### 右臂插补控制

**Topic**: `/human_arm_ctrl_right`

**消息类型**: `sensor_msgs/JointState`

### 轴空间运动控制（Service）

**Service**: `/aearm_traj_test_left/plan_joint_traj`

**请求类型**: `motion_control_msgs/PlanJointTraj`

**响应类型**: `motion_control_msgs/PlanJointTrajResponse`

**使用示例**:
```cpp
ros::ServiceClient client = nh.serviceClient<motion_control_msgs::PlanJointTraj>(
    "/aearm_traj_test_left/plan_joint_traj");

motion_control_msgs::PlanJointTraj srv;
srv.request.joint_names = {"left_arm_joint1", "left_arm_joint2"};
srv.request.waypoints = {0.5, 0.3};
srv.request.duration = 2.0;  // 运动时间（秒）

if (client.call(srv)) {
    ROS_INFO("轨迹规划成功");
} else {
    ROS_ERROR("轨迹规划失败");
}
```

---

## 手部电机接口（Services）

### 左手控制

#### 设置角度

**Service**: `/inspire_hand/set_angle/left_hand`

**请求类型**: `inspire_hand_msgs/SetAngle`

**请求内容**:
```yaml
angle1: float64  # 拇指弯曲
angle2: float64  # 食指弯曲
angle3: float64  # 中指弯曲
angle4: float64  # 无名指弯曲
angle5: float64  # 小指弯曲
angle6: float64  # 拇指旋转
```

**有效范围**: [0, 100]（百分比）

#### 获取角度

**Service**: `/inspire_hand/get_angle/left_hand`

**响应类型**: `inspire_hand_msgs/GetAngleResponse`

#### 设置力矩

**Service**: `/inspire_hand/set_torque/left_hand`

**请求类型**: `inspire_hand_msgs/SetTorque`

#### 设置速度

**Service**: `/inspire_hand/set_speed/left_hand`

**请求类型**: `inspire_hand_msgs/SetSpeed`

### 右手控制

与左手类似，Service路径中的`left_hand`替换为`right_hand`:
- `/inspire_hand/set_angle/right_hand`
- `/inspire_hand/get_angle/right_hand`
- `/inspire_hand/set_torque/right_hand`
- `/inspire_hand/set_speed/right_hand`

**使用示例**:
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from inspire_hand_msgs.srv import SetAngle

rospy.init_node('hand_controller')
rospy.wait_for_service('/inspire_hand/set_angle/left_hand')

try:
    set_angle = rospy.ServiceProxy('/inspire_hand/set_angle/left_hand', SetAngle)
    resp = set_angle(angle1=50, angle2=80, angle3=80, angle4=80, angle5=80, angle6=0)
    rospy.loginfo("手部控制成功")
except rospy.ServiceException as e:
    rospy.logerr("服务调用失败: %s", e)
```

---

## 六维力传感器接口

### 左臂六维力

**Topic**: `/human_arm_6dof_left`

**消息类型**: `geometry_msgs/WrenchStamped`

**数据内容**:
```yaml
header:
  seq: uint32
  stamp: time
  frame_id: string

wrench:
  force:          # 力 (N)
    x: float64
    y: float64
    z: float64
  torque:         # 力矩 (Nm)
    x: float64
    y: float64
    z: float64
```

### 右臂六维力

**Topic**: `/human_arm_6dof_right`

**消息类型**: `geometry_msgs/WrenchStamped`

---

## 相机接口

### 导航Orin相机

| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/camera/nav/color/image_raw` | `sensor_msgs/Image` | 彩色图像 |
| `/camera/nav/depth/image_rect_raw` | `sensor_msgs/Image` | 深度图像 |
| `/camera/nav/color/camera_info` | `sensor_msgs/CameraInfo` | 相机参数 |

### 大模型Orin相机

| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/camera/llm/color/image_raw` | `sensor_msgs/Image` | 彩色图像 |
| `/camera/llm/color/camera_info` | `sensor_msgs/CameraInfo` | 相机参数 |

**使用示例**:
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2.imshow("Camera", cv_image)
    cv2.waitKey(1)

rospy.init_node('camera_viewer')
rospy.Subscriber('/camera/nav/color/image_raw', Image, image_callback)
rospy.spin()
```

---

## 接口速查表

### Topics（状态获取）

| 功能 | Topic名称 | 消息类型 | 频率 |
|------|----------|---------|------|
| IMU | `/BodyControl/imu` | `sensor_msgs/Imu` | 100Hz |
| 头部状态 | `/BodyControl/ey/status` | `std_msgs/Float64MultiArray` | 50Hz |
| 腰部状态 | `/BodyControl/ze/status` | `std_msgs/Float64MultiArray` | 50Hz |
| 左臂状态 | `/human_arm_state_left` | `sensor_msgs/JointState` | 100Hz |
| 右臂状态 | `/human_arm_state_right` | `sensor_msgs/JointState` | 100Hz |
| 左臂六维力 | `/human_arm_6dof_left` | `geometry_msgs/WrenchStamped` | 100Hz |
| 右臂六维力 | `/human_arm_6dof_right` | `geometry_msgs/WrenchStamped` | 100Hz |

### Topics（控制指令）

| 功能 | Topic名称 | 消息类型 | 说明 |
|------|----------|---------|------|
| 头部控制 | `/BodyControl/ey/set_pos` | `std_msgs/Float64` | 位置控制 |
| 腰部控制 | `/BodyControl/ze/set_pos` | `std_msgs/Float64` | 位置控制 |
| 腿部控制 | `/BodyControl/motor_ctrl` | `std_msgs/Float64MultiArray` | 力位混合 |
| 左臂插补 | `/human_arm_ctrl_left` | `sensor_msgs/JointState` | 关节插补 |
| 右臂插补 | `/human_arm_ctrl_right` | `sensor_msgs/JointState` | 关节插补 |

### Services

| 功能 | Service名称 | 请求类型 | 说明 |
|------|------------|---------|------|
| 左手设角度 | `/inspire_hand/set_angle/left_hand` | `SetAngle` | 设置手指角度 |
| 左手获角度 | `/inspire_hand/get_angle/left_hand` | `GetAngle` | 获取手指角度 |
| 左手设力矩 | `/inspire_hand/set_torque/left_hand` | `SetTorque` | 设置力矩 |
| 左手设速度 | `/inspire_hand/set_speed/left_hand` | `SetSpeed` | 设置速度 |
| 右手设角度 | `/inspire_hand/set_angle/right_hand` | `SetAngle` | 设置手指角度 |
| 右手获角度 | `/inspire_hand/get_angle/right_hand` | `GetAngle` | 获取手指角度 |
| 左臂轨迹 | `/aearm_traj_test_left/plan_joint_traj` | `PlanJointTraj` | 轴空间运动 |
| 右臂轨迹 | `/aearm_traj_test_right/plan_joint_traj` | `PlanJointTraj` | 轴空间运动 |

---

## 数据格式详解

### JointState消息

`sensor_msgs/JointState`是臂部控制中最常用的消息类型：

```yaml
# 标准ROS消息格式
string[] name      # 关节名称数组
float64[] position # 关节位置 [rad]
float64[] velocity # 关节速度 [rad/s]
float64[] effort   # 关节力矩 [Nm]
```

**注意事项**:
- `name`、`position`、`velocity`、`effort`数组长度必须一致
- 关节名称必须与URDF中定义的joint名称匹配
- position值必须在URDF定义的limits范围内

### Float64MultiArray消息

用于头部、腰部和腿部控制：

```yaml
std_msgs/MultiArrayLayout layout
float64[] data
```

**腿部电机顺序**:
```
data[0]: 左腿髋关节
data[1]: 左腿膝关节
data[2]: 左腿踝关节
data[3]: 右腿髋关节
data[4]: 右腿膝关节
data[5]: 右腿踝关节
```

---

## 调试命令

### 查看所有话题
```bash
rostopic list | grep -E "(BodyControl|human_arm|inspire_hand)"
```

### 查看话题类型
```bash
rostopic type /BodyControl/imu
```

### 查看消息格式
```bash
rosmsg show sensor_msgs/Imu
```

### 实时监控数据
```bash
# IMU数据
rostopic echo /BodyControl/imu/orientation

# 臂部关节位置
rostopic echo /human_arm_state_left/position

# 六维力数据
rostopic echo /human_arm_6dof_left/wrench/force
```

### 测试发布
```bash
# 测试头部控制
rostopic pub /BodyControl/ey/set_pos std_msgs/Float64 "data: 0.5" --once

# 测试臂部控制
rostopic pub /human_arm_ctrl_left sensor_msgs/JointState "header: auto, name: ['left_arm_joint1'], position: [0.5], velocity: [], effort: []" --once
```

---

**最后更新**: 2025-01-30
