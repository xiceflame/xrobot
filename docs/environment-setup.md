# 开发环境配置指南

## 概述

本指南详细介绍如何从零开始搭建天工Pro机器人的开发环境，包括操作系统安装、ROS配置、工作空间设置和依赖包安装。

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| 内存 | 8 GB | 16 GB 或更高 |
| 存储 | 50 GB 可用空间 | 100 GB SSD |
| 显卡 | 集成显卡 | NVIDIA独立显卡 |
| 网络 | 100 Mbps | 1 Gbps |

### 软件要求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 20.04/22.04 LTS | 64位桌面版 |
| ROS | Noetic (ROS 1) | 机器人操作系统 |
| CMake | 3.8+ | 构建工具 |
| GCC | 9+ | C++编译器 |
| Python | 3.8+ | 脚本语言 |

## 操作系统安装

### 下载Ubuntu

1. 访问 [Ubuntu官网](https://ubuntu.com/download/desktop)
2. 下载 Ubuntu 20.04.6 LTS 或 22.04.4 LTS
3. 制作启动U盘（使用Rufus或Etcher）

### 安装步骤

1. 从U盘启动，选择"Install Ubuntu"
2. 选择语言：中文或英文
3. 键盘布局：默认即可
4. 网络连接：连接到互联网
5. 更新和其他软件：
   - 选择"正常安装"
   - 勾选"安装第三方软件"
6. 安装类型：
   - 双系统：选择"其他选项"手动分区
   - 单系统：选择"清除整个磁盘并安装Ubuntu"
7. 设置用户名和密码
8. 等待安装完成并重启

### 系统初始化

```bash
# 更新系统
sudo apt-get update
sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y git vim wget curl net-tools htop

# 安装中文支持（可选）
sudo apt-get install -y language-pack-zh-hans
```

## ROS安装

### 配置软件源

```bash
# 添加ROS软件源
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# 添加密钥
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 更新软件源
sudo apt-get update
```

### 安装ROS Noetic

```bash
# 完整安装（推荐）
sudo apt-get install -y ros-noetic-desktop-full

# 或者最小安装
# sudo apt-get install -y ros-noetic-ros-base
```

### 配置ROS环境

```bash
# 初始化rosdep
sudo rosdep init
rosdep update

# 添加环境变量到.bashrc
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 安装构建工具
sudo apt-get install -y python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 验证ROS安装

```bash
# 检查ROS版本
rosversion -d
# 应输出: noetic

# 测试roscore
roscore
```

## 工作空间配置

### 创建工作空间

```bash
# 创建工作空间目录
mkdir -p ~/tiankung_ws/src
cd ~/tiankung_ws/src

# 初始化工作空间
catkin_init_workspace
```

### 获取天工软件系统

```bash
cd ~/tiankung_ws/src

# 克隆Lite版本ROS系统（参考用）
git clone https://github.com/x-humanoid-robomind/tienkung_ros.git

# 克隆URDF模型
git clone https://github.com/x-humanoid-robomind/tienkung_urdf.git

# 克隆强化学习控制（可选）
git clone https://github.com/Open-X-Humanoid/Deploy_Tienkung.git
```

### 编译工作空间

```bash
cd ~/tiankung_ws

# 安装依赖
rosdep install --from-paths src --ignore-src -r -y

# 编译
catkin_make

# 添加环境变量
echo "source ~/tiankung_ws/devel/setup.bash" >> ~/.bashrc
source ~/tiankung_ws/devel/setup.bash
```

### 验证编译

```bash
# 检查环境变量
echo $ROS_PACKAGE_PATH
# 应包含: /home/username/tiankung_ws/src

# 查看包列表
rospack list | grep tienkung
```

## 依赖包安装

### ROS基础包

```bash
# 核心消息包
sudo apt-get install -y ros-noetic-std-msgs
sudo apt-get install -y ros-noetic-sensor-msgs
sudo apt-get install -y ros-noetic-geometry-msgs
sudo apt-get install -y ros-noetic-nav-msgs

# 控制相关
sudo apt-get install -y ros-noetic-control-msgs
sudo apt-get install -y ros-noetic-trajectory-msgs

# 视觉相关
sudo apt-get install -y ros-noetic-cv-bridge
sudo apt-get install -y ros-noetic-image-transport
sudo apt-get install -y ros-noetic-camera-info-manager

# 工具包
sudo apt-get install -y ros-noetic-rqt-graph
sudo apt-get install -y ros-noetic-rviz
```

### 开发工具

```bash
# C++开发
sudo apt-get install -y build-essential cmake git
sudo apt-get install -y libeigen3-dev
sudo apt-get install -y libboost-all-dev

# Python开发
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-numpy
sudo apt-get install -y python3-opencv

# 安装Python ROS包
pip3 install rospkg catkin_pkg
```

### 仿真工具（可选）

```bash
# Gazebo仿真
sudo apt-get install -y ros-noetic-gazebo-ros
sudo apt-get install -y ros-noetic-gazebo-ros-pkgs
sudo apt-get install -y ros-noetic-gazebo-ros-control

# 运动规划
sudo apt-get install -y ros-noetic-moveit
```

## 网络配置

### 单机配置

如果只在本地开发，无需额外配置。

### 多机通信配置

#### 主机配置（机器人）

```bash
# 编辑.bashrc
export ROS_MASTER_URI=http://机器人IP:11311
export ROS_IP=机器人IP
```

#### 从机配置（开发机）

```bash
# 编辑.bashrc
export ROS_MASTER_URI=http://机器人IP:11311
export ROS_IP=本机IP
```

### 防火墙设置

```bash
# 开放ROS端口
sudo ufw allow 11311/tcp  # ROS Master
sudo ufw allow 11411/tcp  # ROS Parameter Server

# 或者临时关闭防火墙（仅开发环境）
sudo ufw disable
```

## 开发工具配置

### VSCode（推荐）

```bash
# 下载并安装VSCode
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg

sudo apt update
sudo apt install -y code
```

**推荐插件**:
- C/C++
- Python
- ROS
- CMake Tools
- GitLens

### Git配置

```bash
# 配置Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor vim

# 生成SSH密钥（用于GitHub）
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
cat ~/.ssh/id_rsa.pub
# 将公钥添加到GitHub
```

## 验证安装

### 完整测试流程

```bash
# 1. 检查ROS环境
source ~/.bashrc
rosversion -d

# 2. 启动roscore
roscore

# 3. 新终端：检查话题列表
rostopic list

# 4. 测试发布/订阅
# 终端A：
rostopic pub /test std_msgs/String "data: 'Hello TianKung'" -r 1

# 终端B：
rostopic echo /test

# 5. 测试RViz
rosrun rviz rviz
```

### 常见问题排查

#### 问题1: rosdep init失败

**错误信息**: `ERROR: cannot download default sources list from: ...`

**解决方案**:
```bash
# 使用代理或手动创建文件
sudo mkdir -p /etc/ros/rosdep/sources.list.d/
sudo curl -o /etc/ros/rosdep/sources.list.d/20-default.list https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/sources.list.d/20-default.list
rosdep update
```

#### 问题2: catkin_make失败

**错误信息**: `Command 'catkin_make' not found`

**解决方案**:
```bash
# 确保source了ROS环境
source /opt/ros/noetic/setup.bash

# 或者安装python-catkin-tools
sudo apt-get install -y python3-catkin-tools
```

#### 问题3: 找不到ROS包

**错误信息**: `[rospack] Error: package 'xxx' not found`

**解决方案**:
```bash
# 检查环境变量
source ~/tiankung_ws/devel/setup.bash
echo $ROS_PACKAGE_PATH

# 重新编译
cd ~/tiankung_ws
catkin_make
source devel/setup.bash
```

#### 问题4: 权限不足

**错误信息**: `Permission denied`

**解决方案**:
```bash
# 修改工作空间权限
sudo chown -R $USER:$USER ~/tiankung_ws

# 或者使用sudo（不推荐）
sudo -E bash -c 'source /opt/ros/noetic/setup.bash; roscore'
```

## 优化配置

### 性能优化

```bash
# 限制ROS日志大小
echo "export ROS_LOG_DIR=/tmp/ros_logs" >> ~/.bashrc
mkdir -p /tmp/ros_logs

# 禁用不必要的日志
export ROSCONSOLE_FORMAT='[${severity}] [${time}]: ${message}'
```

### 别名设置

```bash
# 编辑 ~/.bashrc，添加常用别名
alias cw='cd ~/tiankung_ws'
alias cs='cd ~/tiankung_ws/src'
alias cm='cd ~/tiankung_ws && catkin_make'
alias sb='source ~/tiankung_ws/devel/setup.bash'
alias rosenv='env | grep ROS'
```

## 下一步

环境配置完成后，请阅读以下文档：
- [SDK开发指南](sdk-guide.md) - 了解SDK架构和开发流程
- [ROS接口参考](ros-api-reference.md) - 查看所有可用接口
- [示例代码](../examples/) - 运行示例程序

## 参考资源

- [ROS官方安装指南](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [Ubuntu官方文档](https://help.ubuntu.com/)
- [天工开源社区](https://opensource.x-humanoid-cloud.com/)

---

**最后更新**: 2025-01-30
