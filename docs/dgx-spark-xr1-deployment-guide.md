# DGX Spark + XR-1 开发环境详细部署指南

## 📋 硬件概述

### NVIDIA DGX Spark 规格

| 组件 | 规格 | XR-1适配性 |
|------|------|-----------|
| **架构** | NVIDIA Grace Blackwell (GB10) | ✅ 完美支持 |
| **CPU** | 20核 Arm (10×Cortex-X925 + 10×Cortex-A725) | ✅ 原生支持 |
| **GPU** | Blackwell架构，6144 CUDA核心 | ✅ 强大算力 |
| **内存** | **128GB LPDDR5x统一内存** | ✅ 远超需求(需7-10GB) |
| **存储** | 1TB/4TB NVMe SSD | ✅ 充足 |
| **AI性能** | 1000 TOPS推理 / 1 PFLOP FP4 | ✅ 实时推理 |
| **网络** | 10GbE + Wi-Fi 7 + ConnectX-7 | ✅ 高速连接 |
| **功耗** | 140W TDP (240W电源) | ✅ 低功耗 |
| **尺寸** | 150×150×50.5mm (1.2kg) | ✅ 便携 |

**结论**: DGX Spark的128GB统一内存和强大AI性能完全满足XR-1的3B模型需求，是理想的开发平台。

---

## 第一阶段：DGX Spark 初始化设置

### 1.1 硬件连接

#### 连接清单
```
必需项:
✓ DGX Spark主机
✓ 240W电源适配器（随附）
✓ 网线（推荐）或Wi-Fi环境
✓ HDMI显示器（首次设置）
✓ USB键盘鼠标（首次设置）

可选项:
○ USB-C扩展坞（增加接口）
○ 外接SSD（额外存储）
```

#### 物理连接步骤

**步骤1: 连接电源**
```
1. 将240W电源适配器插入DGX Spark电源接口
2. 插入电源插座
3. ⚠️ 注意：设备会立即启动（无电源按钮）
```

**步骤2: 连接显示和输入设备**
```
1. HDMI线连接显示器
2. USB键盘插入任意USB-C口（需转接器或扩展坞）
3. USB鼠标插入
4. 网线插入RJ-45接口（推荐有线网络）
```

**步骤3: 验证启动**
```
指示灯状态:
- 电源LED: 常亮（白色/蓝色）
- 风扇: 轻微转动（可能听不到声音）
- 显示器: 显示DGX OS启动画面
```

---

### 1.2 首次启动配置

#### 启动模式选择

DGX Spark支持两种初始化方式：

**方式A: 本地设置（推荐首次使用）**
```
适用场景: 首次开箱设置
需要设备: 显示器 + 键盘 + 鼠标
优势: 直观，易于排查问题
```

**方式B: 网络设置（远程配置）**
```
适用场景: 无显示器环境
需要设备: 另一台电脑 + Wi-Fi/网线
优势: 无需外设，远程完成
```

#### 本地设置详细步骤

**步骤1: 首次启动向导**
```
1. 开机后等待30-60秒
2. 屏幕显示"First-Time Setup Utility"
3. 选择语言: English / 简体中文
4. 选择时区: Asia/Shanghai (UTC+8)
```

**步骤2: 网络配置**
```
推荐: 有线网络（Ethernet）
- 自动获取IP（DHCP）
- 测试网络连接: ping www.nvidia.com

备选: Wi-Fi配置
- 选择Wi-Fi网络名称（SSID）
- 输入密码
- 等待连接成功
```

**步骤3: 创建用户账户**
```
用户名: xr1-dev（建议）
密码: [强密码，至少12位]
确认密码

⚠️ 重要: 记住这个密码，后续SSH和sudo都需要
```

**步骤4: 系统更新**
```
自动下载并安装:
- DGX OS系统更新
- NVIDIA驱动更新
- 安全补丁

耗时: 10-30分钟（取决于网络）
```

**步骤5: 完成设置**
```
- 重启系统
- 使用新创建的用户登录
- 验证: 桌面环境正常显示
```

---

### 1.3 网络配置详解

#### 有线网络配置（推荐）

**自动DHCP（默认）**
```bash
# 验证网络连接
ping -c 4 www.nvidia.com

# 查看IP地址
ip addr show eth0

# 预期输出:
# inet 192.168.x.x/24 brd 192.168.x.255 scope global dynamic eth0
```

**静态IP配置（如需）**
```bash
# 编辑网络配置
sudo nano /etc/netplan/01-netcfg.yaml

# 添加静态IP配置:
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      routes:
        - to: default
          via: 192.168.1.1
      nameservers:
        addresses:
          - 8.8.8.8
          - 114.114.114.114

# 应用配置
sudo netplan apply
```

#### Wi-Fi配置（备选）

**图形界面配置**
```
1. 点击桌面右上角网络图标
2. 选择"Wi-Fi Settings"
3. 选择网络名称
4. 输入密码
5. 等待连接
```

**命令行配置**
```bash
# 查看可用Wi-Fi网络
nmcli dev wifi list

# 连接到Wi-Fi
sudo nmcli dev wifi connect "SSID" password "password"

# 验证连接
nmcli connection show
```

---

### 1.4 远程访问配置

#### SSH服务配置

**启用SSH**
```bash
# DGX Spark默认已安装SSH服务
# 验证SSH服务状态
sudo systemctl status ssh

# 如未启动，手动启动
sudo systemctl enable ssh
sudo systemctl start ssh

# 查看IP地址（用于远程连接）
ip addr show | grep "inet "
```

**SSH连接测试（从另一台电脑）**
```bash
# 在Mac/Linux终端或Windows PowerShell
ssh xr1-dev@192.168.1.100  # 替换为实际IP

# 输入密码后应成功登录
```

#### NVIDIA Sync配置（推荐）

NVIDIA Sync是NVIDIA官方提供的远程管理工具，支持Windows/Mac/Linux。

**步骤1: 在本地电脑安装NVIDIA Sync**
```
下载地址: https://www.nvidia.com/en-us/sync/
支持平台: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
```

**步骤2: 添加DGX Spark设备**
```
1. 打开NVIDIA Sync应用
2. 点击"Add Device"
3. 输入信息:
   - Hostname/IP: 192.168.1.100（DGX Spark IP）
   - Username: xr1-dev
   - Password: [你的密码]
4. 点击"Connect"
5. 等待SSH密钥自动配置
```

**步骤3: 使用NVIDIA Sync功能**
```
功能列表:
- DGX Dashboard: Web界面监控GPU/内存/存储
- Terminal: 内置SSH终端
- File Transfer: 拖拽文件传输
- JupyterLab: 一键启动（如果已安装）
```

---

## 第二阶段：开发环境配置

### 2.1 系统更新与基础工具

#### 系统更新
```bash
# 更新软件包列表
sudo apt update

# 升级所有软件包
sudo apt upgrade -y

# 安装基础开发工具
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    net-tools
```

#### 配置Git
```bash
# 设置Git用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 生成SSH密钥（用于GitHub）
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub
# 将公钥添加到GitHub: Settings -> SSH and GPG keys
```

---

### 2.2 Docker环境配置

DGX Spark预装了NVIDIA Container Runtime，但需要配置用户权限。

#### 配置Docker权限
```bash
# 将当前用户添加到docker组
sudo usermod -aG docker $USER

# 应用组变更（无需重启）
newgrp docker

# 验证Docker权限
docker ps
# 应显示空列表，无权限错误
```

#### 验证NVIDIA Container Runtime
```bash
# 测试GPU容器
docker run -it --runtime=nvidia --gpus=all nvidia/cuda:12.0-base nvidia-smi

# 预期输出: 显示GPU信息（Blackwell架构，6144 CUDA核心）
```

#### 配置Docker镜像加速（国内用户）
```bash
# 创建/编辑Docker配置
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
EOF

# 重启Docker服务
sudo systemctl restart docker
```

---

### 2.3 Conda环境配置

#### 安装Miniconda
```bash
# 下载Miniconda（Arm64版本）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

# 安装
bash Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda3

# 初始化
~/miniconda3/bin/conda init bash

# 重新加载配置
source ~/.bashrc

# 验证
conda --version
```

#### 配置Conda镜像（国内用户）
```bash
# 添加清华镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

---

### 2.4 PyTorch与CUDA环境

DGX Spark预装了PyTorch，但需要验证版本和CUDA兼容性。

#### 验证现有PyTorch安装
```bash
# 检查PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# 预期输出:
# PyTorch: 2.5.x
# CUDA available: True
# CUDA version: 12.x
```

#### 如需要重新安装PyTorch
```bash
# 创建XR-1专用环境
conda create -n xr1 python=3.10 -y
conda activate xr1

# 安装PyTorch（CUDA 12.4版本）
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 验证GPU可用
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 第三阶段：XR-1 部署

### 3.1 克隆XR-1仓库

```bash
# 创建工作目录
mkdir -p ~/projects
cd ~/projects

# 克隆XR-1仓库
git clone https://github.com/Open-X-Humanoid/XR-1.git
cd XR-1

# 查看目录结构
ls -la
```

### 3.2 安装XR-1依赖

```bash
# 确保在xr1 conda环境中
conda activate xr1

# 安装XR-1及其依赖
pip install -e ".[xr1]"

# 验证安装
python -c "import xr1; print('XR-1 installed successfully')"
```

#### 依赖安装常见问题

**问题1: Arm架构兼容性问题**
```bash
# 某些包可能没有Arm预编译版本，需要源码编译
# 如遇到错误，尝试:
pip install --no-binary :all: package_name

# 或安装编译依赖
sudo apt install -y python3-dev libopenblas-dev
```

**问题2: 内存不足（OOM）**
```bash
# DGX Spark有128GB内存，通常不会OOM
# 但如遇到，启用swap:
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 3.3 下载预训练模型

```bash
# 下载基础模型（SigLIP, PaliGemma等）
bash scripts/hf_download.sh

# 下载XR-1预训练模型
bash scripts/hf_xr1_pretrain_model_download.sh

# 或使用ModelScope（国内更快）
bash scripts/modelscope_xr1_pretrain_model_download.sh
```

**模型存储位置**
```
~/.cache/huggingface/hub/    # HuggingFace缓存
~/.cache/modelscope/          # ModelScope缓存
```

### 3.4 配置天工Pro接口

#### 创建天工Pro配置文件
```bash
# 创建配置目录
mkdir -p ~/projects/XR-1/configs/tiankung

# 创建配置文件
cat > ~/projects/XR-1/configs/tiankung/tiankung_pro.yaml <<EOF
robot:
  name: "tiankung_pro"
  type: "dual_arm_mobile"
  
  # 臂部配置
  arms:
    left:
      dof: 6
      state_topic: "/human_arm_state_left"
      cmd_topic: "/human_arm_ctrl_left"
      joint_names:
        - "left_arm_joint1"
        - "left_arm_joint2"
        - "left_arm_joint3"
        - "left_arm_joint4"
        - "left_arm_joint5"
        - "left_arm_joint6"
    right:
      dof: 6
      state_topic: "/human_arm_state_right"
      cmd_topic: "/human_arm_ctrl_right"
      joint_names:
        - "right_arm_joint1"
        - "right_arm_joint2"
        - "right_arm_joint3"
        - "right_arm_joint4"
        - "right_arm_joint5"
        - "right_arm_joint6"
  
  # 相机配置
  cameras:
    head:
      type: "rgb"
      topic: "/camera/nav/color/image_raw"
      resolution: [640, 480]
      fps: 30
    left_wrist:
      type: "rgb"
      topic: "/camera/left_wrist/color/image_raw"
      resolution: [640, 480]
      fps: 30
    right_wrist:
      type: "rgb"
      topic: "/camera/right_wrist/color/image_raw"
      resolution: [640, 480]
      fps: 30
  
  # 网络配置（连接天工Pro）
  network:
    ros_master_uri: "http://192.168.1.50:11311"  # 天工Pro IP
    local_ip: "192.168.1.100"                     # DGX Spark IP

# XR-1模型配置
model:
  checkpoint_path: "./checkpoints/xr1-stage2-pretrain"
  use_uvmc: true
  action_horizon: 8
  
  # 推理优化（DGX Spark专用）
  inference:
    precision: "bf16"          # DGX Spark支持bf16
    use_tensorrt: false        # 如需可启用
    batch_size: 1
    num_samples: 10
EOF
```

#### 配置ROS网络（连接天工Pro）

```bash
# 编辑bashrc
nano ~/.bashrc

# 添加以下内容到文件末尾:
# ROS网络配置（连接天工Pro）
export ROS_MASTER_URI=http://192.168.1.50:11311  # 天工Pro的IP
export ROS_IP=192.168.1.100                       # DGX Spark的IP

# 应用配置
source ~/.bashrc

# 验证ROS连接
rostopic list
# 应显示天工Pro的话题列表
```

---

## 第四阶段：网络架构设计

### 4.1 推荐网络拓扑

```
                    路由器/交换机 (192.168.1.1)
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    [DGX Spark]      [天工Pro]       [开发电脑]
   (192.168.1.100)  (192.168.1.50)  (192.168.1.10)
    - XR-1推理       - 机器人本体    - 远程开发
    - 数据收集       - ROS Master    - 监控调试
    - 模型训练       - 传感器        - NVIDIA Sync
```

### 4.2 网络配置检查清单

**DGX Spark网络配置**
```bash
# 1. 验证IP地址
ip addr show eth0
# 预期: inet 192.168.1.100/24

# 2. 验证网关
ip route | grep default
# 预期: default via 192.168.1.1

# 3. 验证DNS
nslookup www.nvidia.com
# 应返回IP地址

# 4. 验证与天工Pro连通性
ping 192.168.1.50
# 应收到回复

# 5. 验证ROS连接
export ROS_MASTER_URI=http://192.168.1.50:11311
rostopic list
# 应显示天工Pro的话题
```

**防火墙配置（如需要）**
```bash
# 开放ROS端口
sudo ufw allow 11311/tcp  # ROS Master
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 8888/tcp   # JupyterLab

# 启用防火墙
sudo ufw enable
```

---

## 第五阶段：数据收集与微调

### 5.1 遥操作数据收集

#### 配置VR遥操作（推荐）

**硬件需求**
- Meta Quest 3 / Quest Pro / Apple Vision Pro
- Wi-Fi 6/6E/7网络（低延迟）

**软件配置**
```bash
# 安装VR遥操作依赖
pip install openvr pyopenvr

# 启动VR遥操作节点
python deploy/vr_teleop/vr_teleop_node.py \
  --config configs/tiankung/tiankung_pro.yaml \
  --output_dir ./data/teleop_vr/
```

#### 配置手柄遥操作（备选）

```bash
# 使用Xbox/PS5手柄
pip install pygame

# 启动手柄遥操作
python deploy/gamepad_teleop/gamepad_teleop_node.py \
  --config configs/tiankung/tiankung_pro.yaml \
  --output_dir ./data/teleop_gamepad/
```

### 5.2 数据格式转换

```bash
# 将原始数据转换为LeRobot格式
python scripts/convert_to_lerobot.py \
  --input_dir ./data/teleop_vr/ \
  --output_dir ./data/lerobot_tiankung/ \
  --robot_type tiankung_pro

# 验证数据集
python -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('./data/lerobot_tiankung'); print(f'Episodes: {len(ds)}')"
```

### 5.3 模型微调

#### Stage 3快速微调（推荐）

```bash
# 启动微调（使用DGX Spark的GPU）
python scripts/xr1_stage3_finetune.py \
  --config configs/tiankung/tiankung_pro.yaml \
  --dataset_path ./data/lerobot_tiankung/ \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --num_epochs 50 \
  --output_dir ./checkpoints/stage3_tiankung/

# 监控训练（使用tmux保持会话）
tmux new -s xr1_training
# [在tmux会话中运行训练命令]
# Ctrl+B, D  detach会话
# tmux attach -t xr1_training  重新连接
```

**DGX Spark训练性能预期**
```
配置: batch_size=8, bf16
- 显存占用: ~12GB (128GB总内存，余量充足)
- 训练速度: ~2-4 iterations/second
- 50 epochs耗时: ~1-2小时 (100条轨迹)
```

---

## 第六阶段：部署与测试

### 6.1 启动XR-1推理服务

```bash
# 加载微调后的模型
python deploy/real_robot/xr1_deploy.py \
  --config configs/tiankung/tiankung_pro.yaml \
  --checkpoint ./checkpoints/stage3_tiankung/best_model.pt \
  --mode inference

# 或使用ROS节点方式
ros2 run xr1_deploy xr1_tiankung_node \
  --ros-args \
  -p checkpoint:="./checkpoints/stage3_tiankung/best_model.pt"
```

### 6.2 测试验证

```bash
# 测试脚本
python tests/test_xr1_tiankung.py \
  --task "拿起红色的杯子" \
  --max_steps 50

# 批量测试
python tests/benchmark_xr1.py \
  --tasks tasks/tiankung_benchmark.json \
  --output results/benchmark_results.json
```

---

## 第七阶段：监控与维护

### 7.1 系统监控

#### DGX Dashboard（Web界面）
```
访问地址: http://192.168.1.100:8080
功能:
- GPU/CPU/内存实时监控
- 温度监控
- 存储使用
- 进程管理
```

#### 命令行监控
```bash
# GPU监控
watch -n 1 nvidia-smi

# 系统资源
htop

# 网络监控
iftop

# ROS话题监控
rostopic hz /human_arm_state_left
```

### 7.2 日志管理

```bash
# XR-1日志
mkdir -p ~/logs/xr1

# 启动带日志记录的推理
python deploy/real_robot/xr1_deploy.py \
  --config configs/tiankung/tiankung_pro.yaml \
  --log_dir ~/logs/xr1/$(date +%Y%m%d_%H%M%S)
```

### 7.3 备份策略

```bash
# 自动备份脚本
#!/bin/bash
# backup_xr1.sh

BACKUP_DIR="/mnt/external_ssd/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 备份代码
tar -czf $BACKUP_DIR/xr1_code_$DATE.tar.gz ~/projects/XR-1/

# 备份模型
tar -czf $BACKUP_DIR/xr1_models_$DATE.tar.gz ~/projects/XR-1/checkpoints/

# 备份数据
tar -czf $BACKUP_DIR/xr1_data_$DATE.tar.gz ~/projects/XR-1/data/

# 保留最近10个备份
ls -t $BACKUP_DIR/*.tar.gz | tail -n +11 | xargs rm -f
```

---

## 附录：常见问题排查

### Q1: DGX Spark无法启动

**症状**: 插电后无反应，显示器无信号

**排查步骤**:
```bash
1. 检查电源LED是否亮起
2. 检查HDMI线是否插紧（尝试不同接口）
3. 长按电源键10秒强制重启
4. 检查显示器输入源设置
5. 尝试最小启动（仅电源+HDMI，无其他外设）
```

### Q2: 网络连接失败

**症状**: 无法获取IP，无法ping通网关

**排查步骤**:
```bash
# 检查网线连接
ethtool eth0 | grep Link

# 重启网络服务
sudo systemctl restart NetworkManager

# 手动获取IP
sudo dhclient -v eth0

# 检查路由器DHCP设置
```

### Q3: ROS无法连接天工Pro

**症状**: rostopic list显示为空或报错

**排查步骤**:
```bash
# 1. 检查网络连通性
ping 192.168.1.50

# 2. 检查ROS环境变量
echo $ROS_MASTER_URI
echo $ROS_IP

# 3. 检查防火墙
sudo ufw status

# 4. 重启ROS节点
roscore
```

### Q4: XR-1推理延迟高

**症状**: 动作响应慢，卡顿

**优化方案**:
```bash
# 1. 启用TensorRT加速
pip install tensorrt
python scripts/optimize_tensorrt.py --checkpoint ./checkpoints/stage3_tiankung/best_model.pt

# 2. 降低推理精度（如需要）
# 在配置中设置: precision: "fp16"

# 3. 减少采样数量
# 在配置中设置: num_samples: 5

# 4. 监控GPU使用率
nvidia-smi dmon
```

---

## 快速参考卡片

### 常用命令速查

```bash
# DGX Spark管理
nvidia-smi                    # GPU状态
htop                          # 系统资源
docker ps                     # Docker容器
tmux ls                       # 查看会话
tmux attach -t xr1           # 连接会话

# ROS操作
rostopic list                 # 查看话题
rostopic echo /topic          # 监听话题
rosnode list                  # 查看节点

# XR-1操作
conda activate xr1           # 激活环境
cd ~/projects/XR-1           # 进入目录
python deploy/real_robot/xr1_deploy.py  # 启动部署

# 网络诊断
ip addr                       # 查看IP
ping 192.168.1.50            # 测试连通
ssh xr1-dev@192.168.1.100    # SSH连接
```

### 关键配置文件位置

```
~/projects/XR-1/                      # XR-1代码
~/projects/XR-1/configs/tiankung/     # 天工配置
~/projects/XR-1/checkpoints/          # 模型权重
~/projects/XR-1/data/                 # 数据集
~/.bashrc                             # 环境变量
/etc/docker/daemon.json               # Docker配置
```

---

**文档版本**: v1.0  
**适用硬件**: NVIDIA DGX Spark (GB10)  
**目标模型**: XR-1 VLA 3B  
**目标机器人**: 天工Pro (Tien Kung Pro)  
**最后更新**: 2025-01-30

---

**下一步**: 按照"第一阶段"开始DGX Spark的物理连接和首次启动配置。如遇到问题，参考"常见问题排查"章节或联系NVIDIA技术支持。
