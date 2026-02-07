# Docker化部署方案 - DGX Spark + XR-1

> **建议**: 对于DGX Spark + XR-1开发，**强烈建议使用Docker部署**。DGX Spark原生支持NVIDIA Container Runtime，Docker可以提供更好的环境隔离和可重复性。

---

## 🤔 为什么使用Docker？

### 使用Docker的优势

| 优势 | 说明 | 适用场景 |
|------|------|----------|
| **环境隔离** | XR-1依赖复杂，Docker避免污染系统环境 | 开发/测试 |
| **可重复性** | 一次构建，到处运行，团队协作更顺畅 | 团队开发 |
| **版本管理** | 轻松切换XR-1版本、PyTorch版本、CUDA版本 | 实验对比 |
| **快速恢复** | 容器损坏可秒级重建，数据通过volume持久化 | 生产环境 |
| **资源限制** | 可限制容器内存/CPU，避免影响其他服务 | 多任务并行 |
| **易于备份** | 镜像可导出，环境配置可版本控制 | 环境迁移 |

### DGX Spark + Docker = 完美组合

DGX Spark已经**预装NVIDIA Container Runtime**：
- ✅ GPU直通支持（`--gpus=all`）
- ✅ CUDA/cuDNN已集成
- ✅ 128GB统一内存充足
- ✅ 官方推荐容器化部署

---

## 📁 推荐的Docker架构

### 多容器架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    DGX Spark (Host)                         │
│                   Ubuntu 22.04 + DGX OS                     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  xr1-dev     │    │  xr1-train   │    │  xr1-deploy  │
│  (开发环境)   │    │  (训练环境)   │    │  (部署环境)   │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ • JupyterLab │    │ • 无GUI      │    │ • ROS节点    │
│ • VSCode     │    │ • tmux       │    │ • 推理服务   │
│ • 调试工具   │    │ • 后台训练   │    │ • 生产优化   │
│ • 代码编辑   │    │ • 日志记录   │    │ • 低延迟     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Data Volume    │
                    │  (数据持久化)     │
                    ├──────────────────┤
                    │ • XR-1代码       │
                    │ • 数据集         │
                    │ • 模型权重       │
                    │ • 配置文件       │
                    │ • 日志文件       │
                    └──────────────────┘
```

### 容器职责分离

| 容器 | 用途 | 特点 | 启动时机 |
|------|------|------|----------|
| **xr1-dev** | 日常开发 | 带JupyterLab、VSCode Server | 开发时 |
| **xr1-train** | 模型训练 | 后台运行、资源优化 | 训练时 |
| **xr1-deploy** | 生产部署 | ROS集成、实时推理 | 部署时 |
| **xr1-data** | 数据收集 | VR/手柄遥操作支持 | 收集数据时 |

---

## 🛠️ Docker环境配置

### 1. 创建项目目录结构

```bash
# 在DGX Spark上执行
ssh spark

# 创建Docker项目目录
mkdir -p ~/docker/xr1/{dev,train,deploy,data}
cd ~/docker/xr1

# 创建共享数据目录
mkdir -p ~/projects/XR-1-docker/{data,checkpoints,logs,configs}
```

### 2. 编写Dockerfile

#### 基础镜像 Dockerfile.base

```dockerfile
# ~/docker/xr1/Dockerfile.base
FROM nvcr.io/nvidia/pytorch:24.12-py3

LABEL maintainer="xr1-dev-team"
LABEL description="XR-1 VLA Base Image for DGX Spark"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    net-tools \
    iputils-ping \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:${PATH}

# 创建conda环境
RUN conda create -n xr1 python=3.10 -y \
    && conda clean -afy

# 设置工作目录
WORKDIR /workspace

# 默认使用conda环境
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
```

#### 开发环境 Dockerfile.dev

```dockerfile
# ~/docker/xr1/dev/Dockerfile
FROM xr1-base:latest

# 安装开发工具
RUN apt-get update && apt-get install -y \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# 安装JupyterLab
RUN /opt/conda/bin/conda run -n xr1 pip install \
    jupyterlab \
    jupyterlab-git \
    jupyterlab-code-formatter \
    black \
    isort \
    nb_conda_kernels

# 安装VSCode Server (code-server)
RUN curl -fsSL https://code-server.dev/install.sh | sh

# 配置SSH
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd  # 生产环境请修改
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# 暴露端口
EXPOSE 22 8888 8080

# 启动脚本
COPY start-dev.sh /start-dev.sh
RUN chmod +x /start-dev.sh

CMD ["/start-dev.sh"]
```

#### 训练环境 Dockerfile.train

```dockerfile
# ~/docker/xr1/train/Dockerfile
FROM xr1-base:latest

# 安装训练优化工具
RUN /opt/conda/bin/conda run -n xr1 pip install \
    wandb \
    tensorboard \
    mlflow \
    torch-tb-profiler \
    nvitop

# 设置训练工作目录
WORKDIR /workspace/XR-1

# 训练启动脚本
COPY start-train.sh /start-train.sh
RUN chmod +x /start-train.sh

CMD ["/start-train.sh"]
```

#### 部署环境 Dockerfile.deploy

```dockerfile
# ~/docker/xr1/deploy/Dockerfile
FROM xr1-base:latest

# 安装ROS2 Humble（用于天工Pro通信）
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-desktop \
    && rm -rf /var/lib/apt/lists/*

# 配置ROS2环境
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 安装部署优化工具
RUN /opt/conda/bin/conda run -n xr1 pip install \
    fastapi \
    uvicorn \
    redis \
    paho-mqtt

# 暴露API端口
EXPOSE 8000

# 部署启动脚本
COPY start-deploy.sh /start-deploy.sh
RUN chmod +x /start-deploy.sh

CMD ["/start-deploy.sh"]
```

### 3. 编写启动脚本

#### 开发环境启动脚本 start-dev.sh

```bash
#!/bin/bash
# ~/docker/xr1/dev/start-dev.sh

echo "🚀 启动XR-1开发环境..."

# 启动SSH服务
/usr/sbin/sshd

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate xr1

# 启动JupyterLab
echo "📓 启动JupyterLab (端口8888)..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --notebook-dir=/workspace &

# 启动VSCode Server
echo "💻 启动VSCode Server (端口8080)..."
code-server --bind-addr 0.0.0.0:8080 --auth none /workspace &

echo "✅ 开发环境已启动！"
echo "📓 JupyterLab: http://localhost:8888"
echo "💻 VSCode Server: http://localhost:8080"
echo "🔧 SSH: ssh -p 2222 root@localhost"

# 保持容器运行
tail -f /dev/null
```

#### 训练环境启动脚本 start-train.sh

```bash
#!/bin/bash
# ~/docker/xr1/train/start-train.sh

echo "🚀 启动XR-1训练环境..."

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate xr1

# 设置环境变量
export PYTHONPATH=/workspace/XR-1:$PYTHONPATH
export WANDB_MODE=offline  # 离线模式，后续可上传

echo "✅ 训练环境已就绪"
echo "💡 使用tmux启动训练: tmux new -s training"
echo "📊 查看日志: tail -f /workspace/logs/training.log"

# 进入工作目录
cd /workspace/XR-1

# 启动bash
/bin/bash
```

#### 部署环境启动脚本 start-deploy.sh

```bash
#!/bin/bash
# ~/docker/xr1/deploy/start-deploy.sh

echo "🚀 启动XR-1部署环境..."

# 激活conda环境
source /opt/conda/etc/profile.d/conda.sh
conda activate xr1

# 配置ROS2
source /opt/ros/humble/setup.bash

# 设置环境变量
export ROS_MASTER_URI=http://192.168.1.50:11311
export ROS_IP=172.17.0.2  # Docker容器IP，需根据实际情况调整

echo "✅ 部署环境已就绪"
echo "🤖 启动推理服务: python deploy/real_robot/xr1_deploy.py"
echo "🌐 API服务: http://localhost:8000"

# 进入工作目录
cd /workspace/XR-1

# 启动bash
/bin/bash
```

---

## 🐳 Docker Compose配置

### docker-compose.yml

```yaml
# ~/docker/xr1/docker-compose.yml
version: '3.8'

services:
  # 开发环境
  xr1-dev:
    build:
      context: .
      dockerfile: dev/Dockerfile
    image: xr1-dev:latest
    container_name: xr1-dev
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    ports:
      - "8888:8888"    # JupyterLab
      - "8080:8080"    # VSCode Server
      - "2222:22"      # SSH
    volumes:
      - ~/projects/XR-1-docker:/workspace
      - xr1-conda:/opt/conda/envs/xr1  # 持久化conda环境
    networks:
      - xr1-network
    stdin_open: true
    tty: true
    command: /start-dev.sh

  # 训练环境
  xr1-train:
    build:
      context: .
      dockerfile: train/Dockerfile
    image: xr1-train:latest
    container_name: xr1-train
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      - ~/projects/XR-1-docker:/workspace
      - xr1-conda:/opt/conda/envs/xr1
    networks:
      - xr1-network
    stdin_open: true
    tty: true
    command: /start-train.sh

  # 部署环境
  xr1-deploy:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    image: xr1-deploy:latest
    container_name: xr1-deploy
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    ports:
      - "8000:8000"    # API服务
    volumes:
      - ~/projects/XR-1-docker:/workspace
      - xr1-conda:/opt/conda/envs/xr1
    networks:
      - xr1-network
      - ros-network    # ROS网络
    stdin_open: true
    tty: true
    command: /start-deploy.sh

  # 数据收集环境（可选）
  xr1-data:
    build:
      context: .
      dockerfile: dev/Dockerfile  # 复用开发环境
    image: xr1-dev:latest
    container_name: xr1-data
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      - ~/projects/XR-1-docker:/workspace
      - /dev/bus/usb:/dev/bus/usb  # USB设备（VR/手柄）
    networks:
      - xr1-network
      - ros-network
    privileged: true  # 需要特权模式访问USB
    stdin_open: true
    tty: true

volumes:
  xr1-conda:
    driver: local

networks:
  xr1-network:
    driver: bridge
  ros-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## 🚀 使用流程

### 1. 构建镜像

```bash
# 登录DGX Spark
ssh spark

# 进入Docker项目目录
cd ~/docker/xr1

# 构建基础镜像
docker build -t xr1-base:latest -f Dockerfile.base .

# 构建开发镜像
docker build -t xr1-dev:latest -f dev/Dockerfile dev/

# 构建训练镜像
docker build -t xr1-train:latest -f train/Dockerfile train/

# 构建部署镜像
docker build -t xr1-deploy:latest -f deploy/Dockerfile deploy/

# 或使用docker-compose一键构建
docker-compose build
```

### 2. 启动容器

```bash
# 启动开发环境
docker-compose up -d xr1-dev

# 启动训练环境
docker-compose up -d xr1-train

# 启动部署环境
docker-compose up -d xr1-deploy

# 查看运行状态
docker-compose ps
```

### 3. 进入容器

```bash
# 进入开发容器
docker exec -it xr1-dev bash

# 进入训练容器
docker exec -it xr1-train bash

# 进入部署容器
docker exec -it xr1-deploy bash
```

### 4. 在容器中操作

```bash
# 激活conda环境
conda activate xr1

# 进入XR-1目录
cd /workspace/XR-1

# 执行XR-1命令
python deploy/real_robot/xr1_deploy.py --config configs/tiankung/tiankung_pro.yaml
```

---

## 📊 数据持久化方案

### 数据映射关系

| 主机路径 | 容器路径 | 用途 | 备份建议 |
|----------|----------|------|----------|
| `~/projects/XR-1-docker/data` | `/workspace/data` | 数据集 | 定期备份 |
| `~/projects/XR-1-docker/checkpoints` | `/workspace/checkpoints` | 模型权重 | 必须备份 |
| `~/projects/XR-1-docker/logs` | `/workspace/logs` | 训练日志 | 可选备份 |
| `~/projects/XR-1-docker/configs` | `/workspace/configs` | 配置文件 | 版本控制 |
| Docker Volume `xr1-conda` | `/opt/conda/envs/xr1` | Conda环境 | 无需备份 |

### 备份脚本

```bash
#!/bin/bash
# ~/docker/xr1/backup.sh

BACKUP_DIR="/mnt/backup/xr1-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 备份数据
tar -czf $BACKUP_DIR/data.tar.gz ~/projects/XR-1-docker/data/

# 备份模型
tar -czf $BACKUP_DIR/checkpoints.tar.gz ~/projects/XR-1-docker/checkpoints/

# 备份配置
tar -czf $BACKUP_DIR/configs.tar.gz ~/projects/XR-1-docker/configs/

# 导出镜像列表
docker images > $BACKUP_DIR/docker-images.txt

# 保留最近7个备份
ls -td /mnt/backup/xr1-* | tail -n +8 | xargs rm -rf

echo "✅ 备份完成: $BACKUP_DIR"
```

---

## 🔧 高级配置

### 1. GPU资源分配

```yaml
# 限制特定GPU
services:
  xr1-train:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. 内存限制

```yaml
services:
  xr1-train:
    deploy:
      resources:
        limits:
          memory: 64G
        reservations:
          memory: 32G
```

### 3. ROS网络配置

```bash
# 查看容器IP
docker inspect xr1-deploy | grep IPAddress

# 配置ROS环境变量（根据实际IP调整）
export ROS_IP=172.20.0.4
export ROS_MASTER_URI=http://192.168.1.50:11311
```

---

## 🆚 Docker vs 原生部署对比

| 维度 | Docker部署 | 原生部署 | 建议 |
|------|-----------|----------|------|
| **环境隔离** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Docker胜 |
| **启动速度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 原生胜 |
| **资源占用** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 原生胜（多5-10%） |
| **可重复性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Docker胜 |
| **团队协作** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Docker胜 |
| **调试难度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 原生胜 |
| **ROS集成** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 原生胜 |
| **备份恢复** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Docker胜 |

**建议**: 
- **开发阶段**: 使用Docker（环境隔离、易于协作）
- **生产部署**: 可考虑原生部署（性能最优、ROS集成更好）
- **团队开发**: 必须使用Docker（环境一致性）

---

## ⚡ 快速开始（推荐方案）

### 方案A: 纯Docker部署（推荐新手）

```bash
# 1. 登录DGX Spark
ssh spark

# 2. 克隆配置仓库
git clone https://github.com/your-repo/xr1-docker.git ~/docker/xr1
cd ~/docker/xr1

# 3. 构建并启动
docker-compose up -d xr1-dev

# 4. 进入容器
docker exec -it xr1-dev bash

# 5. 激活环境
conda activate xr1
cd /workspace/XR-1

# 6. 开始开发！
```

### 方案B: 混合部署（推荐生产）

```bash
# 开发: 使用Docker
docker-compose up -d xr1-dev
docker exec -it xr1-dev bash

# 训练: 使用Docker（后台）
docker-compose up -d xr1-train
docker exec -it xr1-train bash
tmux new -s training
bash scripts/xr1_stage3_finetune.sh

# 部署: 原生部署（性能最优）
ssh spark
conda activate xr1
cd ~/projects/XR-1
python deploy/real_robot/xr1_deploy.py
```

---

## 📝 总结建议

### 使用Docker的场景
✅ **推荐Docker**:
- 团队协作开发
- 需要环境隔离
- 频繁切换XR-1版本
- 需要快速恢复环境
- 多项目并行开发

### 使用原生部署的场景
✅ **推荐原生**:
- 追求极致性能
- ROS集成复杂
- 单用户长期使用
- 硬件资源紧张

### 最终建议

对于你的场景（天工Pro + XR-1 + DGX Spark），建议采用**混合方案**:

1. **开发阶段**: 使用 `xr1-dev` Docker容器（带JupyterLab和VSCode）
2. **训练阶段**: 使用 `xr1-train` Docker容器（后台运行，资源优化）
3. **部署阶段**: 原生部署（ROS集成更好，延迟更低）

这样既享受Docker的环境管理优势，又保证生产部署的性能。

---

需要我帮你创建完整的Docker配置文件，或者详细解释某个部分吗？
