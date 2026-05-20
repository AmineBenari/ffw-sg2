# Base: CUDA 12.8.1 on Ubuntu 24.04 (matches ROS 2 Jazzy)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# System dependencies
RUN apt-get update && apt-get install -y \
    curl gnupg lsb-release software-properties-common \
    python3 python3-pip python3-dev \
    git build-essential cmake \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Jazzy
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y \
    ros-jazzy-ros-base \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-joint-state-publisher \
    python3-rosdep \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.8 support (matching base image)
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# MuJoCo
RUN pip3 install --no-cache-dir mujoco

# CuRobo (build from source — takes ~20 min)
RUN git clone https://github.com/NVlabs/curobo.git /opt/curobo && \
    cd /opt/curobo && \
    pip3 install -e ".[dev]" --no-build-isolation

# CuRobo 0.8.x requires cuda-core[cu12] for its cuda.core backend
RUN pip3 install --no-cache-dir "cuda-core[cu12]>=0.7"

# Source ROS 2 on shell startup
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /workspace
