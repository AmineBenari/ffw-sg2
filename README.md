# ffw-sg2

CuRobo + MuJoCo motion planning and control stack for the ROBOTIS FFW-SG2 (AI Worker).

## Architecture

```
Goal pose → CuRobo (plan) → Joint trajectory → MuJoCo (sim) + Real Robot (DYNAMIXEL via ROS 2)
```

This container handles motion planning and simulation. Robot hardware control is handled by
the [ROBOTIS ai_worker](https://github.com/ROBOTIS-GIT/ai_worker) container. Both communicate
over ROS 2 Jazzy (DDS, `--network host`).

## Requirements

- NVIDIA GPU (Volta or newer, 4 GB+ VRAM)
- Docker with NVIDIA Container Toolkit
- Ubuntu 22.04 / 24.04

## Setup

### 1. Install NVIDIA Container Toolkit (host, one-time)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Build the container

```bash
cd docker
docker compose build
```

> First build takes ~20 minutes due to CuRobo's CUDA compilation.

### 3. Run

```bash
docker compose run ffw-sg2-planner
```

## Robot Assets

Robot URDF and MuJoCo MJCF files go in `robot/`. Clone from ROBOTIS:

```bash
# URDF + ROS 2 packages
git clone https://github.com/ROBOTIS-GIT/ai_worker.git

# MuJoCo MJCF model
git clone https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie.git
```

## Real Robot Control

Start the ROBOTIS hardware container alongside this one (same host, `network_mode: host`):

```bash
# From the ai_worker repo
docker compose up
```

Both containers share the same ROS 2 DDS network automatically.
