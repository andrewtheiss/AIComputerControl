# AI Agent Platform

This project provides a scalable platform for running multiple, isolated AI agents in Docker containers. Each agent operates within its own virtual desktop environment, using a centralized, GPU-accelerated inference API to perceive and interact with GUI applications.

## Architecture Overview

The platform consists of two primary services orchestrated by Docker Compose:

1.  **`inference` Service:** A single, powerful Triton Inference Server running on a dedicated GPU (e.g., RTX 6000). It exposes a gRPC endpoint that provides high-speed object detection (e.g., YOLOv8) for all running agents.
2.  **`agent` Service(s):** One or more containerized AI agents. Each agent is a complete sandbox environment with its own virtual XFCE desktop, a VNC server for remote viewing, and the core `agent.py` script that drives the capture-infer-act loop.

This architecture allows for efficient resource utilization by centralizing the heavy inference workload while enabling the lightweight agent instances to be scaled up easily.

---

## Prerequisites

Before you begin, ensure your host machine (Windows 11) is configured with the following:

-   **WSL2 (Windows Subsystem for Linux 2):** Properly installed and set as the default.
-   **Docker Desktop:** Installed and configured to use the WSL2 backend.
-   **NVIDIA GPU Driver:** A recent version (>= 550.xx) that supports CUDA in WSL2.
-   **NVIDIA Container Toolkit:** Installed to provide GPU access to Docker containers.
-   **VNC Viewer:** A client like [TigerVNC Viewer](https://tigervnc.org/) or [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) to connect to the agent desktops.

---

## One-Time Setup: Create VNC Password File

The `agent` container requires a pre-generated password file to secure its VNC server. This step only needs to be done once.

1.  **Install VNC Password Tool:**
    Open a terminal on your Linux host (or inside WSL2) and install the necessary tools.
    ```bash
    sudo apt-get update
    sudo apt-get install tigervnc-tools
    ```

2.  **Generate the Password File:**
    Navigate to the `AIAgent/agent/` directory and run the following command. It will prompt you to create a password. This password will be used by all agent instances.
    ```bash
    cd agent/
    vncpasswd passwd
    ```
    This creates a `passwd` file inside the `agent` directory, which will be copied into the Docker image during the build process.

---

## Recommended Usage: Docker Compose

Docker Compose is the simplest and most powerful way to manage the entire platform. These commands should be run from the root of the `AIAgent` project directory.

#### Build and Run All Services

This single command will build the Docker images for both the agent and the inference server, then start all services in the background.

```bash
docker-compose up --build -d
```

#### Connect to an Agent's Desktop

Use your VNC Viewer to connect to the agent instances. By default, the first agent is available at:

-   **Address:** `localhost:5901`
-   **Password:** The password you created during the one-time setup.

#### Scale the Number of Agents

To run multiple agents, use the `--scale` flag. For example, to run 3 agents:

```bash
docker-compose up --build -d --scale agent=3
```
Docker Compose will automatically manage ports and naming. You can connect to them at `localhost:5901`, `localhost:5902`, `localhost:5903`, etc.

#### View Logs

You can monitor the logs for any running service in real-time.

```bash
# View logs for the inference server
docker-compose logs -f inference

# View logs for the agent instances
docker-compose logs -f agent
```

#### Stop All Services

This command will gracefully stop and remove all containers, networks, and volumes associated with the platform.

```bash
docker-compose down
```

---

## Manual Usage (For Debugging and Single Instances)

For testing or debugging a single agent instance, you can use the standard Docker commands. These commands should be run from the `AIAgent/agent/` directory.

#### Build the Agent Container

This command builds the `ai-sandbox` image using the `Dockerfile` in the `agent` directory.

```bash
# Navigate to the agent directory
cd agent/

# Build the image
docker build -t ai-sandbox .
```

#### Run Agent Containers Manually

Run these commands from your host terminal to start one or more agent instances.

```bash
# Run a single agent, accessible on port 5901
docker run -d --rm --gpus 'device=0' --name vnc-instance-1 -p 5901:5901 ai-sandbox

# Run a second agent, accessible on port 5902
docker run -d --rm --gpus 'device=0' --name vnc-instance-2 -p 5902:5901 ai-sandbox

# Run a third agent, accessible on port 5903
docker run -d --rm --gpus 'device=0' --name vnc-instance-3 -p 5903:5901 ai-sandbox
```

#### Access a Running Container

To get a shell inside a specific container for debugging:

```bash
docker exec -it vnc-instance-1 /bin/bash

docker compose build --no-cache
docker compose up -d --force-recreate vnc-instance-1 vnc-instance-2 task-planner