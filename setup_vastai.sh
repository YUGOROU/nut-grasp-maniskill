#!/usr/bin/env bash
# ============================================================
# Vast.ai Setup Script for NutGrasp ManiSkill3 RL Training
#
# Usage (on Vast.ai instance):
#   bash setup_vastai.sh
#   python train_ppo_rgbd.py --num-envs 512 --total-timesteps 50_000_000 --use-wandb
# ============================================================
set -e

REPO_URL="https://github.com/YUGOROU/nut-grasp-maniskill.git"
WORKDIR="/workspace/nut-grasp-maniskill"

echo "=== [1/5] System packages ==="
apt-get update -qq && apt-get install -y -qq \
    git wget curl libvulkan1 vulkan-tools \
    libgl1-mesa-glx libegl1-mesa libgles2-mesa \
    > /dev/null

echo "=== [2/5] Clone repository ==="
if [ -d "$WORKDIR" ]; then
    echo "  Already cloned, pulling latest..."
    cd "$WORKDIR" && git pull
else
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

echo "=== [3/5] Python environment ==="
pip install uv -q
uv venv .venv --python 3.10
source .venv/bin/activate

# ManiSkill3 + dependencies
uv pip install -e ".[train]" -q

# Verify GPU sim backend
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "=== [4/5] Verify environment ==="
python - <<'EOF'
import sys
sys.path.insert(0, "envs")
sys.path.insert(0, "assets/robot")
import so101_agent, nut_grasp_env
import gymnasium as gym
env = gym.make("NutGrasp-v1", obs_mode="state", control_mode="pd_joint_pos", num_envs=1, sim_backend="gpu")
obs, _ = env.reset()
print("NutGrasp-v1 OK")
env.close()
EOF

echo "=== [5/5] Ready! ==="
echo ""
echo "Start training with:"
echo "  source .venv/bin/activate"
echo "  python train_ppo_rgbd.py \\"
echo "    --num-envs 512 \\"
echo "    --sim-backend gpu \\"
echo "    --total-timesteps 50_000_000 \\"
echo "    --use-wandb \\"
echo "    --wandb-entity YUGOROU"
echo ""
echo "Monitor: https://wandb.ai/YUGOROU/GYOZA-sim2real"
