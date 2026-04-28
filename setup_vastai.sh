#!/usr/bin/env bash
# ============================================================
# NutGrasp ManiSkill3 — Vast.ai Setup
#
# One-liner launch:
#   curl -fsSL https://raw.githubusercontent.com/YUGOROU/nut-grasp-maniskill/main/setup_vastai.sh | bash
# ============================================================
set -euo pipefail

REPO_URL="https://github.com/YUGOROU/nut-grasp-maniskill.git"
WORKDIR="/workspace/nut-grasp-maniskill"
PYTHON="python3.10"

echo "=== [1/5] System packages ==="
apt-get update -qq && apt-get install -y -qq \
    git git-lfs curl libvulkan1 vulkan-tools \
    libgl1-mesa-glx libegl1-mesa libgles2-mesa \
    python3.10 python3.10-venv \
    > /dev/null
git lfs install --system > /dev/null

echo "=== [2/5] uv ==="
# Install uv (fast Python package manager)
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv $(uv --version)"

echo "=== [3/5] Clone repository ==="
if [ -d "$WORKDIR/.git" ]; then
    echo "  Pulling latest..."
    git -C "$WORKDIR" pull --ff-only
else
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

echo "=== [4/5] Python environment ==="
uv venv .venv --python "$PYTHON"
source .venv/bin/activate

# Install all deps with uv (significantly faster than pip)
uv pip install -e ".[train]"

# Verify GPU
python - <<'PYEOF'
import torch
cuda = torch.cuda.is_available()
print(f"CUDA: {cuda}")
if cuda:
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF

echo "=== [5/5] Smoke test (RGBD, 4 envs) ==="
python - <<'PYEOF'
import sys
sys.path.insert(0, "envs")
sys.path.insert(0, "assets/robot")
import so101_agent, nut_grasp_env
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
env = gym.make("NutGrasp-v1", obs_mode="rgbd", control_mode="pd_joint_pos",
               reward_mode="normalized_dense",
               num_envs=4, sim_backend="gpu",
               sensor_configs=dict(width=64, height=64))
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True, sep_depth=True)
obs, _ = env.reset()
action = env.action_space.sample()
import torch
obs2, rew, term, trunc, info = env.step(torch.tensor(action))
print(f"NutGrasp-v1 RGBD OK")
print(f"  rgb={obs['rgb'].shape} depth={obs['depth'].shape} state={obs['state'].shape}")
print(f"  reward={rew.mean():.4f}")
env.close()
PYEOF

echo ""
echo "=============================="
echo "  Setup complete. Run:"
echo "=============================="
echo "  cd $WORKDIR"
echo "  source .venv/bin/activate"
echo "  python train_ppo_rgbd.py \\"
echo "    --num-envs 2048 \\"
echo "    --sim-backend gpu \\"
echo "    --total-timesteps 50000000 \\"
echo "    --save-freq 50 \\"
echo "    --use-wandb"
echo ""
echo "  # Resume from checkpoint:"
echo "  python train_ppo_rgbd.py \\"
echo "    --num-envs 2048 --sim-backend gpu \\"
echo "    --total-timesteps 50000000 \\"
echo "    --resume checkpoints/<run_name>/ckpt_XXXXXX.pt \\"
echo "    --use-wandb"
echo ""
