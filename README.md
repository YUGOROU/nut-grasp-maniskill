# NutGrasp ManiSkill3

SO-101 robot arm grasping nuts (almond / ginko) — ManiSkill3 sim-to-real environment.

## Quick Start (Vast.ai)

```bash
bash setup_vastai.sh
source .venv/bin/activate
python train_ppo_rgbd.py --num-envs 512 --sim-backend gpu --total-timesteps 50_000_000 --use-wandb
```

## Local Test (Mac/CPU)

```bash
pip install -e ".[train]"
python train_ppo_rgbd.py --num-envs 1 --sim-backend cpu --total-timesteps 400 --num-minibatches 1
```

## Structure

```
assets/
  robot/          SO-101 URDF + STL/OBJ meshes + agent class
  objects/
    table/        Blender-exported OBJ + texture
    tray/         Steel tray OBJ
    bowls/        Glass bowl OBJs (almond / ginko)
    almond/       3D-scanned almond GLB
    ginko/        3D-scanned ginko GLB
envs/
  nut_grasp_env.py    NutGrasp-v1 task definition
configs/
  nut_grasp.yaml      Hyperparameter config
train_ppo_rgbd.py     PPO-RGBD training script (CleanRL-style)
setup_vastai.sh       One-shot Vast.ai setup
```

## Task

| Item | Detail |
|------|--------|
| Robot | SO-101 (6-DOF + gripper) |
| Obs | RGBD 64×64 + joint state |
| Action | pd_joint_pos (6-dim continuous) |
| Reward | Normalized dense (reach → lift → place) |
| Success | Nut within 5cm of target bowl |
