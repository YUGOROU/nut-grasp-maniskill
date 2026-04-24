"""
PPO training for NutGrasp-v1 with RGBD observations.
CleanRL-style single-file implementation with ManiSkill3 GPU parallel envs.

Usage (Vast.ai, 512 envs):
  python train_ppo_rgbd.py --num-envs 512 --total-timesteps 50_000_000

Usage (local test, CPU):
  python train_ppo_rgbd.py --num-envs 8 --sim-backend physx --total-timesteps 100_000
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# ── ManiSkill3 env registration ──────────────────────────────────────────────
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "envs"))
sys.path.insert(0, os.path.join(_HERE, "assets", "robot"))
import so101_agent   # noqa: F401 – triggers @register_agent
import nut_grasp_env  # noqa: F401 – triggers @register_env

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


# ── Hyper-parameters ─────────────────────────────────────────────────────────

@dataclass
class Args:
    # Env
    env_id:            str   = "NutGrasp-v1"
    nut_type:          str   = "almond"        # almond | ginko
    num_envs:          int   = 512
    sim_backend:       str   = "gpu"            # gpu (Vast.ai) | cpu (local test)
    capture_video:     bool  = False

    # PPO
    total_timesteps:   int   = 50_000_000
    learning_rate:     float = 3e-4
    num_steps:         int   = 50              # steps per rollout per env
    gamma:             float = 0.8             # short horizon for manipulation
    gae_lambda:        float = 0.9
    num_minibatches:   int   = 32
    update_epochs:     int   = 4
    clip_coef:         float = 0.2
    ent_coef:          float = 0.0
    vf_coef:           float = 0.5
    max_grad_norm:     float = 0.5
    norm_adv:          bool  = True
    clip_vloss:        bool  = True
    target_kl:         Optional[float] = 0.1

    # Network
    img_channels:      int   = 4              # RGB(3) + D(1)
    cnn_out_dim:       int   = 256
    hidden_dim:        int   = 512

    # Logging
    exp_name:          str   = "nut_grasp_ppo_rgbd"
    wandb_project:     str   = "GYOZA-sim2real"
    wandb_entity:      str   = "YUGOROU"
    use_wandb:         bool  = False
    save_freq:         int   = 100            # save every N updates
    log_freq:          int   = 1

    # Derived (set after init)
    batch_size:        int   = 0
    minibatch_size:    int   = 0


def parse_args() -> Args:
    args = Args()
    parser = argparse.ArgumentParser()
    for field, val in args.__dict__.items():
        t = type(val) if val is not None else str
        if t == bool:
            parser.add_argument(f"--{field.replace('_','-')}",
                                action="store_true", default=val)
        else:
            parser.add_argument(f"--{field.replace('_','-')}",
                                type=t, default=val)
    parsed = parser.parse_args()
    for field in args.__dict__:
        setattr(args, field, getattr(parsed, field))
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


# ── Network ──────────────────────────────────────────────────────────────────

class NatureCNN(nn.Module):
    """DQN-style CNN encoder for stacked RGBD frames."""
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),          nn.ReLU(),
            nn.Flatten(),
        )
        # Infer conv output size with a dummy tensor (84x84 assumed; will adapt)
        self._out_dim = out_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return self.net(x)

    def _conv_out(self, h: int, w: int, device) -> int:
        dummy = torch.zeros(1, self._in_channels, h, w, device=device)
        return self.net(dummy).shape[1]


class ActorCritic(nn.Module):
    def __init__(self, img_channels: int, state_dim: int,
                 cnn_out_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()

        # CNN for RGBD images (shared trunk)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),           nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),           nn.ReLU(),
            nn.Flatten(),
        )

        # We'll compute conv_flat_dim lazily on first forward pass
        self._conv_flat_dim: Optional[int] = None
        self._img_channels = img_channels
        self.cnn_proj: Optional[nn.Linear] = None  # built lazily

        # State MLP
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )

        feat_dim = cnn_out_dim + 128
        self.actor_mean = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cnn_out_dim = cnn_out_dim

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def _encode(self, rgb: torch.Tensor, depth: torch.Tensor,
                state: torch.Tensor):
        # rgb: (B, H, W, 3) uint8 | depth: (B, H, W, 1) float
        img = torch.cat([
            rgb.float() / 255.0,
            depth.float(),
        ], dim=-1)  # (B, H, W, 4)
        img = img.permute(0, 3, 1, 2)  # (B, 4, H, W)

        conv_feat = self.cnn(img)

        # Build projection layer on first pass
        if self._conv_flat_dim is None:
            self._conv_flat_dim = conv_feat.shape[1]
            self.cnn_proj = nn.Linear(
                self._conv_flat_dim, self.cnn_out_dim
            ).to(conv_feat.device)
            nn.init.orthogonal_(self.cnn_proj.weight, np.sqrt(2))
            nn.init.zeros_(self.cnn_proj.bias)

        img_feat   = torch.relu(self.cnn_proj(conv_feat))   # (B, cnn_out_dim)
        state_feat = self.state_enc(state)                   # (B, 128)
        return torch.cat([img_feat, state_feat], dim=-1)     # (B, feat_dim)

    def get_value(self, rgb, depth, state):
        return self.critic(self._encode(rgb, depth, state))

    def get_action_and_value(self, rgb, depth, state, action=None):
        feat = self._encode(rgb, depth, state)
        mean = self.actor_mean(feat)
        std  = self.actor_logstd.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(feat)
        return action, log_prob, entropy, value


# ── Training loop ─────────────────────────────────────────────────────────────

def make_env(args: Args):
    env = gym.make(
        args.env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        reward_mode="normalized_dense",
        nut_type=args.nut_type,
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        sensor_configs=dict(width=64, height=64),  # 64x64 for efficiency
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True,
                                        state=True, sep_depth=True)
    return env


def train(args: Args):
    run_name = f"{args.exp_name}__{int(time.time())}"
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | run: {run_name}")

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=False,
        )

    # ── Env ──
    env = make_env(args)
    obs_sample, _ = env.reset()
    rgb_sample   = obs_sample["rgb"]    # (N, H, W, 3*n_cams)
    depth_sample = obs_sample["depth"]  # (N, H, W, 1*n_cams)
    state_sample = obs_sample["state"]  # (N, state_dim)

    H, W       = rgb_sample.shape[1], rgb_sample.shape[2]
    n_cams_rgb = rgb_sample.shape[3] // 3
    n_cams_d   = depth_sample.shape[3]
    img_ch     = n_cams_rgb * 3 + n_cams_d   # total channels after concat
    state_dim  = state_sample.shape[1]
    # single_action_space gives (action_dim,), action_space gives (num_envs, action_dim)
    action_dim = env.unwrapped.single_action_space.shape[0]

    print(f"Obs: rgb={rgb_sample.shape} depth={depth_sample.shape} "
          f"state={state_sample.shape}")
    print(f"img_ch={img_ch} state_dim={state_dim} action_dim={action_dim}")

    # ── Model ──
    agent = ActorCritic(img_ch, state_dim, args.cnn_out_dim,
                        args.hidden_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ── Rollout buffers (on device) ──
    rb_rgb   = torch.zeros(args.num_steps, args.num_envs, H, W,
                           n_cams_rgb * 3, dtype=torch.uint8, device=device)
    rb_depth = torch.zeros(args.num_steps, args.num_envs, H, W,
                           n_cams_d, dtype=torch.float32, device=device)
    rb_state  = torch.zeros(args.num_steps, args.num_envs, state_dim, device=device)
    rb_action = torch.zeros(args.num_steps, args.num_envs, action_dim, device=device)
    rb_logp   = torch.zeros(args.num_steps, args.num_envs, device=device)
    rb_reward = torch.zeros(args.num_steps, args.num_envs, device=device)
    rb_done   = torch.zeros(args.num_steps, args.num_envs, device=device)
    rb_value  = torch.zeros(args.num_steps, args.num_envs, device=device)

    # ── Main loop ──
    global_step      = 0
    start_time       = time.time()
    num_updates      = args.total_timesteps // args.batch_size
    save_dir         = os.path.join(_HERE, "checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)

    obs, _  = env.reset()
    rgb     = obs["rgb"].to(device)
    depth   = obs["depth"].to(device)
    state   = obs["state"].to(device)
    done    = torch.zeros(args.num_envs, device=device)

    for update in range(1, num_updates + 1):
        # Linear LR annealing
        frac = 1.0 - (update - 1) / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ── Rollout collection ──
        for step in range(args.num_steps):
            global_step += args.num_envs
            rb_rgb[step]   = rgb
            rb_depth[step] = depth
            rb_state[step] = state
            rb_done[step]  = done

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(
                    rgb, depth, state)
            rb_action[step] = action
            rb_logp[step]   = logp
            rb_value[step]  = value.flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            done = (terminated | truncated).float()
            rb_reward[step] = reward.to(device)

            rgb   = obs["rgb"].to(device)
            depth = obs["depth"].to(device)
            state = obs["state"].to(device)

        # ── Compute GAE ──
        with torch.no_grad():
            next_value = agent.get_value(rgb, depth, state).flatten()
            advantages = torch.zeros_like(rb_reward)
            last_gae   = 0.0
            for t in reversed(range(args.num_steps)):
                nv    = next_value if t == args.num_steps - 1 else rb_value[t + 1]
                nd    = done       if t == args.num_steps - 1 else rb_done[t + 1]
                delta = rb_reward[t] + args.gamma * nv * (1 - nd) - rb_value[t]
                last_gae = delta + args.gamma * args.gae_lambda * (1 - nd) * last_gae
                advantages[t] = last_gae
            returns = advantages + rb_value

        # ── Flatten batch ──
        b_rgb   = rb_rgb.reshape(-1, H, W, n_cams_rgb * 3)
        b_depth = rb_depth.reshape(-1, H, W, n_cams_d)
        b_state = rb_state.reshape(-1, state_dim)
        b_act   = rb_action.reshape(-1, action_dim)
        b_logp  = rb_logp.reshape(-1)
        b_val   = rb_value.reshape(-1)
        b_adv   = advantages.reshape(-1)
        b_ret   = returns.reshape(-1)
        if args.norm_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ── PPO update ──
        inds = np.arange(args.batch_size)
        pg_losses, v_losses, ent_losses, approx_kls = [], [], [], []
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = inds[start: start + args.minibatch_size]
                _, new_logp, entropy, new_val = agent.get_action_and_value(
                    b_rgb[mb], b_depth[mb], b_state[mb], b_act[mb])

                log_ratio  = new_logp - b_logp[mb]
                ratio      = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    approx_kls.append(approx_kl.item())

                mb_adv = b_adv[mb]
                pg1    = -mb_adv * ratio
                pg2    = -mb_adv * ratio.clamp(1 - args.clip_coef,
                                               1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                pg_losses.append(pg_loss.item())

                new_val = new_val.flatten()
                if args.clip_vloss:
                    v_unclip = (new_val - b_ret[mb]) ** 2
                    v_clip   = (b_val[mb] + (new_val - b_val[mb]).clamp(
                                    -args.clip_coef, args.clip_coef) - b_ret[mb]) ** 2
                    v_loss   = 0.5 * torch.max(v_unclip, v_clip).mean()
                else:
                    v_loss   = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()
                v_losses.append(v_loss.item())

                ent_loss = entropy.mean()
                ent_losses.append(ent_loss.item())

                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl and np.mean(approx_kls) > args.target_kl:
                break

        # ── Logging ──
        if update % args.log_freq == 0:
            sps      = int(global_step / (time.time() - start_time))
            ep_rew   = rb_reward.sum(0).mean().item()
            success  = info.get("success", torch.zeros(1))
            suc_rate = float(success.float().mean()) if hasattr(success, "float") else 0.0
            print(
                f"update={update}/{num_updates} "
                f"steps={global_step:,} "
                f"sps={sps:,} "
                f"ep_rew={ep_rew:.3f} "
                f"success={suc_rate:.2%} "
                f"pg={np.mean(pg_losses):.4f} "
                f"v={np.mean(v_losses):.4f} "
                f"kl={np.mean(approx_kls):.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            if args.use_wandb:
                import wandb
                wandb.log({
                    "charts/sps":            sps,
                    "charts/ep_reward":      ep_rew,
                    "charts/success_rate":   suc_rate,
                    "losses/policy":         np.mean(pg_losses),
                    "losses/value":          np.mean(v_losses),
                    "losses/entropy":        np.mean(ent_losses),
                    "losses/approx_kl":      np.mean(approx_kls),
                    "charts/learning_rate":  optimizer.param_groups[0]["lr"],
                }, step=global_step)

        # ── Checkpoint ──
        if update % args.save_freq == 0:
            ckpt_path = os.path.join(save_dir, f"ckpt_{update:06d}.pt")
            torch.save({
                "update":       update,
                "global_step":  global_step,
                "model":        agent.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "args":         vars(args),
            }, ckpt_path)
            print(f"  [saved] {ckpt_path}")

    env.close()
    print(f"Training complete. Total steps: {global_step:,}")


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"NutGrasp PPO-RGBD  |  envs={args.num_envs}  "
          f"backend={args.sim_backend}  total={args.total_timesteps:,}")
    print(f"batch={args.batch_size}  minibatch={args.minibatch_size}  "
          f"updates={args.total_timesteps // args.batch_size}")
    print("=" * 60)
    train(args)
