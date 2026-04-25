# CLAUDE.md — NutGrasp ManiSkill3 (Project GYOZA)

SO-101ロボットアームでナッツ（アーモンド/銀杏）を把持するManiSkill3強化学習環境。
目標: sim-to-real転送でLeRobotと実機デプロイ。

## リポジトリ

- GitHub: `YUGOROU/nut-grasp-maniskill` (branch: main)
- HuggingFace checkpoints: `YUGOROU/gyoza-nutgrasp-checkpoints`
- Vast.ai SSH key: `~/.ssh/vast_ai`

## プロジェクト構造

```
assets/
  robot/          so101.urdf, so101_agent.py, meshes/
  objects/
    table/        table.obj + texture
    tray/         tray.obj (255×202×26mm)
    bowls/        almond_bowl.obj, ginko_bowl.obj
    almond/       almond_single.glb
    ginko/        Ginko_single.glb
envs/
  nut_grasp_env.py    NutGrasp-v1 タスク定義
configs/
  nut_grasp.yaml
train_ppo_rgbd.py     PPO-RGBD 学習スクリプト（CleanRL-style）
setup_vastai.sh       Vast.ai ワンショットセットアップ
```

## 環境仕様 (NutGrasp-v1)

| 項目 | 詳細 |
|------|------|
| ロボット | SO-101（5-DOF arm + 1-DOF gripper） |
| 観測 | RGBD 64×64 + joint state（FlattenRGBDObservationWrapper） |
| アクション | pd_joint_pos（6次元: arm5 + gripper1） |
| 報酬 | normalized_dense（reach+grasp×2+lift×2+place×3 = max8, /8） |
| 成功条件 | ナット ≤ 5cm from target bowl, z > 0.005m |

### グリッパー仕様
- joint名: `gripper`、lower=0.0m（閉じ）、upper=0.04m（開き）
- normalize_action=True: action[-1]=-1→閉じ, +1→開き
- qpos[-1]で閉開度を読み取り

### 報酬関数 (envs/nut_grasp_env.py:158)
```python
reach_rew   = 1 - tanh(5 * ||tcp - nut||)              # stage1
grasp_rew   = (1 - gripper_open) * reach_rew            # stage2
lift_rew    = tanh(10 * lift_height)                    # stage3
place_rew   = 1 - tanh(5 * ||nut - bowl||)              # stage4
reward = reach_rew + grasp_rew*2 + lift_rew*2 + place_rew*3  # max=8
```

### シーンレイアウト（world frame, Z-up）
- テーブル面: z=0
- トレイ中心: (0, 0, 0.013), half=(0.1275, 0.101, 0.013)
- アーモンドボウル: (-0.18, 0, 0.03)、銀杏ボウル: (0.18, 0, 0.03)
- ロボットベース: (-0.35, 0, 0.02)

## ネットワーク構造 (train_ppo_rgbd.py:130)

```
ActorCritic
├── cnn (NatureCNN: Conv32/8s4 → Conv64/4s2 → Conv64/3s1 → Flatten)
├── cnn_proj (lazy init, 64×64入力→conv_flat_dim=1024→cnn_out_dim=256)
├── state_enc (Linear→128→ReLU→128→ReLU)
├── actor_mean (Linear 384→512→action_dim=6)
├── actor_logstd (Parameter, zeros init)
└── critic (Linear 384→512→1) ← feat.detach()で更新
```

**重要バグ修正済み（再導入禁止）:**
1. `cnn_proj`はoptimizer生成前にダミーforwardでlazy init → optimizerに登録される
2. target_klチェックはper-minibatch、backward前にbreak
3. 深度は`clamp(0,10)/10`で正規化（生meters=最大9899mがKL爆発の原因だった）
4. criticはfeat.detach()使用（shared encoderへのv_loss勾配遮断）

## PPOハイパーパラメータ (train_ppo_rgbd.py:44)

| パラメータ | 値 | 理由 |
|-----------|-----|------|
| learning_rate | 1e-4 | 安定化 |
| num_steps | 100 | 遅延報酬のカバー |
| gamma | 0.85 | 操作タスク向け |
| num_minibatches | 32 | |
| update_epochs | 4 | |
| clip_coef | 0.1 | 大きな方策ジャンプ防止 |
| ent_coef | 0.01 | entropy崩壊防止 |
| target_kl | 0.05 | per-minibatch早期停止 |

## Vast.ai 運用

### インスタンス管理 (uvxで実行)
```bash
# インスタンス検索
uvx vastai search offers 'gpu_name=RTX_4090 num_gpus=1 inet_up>500 disk_space>50' --order dph_total

# インスタンス作成
uvx vastai create instance <ID> --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime --disk 50

# SSH接続（非対話的コマンド送信）
ssh -o StrictHostKeyChecking=no -i ~/.ssh/vast_ai root@<host> -p <port> '<command>'

# SSH URL取得
uvx vastai ssh-url <contract_id>

# デストロイ
uvx vastai destroy instance <contract_id> -y
```

**注意**: `--onstart-cmd`は動作しない場合がある。SSH接続後に手動でセットアップを実行：
```bash
ssh ... 'nohup bash -c "curl -fsSL https://raw.githubusercontent.com/YUGOROU/nut-grasp-maniskill/main/setup_vastai.sh | bash > /workspace/setup.log 2>&1" &'
```

### 訓練コマンド (Vast.ai RTX 4090)
```bash
cd /workspace/nut-grasp-maniskill
source .venv/bin/activate
python train_ppo_rgbd.py \
  --num-envs 2048 \
  --sim-backend gpu \
  --total-timesteps 50000000 \
  --save-freq 50 \
  --use-wandb
```

- num_envs=2048: VRAM 11.1/24GB, GPU 99%, SPS ~8000
- save_freq=50: update=50毎にcheckpoints/<run_name>/ckpt_*.pt を保存

### チェックポイントのHFアップロード
```bash
# インスタンス上で
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi(token='<HF_TOKEN>')
api.upload_folder(folder_path='checkpoints', repo_id='YUGOROU/gyoza-nutgrasp-checkpoints', repo_type='model')
"
```

## 監視指標

| 指標 | 初期値 | 目標 |
|------|--------|------|
| ep_rew | ~13 | >20 |
| success | 0.00% | >0%（10M steps前後） |
| ent | ~8.5 | 緩やかに低下（急崩壊は問題） |
| kl | <0.05 | 0.05以下を維持 |
| v_loss | <1.0 | 安定 |

## ローカルテスト (Mac/CPU)
```bash
pip install -e ".[train]"
python train_ppo_rgbd.py --num-envs 1 --sim-backend cpu --total-timesteps 100000 --num-minibatches 1
```
