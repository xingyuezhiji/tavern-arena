"""TavernArena policy adapter for AutoTavern PPO checkpoint.

Usage:
  export ARENA_API_KEY="your-key-here"
  cd tavern-arena
  python arena_client.py --url https://rizling.top --policy my_agent:act --mode practice
  python arena_client.py --url https://rizling.top --policy my_agent:act --mode arena
"""
from __future__ import annotations

import os
import sys

import numpy as np

# 确保能找到同目录下的运行时依赖
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch

from tavern_bc_runtime import _load_model, _DEVICE

# 默认 checkpoint 路径，强制传给 tavern_bc_runtime
_CHECKPOINT = os.getenv(
    "TAVERN_BC_CHECKPOINT",
    os.path.join(_THIS_DIR, "models", "tavern_ppo_final.pt"),
)
os.environ["TAVERN_BC_CHECKPOINT"] = _CHECKPOINT  # 强制让 _load_model 读到


def act(observation: dict, context: dict) -> int:
    """Arena policy entry point — return a legal action index."""
    model = _load_model()

    features = torch.from_numpy(
        np.array(observation["features"], dtype=np.float32, copy=True)
    ).unsqueeze(0).to(_DEVICE)

    cards = torch.from_numpy(
        np.array(observation["cards"], dtype=np.int64, copy=True)
    ).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = model(features, cards)[0].detach().cpu().numpy()

    mask = np.asarray(observation["action_mask"], dtype=bool)
    masked = np.where(mask, logits, -np.inf)
    return int(np.argmax(masked))


# 加载 checkpoint 元数据，方便 arena_client 做契约校验
def get_ruleset_id() -> str:
    ckpt = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
    return ckpt["ruleset_id"]


def get_catalog_sha256() -> str:
    ckpt = torch.load(_CHECKPOINT, map_location="cpu", weights_only=False)
    return ckpt["catalog_sha256"]


if __name__ == "__main__":
    # 快速自检：加载模型并跑一条随机观测
    print(f"Loading checkpoint: {_CHECKPOINT}")
    print(f"Device: {_DEVICE}")
    print(f"Ruleset: {get_ruleset_id()}")
    print(f"Catalog SHA: {get_catalog_sha256()}")

    model = _load_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,}")

    fake_obs = {
        "features": np.random.randn(4017).astype(np.float32),
        "cards": np.random.randint(0, 4096, size=447).astype(np.int32),
        "action_mask": np.random.choice([True, False], size=245, p=[0.3, 0.7]).astype(bool),
    }
    # 确保至少有一个合法动作
    fake_obs["action_mask"][0] = True

    action = act(fake_obs, {"match_id": "test", "seat": 0})
    print(f"Test action: {action} (legal={fake_obs['action_mask'][action]})")
    print("✅ Model adapter ready!")
