"""Self-play with meta-guide from 【酒馆战棋】10月上分攻略 (2026-10)

Guide highlights injected as action-shaping heuristics + rollout tags:
  - 运营节奏: T3不跳三本, T4升三本+黑赐, T5黑赐找3本, T6选便宜饰品, T7升五本看血量, T8-10不升六本
  - T0英雄: Zerek (经济复制滚雪球), Marin (饰品神), Trade Prince (理财), Putricide (亡灵), Great Akazamzarak (奥秘)
  - 阵容偏好: 元素/亡灵/机械/鱼人 强度5星

Two self-play modes:
  --mode pure       : 8 copies of PPO checkpoint (standard self-play)
  --mode guided     : PPO + guide_shaper mixed (higher quality, more diverse rollouts)

Usage:
  cd /path/to/AutoTavern
  pip install .
  python3 selfplay.py --mode guided --games 50 --output guided_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

AUTO_TAVERN = Path(os.environ.get("AUTO_TAVERN", "/Users/jusbinzhong/python_project/project/AutoTavern"))
sys.path.insert(0, str(AUTO_TAVERN))

import taverngym
from action_limits import limit_action_mask
from tavern_bc_runtime import _load_model, _DEVICE

FEATURE_DIM = 4017
CARD_DIM = 447
ACTION_DIM = 245

# Action indices (from TavernGym action_name)
ACT_END_TURN = 0
ACT_REFRESH = 1
ACT_FREEZE = 2
ACT_UPGRADE = 3
ACT_HERO_POWER_0 = 4
ACT_HERO_POWER_1 = 5
ACT_DARK_GIFT = 6
ACT_BACK_OUT = 7
ACT_BUY_BASE = 8       # 8-15: buy tavern card 0-7
ACT_CHOOSE_BASE = 30   # 30-45: choose pending choice option
ACT_PLAY_BASE = 46     # 46-125: play hand card
ACT_MAGNETIZE_BASE = 126
ACT_MOVE_BASE = 196


def decode_obs(obs_bytes, mask_bytes):
    features = np.frombuffer(obs_bytes[0], dtype="<f4").copy()
    cards = np.frombuffer(obs_bytes[1], dtype="<i4").copy()
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).astype(bool)
    if features.shape != (FEATURE_DIM,) or cards.shape != (CARD_DIM,) or mask.shape != (ACTION_DIM,):
        raise ValueError(f"bad obs shapes: {features.shape}, {cards.shape}, {mask.shape}")
    mask = limit_action_mask(features, mask)
    return features, cards, mask


def pick_action_logits(model, features, cards, mask):
    import torch
    f = torch.from_numpy(features).float().unsqueeze(0).to(_DEVICE)
    c = torch.from_numpy(cards).long().unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        logits = model(f, c)[0].cpu().numpy().copy()
    logits[~mask] = -np.inf
    return logits


def sample_action(logits, mask, temperature=1.0):
    if temperature < 0.01:
        return int(np.argmax(logits))
    l = logits.copy() / temperature
    l[~mask] = -np.inf
    l -= l.max()
    probs = np.exp(l)
    probs[~mask] = 0
    s = probs.sum()
    if s == 0:
        return int(np.argmax(mask))
    probs /= s
    return int(np.random.choice(ACTION_DIM, p=probs))


# --- 攻略参数化表（10月上分攻略 + 用户口径） ---
# 种族强度（满分5）：5=本局核心可梭哈，4=能玩，3=凑数，1=不碰
RACE_SCORE = {
    "ELEMENTAL": 5, "UNDEAD": 5, "MECHANICAL": 5, "MURLOC": 5,
    "BEAST": 4, "DEMON": 4,
    "PIRATE": 3, "QUILBOAR": 3, "ABERRATION": 3, "NAGA": 3,
    "DRAGON": 1,
}

# T0 英雄 → 打法代号
T0_HERO = {
    "BG30_HERO_304": "marin",          # 经理马林：饰品神，须保证中立池（场上同种族 <2）
    "BG31_HERO_005": "zerek",          # 克隆大师泽里克：黑赐经济随从直接复制滚雪球，技能越晚用越亏
    "TB_BaconShop_HERO_10": "gallywix",     # 贸易亲王加里维克斯：卖随从→下回合1币
    "TB_BaconShop_HERO_21": "akazamzarak",  # 伟大的阿卡扎曼扎拉克：前期奥秘战力，后期自带毒
    "BG25_HERO_100": "putricide",      # 普崔塞德教授：5币跳三本，进退皆可
}

# 黑赐/发现 选项语义优先级（用户口径）
#   1 刷新（最高） > 2 经济/给随从/两回合收益 > 3 身材+5+5 / 抓取随从 > 4 圣盾风怒等关键词
GIFT_KEYWORDS = (
    ("refresh", ("刷新", "酒馆", "免费"), 1.6),
    ("economy", ("铸币", "金币", "铸币上限", "下回合", "两个回合", "经济"), 1.1),
    ("stats", ("+5/+5", "+5+5", "属性值", "身材", "抓取", "复制", "随从"), 0.8),
    ("keyword", ("圣盾", "风怒", "嘲讽", "复生", "剧毒", "潜行"), 0.45),
)

ACT_SELL_BASE = 16


class GuideContext:
    """缓存每个 seat 的 player_view（完整 JSON 序列化开销大，按回合缓存）。"""

    def __init__(self):
        self._cache = {}

    def snapshot(self, game, seat, turn, refresh=False):
        key = (seat, turn)
        if refresh or key not in self._cache:
            try:
                self._cache[key] = json.loads(game.player_view(seat))
            except Exception:
                self._cache[key] = None
        return self._cache[key]


def _me(view):
    return (view or {}).get("me") or {}


def _classify_gift_option(opt):
    text = " ".join(str(opt.get(k) or "") for k in ("name_zh", "text_zh", "name", "text"))
    for tag, keys, weight in GIFT_KEYWORDS:
        if any(k in text for k in keys):
            return tag, weight
    return None, 0.0


def guide_shapings(game, seat, mask, logits, ctx):
    """按攻略对 logits 做软加权。返回 (adjusted_logits, tags)。

    节奏（用户口径）：T2升2本 → T3黑赐优先 → T4黑赐优先 → T5升三本+黑赐找3本
    → T6饰品后升四 → T7看血升五 → T8-10不升六（先在五本找 key 牌和经济）
    turn 0 = T1，因此 Tn 对应 turn = n-1。
    """
    tags = {}
    l = logits.copy()
    turn = int(game.turn)

    has_choose = bool(mask[ACT_CHOOSE_BASE:ACT_CHOOSE_BASE + 16].any())
    view = ctx.snapshot(game, seat, turn, refresh=has_choose)
    me = _me(view)

    hero = T0_HERO.get((me.get("hero") or {}).get("id"))
    if hero:
        tags[f"hero_{hero}"] = True

    health = int(me.get("health") or 30) + int(me.get("armor") or 0)
    tier = int(me.get("tier") or 1)
    gold = int(me.get("gold") or 0)
    upgrade_price = int(me.get("upgrade_price") or 0)
    races = view.get("lobby_races") or []
    race_best = max((RACE_SCORE.get(r, 3) for r in races), default=3)
    if race_best >= 5:
        tags["race_strong_pool"] = True

    low_hp = health <= 15          # 血线告急：稳战力优先
    dead_gold = upgrade_price > 0 and gold - upgrade_price < 3   # 上本后买不起牌 → 不上

    def up(delta):
        if mask[ACT_UPGRADE]:
            l[ACT_UPGRADE] += delta

    def push(delta, tag, act=ACT_REFRESH):
        if mask[act]:
            l[act] += delta
            tags[tag] = True

    # ===== T2 (turn 1): 升 2 本 =====
    if turn == 1 and tier < 2:
        up(3.0)
        tags["guide_T2_upgrade_t2"] = True

    # ===== T3 (turn 2) / T4 (turn 3): 黑赐优先，靠黑赐稳血赚钱 =====
    if turn in (2, 3):
        tn = turn + 1
        if mask[ACT_DARK_GIFT]:
            l[ACT_DARK_GIFT] += 3.5
            tags[f"guide_T{tn}_dark_gift"] = True
        push(1.5, f"guide_T{tn}_refresh")     # 黑赐内部：刷新优先

    # ===== T5 (turn 4): 升三本 + 黑赐找 3 本 =====
    if turn == 4:
        if tier < 3:
            up(2.0)
            tags["guide_T5_upgrade_t3"] = True
        if mask[ACT_DARK_GIFT]:
            l[ACT_DARK_GIFT] += 2.5
            tags["guide_T5_dark_gift_t3"] = True
        push(1.0, "guide_T5_refresh")

    # ===== T6 (turn 5): 选完饰品后升四本 =====
    if turn == 5 and tier < 4:
        up(2.0)
        tags["guide_T6_after_trinket_upgrade"] = True

    # ===== T7 (turn 6): 看血升五本 =====
    if turn == 6 and tier < 5:
        if low_hp:
            up(-1.5)                           # 血线告急：先稳战力
            tags["guide_T7_hold_low_hp"] = True
        else:
            up(1.8)
            tags["guide_T7_push_t5"] = True
    if turn == 6:
        push(1.0, "guide_T7_refresh")

    # ===== T8-T10 (turn 7-9): 不升六本，先在五本找 key 牌和经济 =====
    if 7 <= turn <= 9:
        if tier >= 6:
            up(-4.0)
            tags["guide_T8_no_t6"] = True
        elif tier >= 5:
            up(-3.0)                           # 明确不上六
            tags["guide_T8_no_t6"] = True
        up(-1.0)
        push(1.5, "guide_T8_refresh_key")      # 五本刷新搜 key
        if mask[ACT_DARK_GIFT]:
            l[ACT_DARK_GIFT] += 1.0            # 黑赐继续用

    # ===== 通用：上本后买不起牌就不上（攻略⑤）=====
    if dead_gold and turn <= 9 and mask[ACT_UPGRADE]:
        up(-2.5)
        tags["guide_no_upgrade_no_gold"] = True

    # ===== 通用：黑赐/发现选项按语义分级加权 =====
    # 刷新 > 经济/给随从 > 身材+5+5/抓取 > 圣盾风怒
    if has_choose:
        kind = ((me.get("pending") or {}).get("kind"))
        options = (me.get("pending") or {}).get("options") or []
        for i, opt in enumerate(options[:16]):
            ci = ACT_CHOOSE_BASE + i
            if ci >= ACTION_DIM or not mask[ci]:
                continue
            if isinstance(opt, dict):
                gtag, weight = _classify_gift_option(opt)
            else:
                gtag, weight = None, 0.0
            l[ci] += 0.3 + weight
            if gtag:
                tags[f"gift_{kind}_{gtag}"] = True
        # 没有识别出语义时，仍然别错过任何选择
        if not options:
            for ci in range(ACT_CHOOSE_BASE, min(ACT_CHOOSE_BASE + 16, ACTION_DIM)):
                if mask[ci]:
                    l[ci] += 0.3

    # ===== 英雄专属打法（T0 五神）=====
    if hero == "putricide":
        # 5 币公式跳三本；T5 时比通用节奏更凶
        if turn == 4 and tier < 3:
            up(1.0)
            tags["hero_putricide_jump_t3"] = True
        for hp in (ACT_HERO_POWER_0, ACT_HERO_POWER_1):
            if mask[hp]:
                l[hp] += 1.2
    elif hero == "zerek":
        # 前 5-7 回合黑赐经济随从立刻复制，技能越晚用越亏
        if 1 <= turn <= 6:
            for hp in (ACT_HERO_POWER_0, ACT_HERO_POWER_1):
                if mask[hp]:
                    l[hp] += 2.0
        if mask[ACT_DARK_GIFT] and turn <= 6:
            l[ACT_DARK_GIFT] += 1.0
    elif hero == "marin":
        # T5 拿小饰品；场上同种族随从 <2 才能进中立池
        if turn == 4:
            for hp in (ACT_HERO_POWER_0, ACT_HERO_POWER_1):
                if mask[hp]:
                    l[hp] += 2.5
            tags["hero_marin_trinket_t5"] = True
        if turn >= 4 and mask[ACT_REFRESH]:
            l[ACT_REFRESH] += 0.8
    elif hero == "gallywix":
        # 卖随从 → 下回合 1 币：缺钱时鼓励卖理财
        if gold <= 6:
            for si in range(ACT_SELL_BASE, ACT_SELL_BASE + 7):
                if mask[si]:
                    l[si] += 1.2
            tags["hero_gallywix_sell_economy"] = True
    elif hero == "akazamzarak":
        # 前期靠奥秘吃战力，后期技能自带不占位毒
        for hp in (ACT_HERO_POWER_0, ACT_HERO_POWER_1):
            if mask[hp]:
                l[hp] += 2.5 if turn <= 3 else 0.8
        if turn <= 3:
            tags["hero_akazamzarak_early_secret"] = True

    # ===== 种族：本局有 5 分种族才值得围绕它滚雪球 =====
    if race_best >= 5 and turn >= 2:
        for bi in range(ACT_BUY_BASE, ACT_BUY_BASE + 8):
            if mask[bi]:
                l[bi] += 0.5
    elif race_best <= 2 and turn >= 3 and mask[ACT_REFRESH]:
        l[ACT_REFRESH] += 0.4                  # 烂种族池：多刷新找中立 key

    l[~mask] = -np.inf
    return l, tags


def play_one_game(seed, model, temperature=1.0, mode="pure", collect=True):
    game = taverngym.Game(seed)
    rollout = []
    guard = 0
    ctx = GuideContext()

    while not game.is_over() and guard < 100000:
        progressed = False
        for seat in range(game.num_players):
            if not game.needs_action(seat):
                continue
            obs = game.observe2(seat)
            mask_bytes = game.legal_mask(seat)
            features, cards, mask = decode_obs(obs, mask_bytes)

            logits = pick_action_logits(model, features, cards, mask)

            tags = {}
            if mode == "guided":
                logits, tags = guide_shapings(game, seat, mask, logits, ctx)

            action = sample_action(logits, mask, temperature)
            if not mask[action]:
                action = int(np.argmax(mask))

            if collect:
                rollout.append({
                    "seed": seed,
                    "seat": seat,
                    "turn": int(game.turn),
                    "features": features.tolist(),
                    "cards": cards.tolist(),
                    "action_mask": mask.astype(np.uint8).tolist(),
                    "action": action,
                    "mode": mode,
                    "guide_tags": tags,
                })
            game.step(seat, action)
            progressed = True
        guard += 1
        if not progressed:
            break

    placements = list(game.placements())
    if collect:
        for r in rollout:
            r["placement"] = placements[r["seat"]]
    return placements, rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", type=str,
                    default="/Users/jusbinzhong/python_project/project/tavern-arena/models/tavern_ppo_final.pt")
    ap.add_argument("--output", type=str, default="rollouts.jsonl")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--mode", choices=["pure", "guided"], default="guided",
                    help="pure=纯自玩 guided=攻略引导自玩（推荐）")
    ap.add_argument("--limit-per-game", type=int, default=0)
    args = ap.parse_args()

    os.environ["TAVERN_BC_CHECKPOINT"] = args.checkpoint

    assert taverngym.OBS2_FLOATS == FEATURE_DIM, f"OBS2_FLOATS {taverngym.OBS2_FLOATS} != {FEATURE_DIM}"
    assert taverngym.OBS2_CARDS == CARD_DIM, f"OBS2_CARDS {taverngym.OBS2_CARDS} != {CARD_DIM}"
    print(f"[selfplay] taverngym OK: OBS2=({taverngym.OBS2_FLOATS},{taverngym.OBS2_CARDS})", flush=True)

    print(f"[selfplay] loading model from {args.checkpoint}", flush=True)
    model = _load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[selfplay] model: {n_params/1e6:.2f}M params, device={_DEVICE}", flush=True)
    print(f"[selfplay] mode={args.mode}, temperature={args.temperature}, games={args.games}", flush=True)

    all_placements = []
    started = time.time()
    guide_tag_counts = {}

    with open(args.output, "w") as f:
        for gi in range(args.games):
            seed = args.seed + gi
            t0 = time.time()
            placements, rollout = play_one_game(seed, model, args.temperature, args.mode)
            elapsed = time.time() - t0
            all_placements.append(placements)

            # 统计 guide tags
            for r in rollout:
                for tag in r.get("guide_tags", {}):
                    guide_tag_counts[tag] = guide_tag_counts.get(tag, 0) + 1

            entries = rollout[:args.limit_per_game] if args.limit_per_game > 0 else rollout
            for r in entries:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()

            avg_pl = np.mean(placements)
            top4 = sum(1 for p in placements if p <= 4)
            turns = max((r["turn"] for r in rollout), default=0)
            print(f"  game {gi+1}/{args.games} seed={seed} avg={avg_pl:.2f} top4={top4}/8 "
                  f"turns={turns} acts={len(rollout)} ({elapsed:.1f}s)", flush=True)

    # --- summary ---
    all_pl = np.array(all_placements)
    flat = all_pl.flatten()
    print(f"\n{'='*60}")
    print(f"[selfplay] DONE: mode={args.mode}, {args.games} games, ")
    print(f"[selfplay] overall avg_placement: {flat.mean():.3f} ± {flat.std():.3f}")
    print(f"[selfplay] overall top4 rate: {sum(1 for p in flat if p<=4)}/{len(flat)} "
          f"= {sum(1 for p in flat if p<=4)/len(flat)*100:.1f}%")
    for seat in range(8):
        sp = all_pl[:, seat]
        print(f"  seat {seat}: avg={sp.mean():.2f} top4={sum(1 for p in sp if p<=4)}/{len(sp)}")

    if guide_tag_counts:
        print(f"\n[selfplay] guide tag activation:")
        for tag, cnt in sorted(guide_tag_counts.items(), key=lambda x: -x[1]):
            print(f"  {tag}: {cnt}")

    print(f"\n[selfplay] rollout → {args.output}")


if __name__ == "__main__":
    main()
