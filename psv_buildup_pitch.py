"""
PSV Eindhoven W — Build-Up Pattern on VerticalPitch
Reads all PSV matches, traces build-up sequences (own half → att. third)
and renders three panels on a mplsoccer VerticalPitch:
  Left   — actual pass locations + arrows, coloured by position group
  Centre — position node network (avg location, sized by touches, weighted arrows)
  Right  — heatmap of pass density in build-up zone
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch

# ── config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
OUT        = BASE / "output"
TEAM       = "PSV Eindhoven W"
TEAM_COLOR = "#E3000F"
TEAM_SECONDARY = "#FFFFFF"

DATA_FILES = [
    "2025-09-06_PSV_Eindhoven_W_vs_Excelsior_W_4014321.json",
    "2025-10-12_Heerenveen_W_vs_HERA_United_W_4014348.json",
    "2025-11-16_Heerenveen_W_vs_PEC_Zwolle_W_4014359.json",
]

POS_LABEL = {
    1:"GK", 2:"RB", 3:"RCB", 4:"CB", 5:"LCB", 6:"LB",
    7:"RWB", 8:"LWB", 9:"RDM", 10:"CDM", 11:"LDM",
    12:"RM", 13:"RCM", 14:"CM", 15:"LCM", 16:"LM",
    17:"RW", 18:"RAM", 19:"CAM", 20:"LAM", 21:"LW",
    22:"RCF", 23:"ST", 24:"LCF", 25:"SS",
}
POS_GROUP = {
    "GK":"GK",
    "RB":"DEF","RCB":"DEF","CB":"DEF","LCB":"DEF","LB":"DEF",
    "RWB":"WB","LWB":"WB",
    "RDM":"DM","CDM":"DM","LDM":"DM",
    "RM":"MID","RCM":"MID","CM":"MID","LCM":"MID","LM":"MID",
    "RW":"ATT","RAM":"ATT","CAM":"ATT","LAM":"ATT","LW":"ATT",
    "RCF":"ATT","ST":"ATT","LCF":"ATT","SS":"ATT",
}
GROUP_COLOR = {
    "GK":"#F5A623","DEF":"#4A90D9","WB":"#7ED321",
    "DM":"#BD10E0","MID":"#9B59B6","ATT":"#E74C3C",
}

ON_BALL = {"Pass","Ball Receipt*","Dribble","Shot"}

def nx(x, d): return 120 - x if d == "right_to_left" else x
def ny(y, d): return  80 - y if d == "right_to_left" else y
def event_pos(e):
    pid = e.get("position", {}).get("id")
    return POS_LABEL.get(pid, "?") if pid else "?"

# ── load all PSV events ───────────────────────────────────────────────────────
all_psv_events = []
for fname in DATA_FILES:
    with open(BASE / fname) as f:
        events = json.load(f)
    psv_ev = [e for e in events if e.get("team", {}).get("name") == TEAM]
    if psv_ev:
        all_psv_events.extend(events)   # keep full match (need possession context)

# ── extract build-up sequences ────────────────────────────────────────────────
def get_sequences(events):
    chains = defaultdict(list)
    for e in events:
        if e.get("team", {}).get("name") == TEAM:
            chains[e["possession"]].append(e)

    seqs = []
    for poss_id, evs in chains.items():
        ob = [e for e in evs
              if e["type"]["name"] in ON_BALL
              and e.get("location") and e.get("player")]
        if len(ob) < 2:
            continue

        xs = [nx(e["location"][0], e.get("attacking_direction", "left_to_right")) for e in ob]
        ys = [ny(e["location"][1], e.get("attacking_direction", "left_to_right")) for e in ob]

        if xs[0] > 60 or max(xs) < 80:
            continue

        pos_raw = [event_pos(e) for e in ob if event_pos(e) != "?"]
        chain   = [k for k, _ in groupby(pos_raw)]
        if len(chain) < 2:
            continue

        # pass arrows: (x0,y0,x1,y1,pos)
        arrows = []
        pass_evs = [(i, e) for i, e in enumerate(ob) if e["type"]["name"] == "Pass"]
        for i, p in pass_evs:
            end = p.get("pass", {}).get("end_location")
            if not end:
                continue
            d    = p.get("attacking_direction", "left_to_right")
            x0   = nx(p["location"][0], d)
            y0   = ny(p["location"][1], d)
            x1   = nx(end[0], d)
            y1   = ny(end[1], d)
            pos  = event_pos(p)
            arrows.append((x0, y0, x1, y1, pos))

        # position → avg location
        pos_locs = defaultdict(list)
        for e, xi, yi in zip(ob, xs, ys):
            p = event_pos(e)
            if p != "?":
                pos_locs[p].append((xi, yi))

        # pass pairs with actual locations
        pairs = []
        for i, e in enumerate(ob):
            if e["type"]["name"] != "Pass":
                continue
            p_pos = event_pos(e)
            if p_pos == "?":
                continue
            for nxt in ob[i+1:]:
                if nxt["type"]["name"] == "Ball Receipt*":
                    r_pos = event_pos(nxt)
                    if r_pos != "?":
                        pairs.append((p_pos, r_pos))
                    break

        seqs.append(dict(
            chain    = chain,
            arrows   = arrows,
            pos_locs = pos_locs,
            pairs    = pairs,
            xs       = xs,
            ys       = ys,
        ))
    return seqs

seqs = get_sequences(all_psv_events)
print(f"PSV: {len(seqs)} build-up sequences across {len(DATA_FILES)} match files loaded")

# aggregate
all_arrows  = [a for s in seqs for a in s["arrows"]]
all_pairs   = [p for s in seqs for p in s["pairs"]]
pair_counts = Counter(all_pairs)

pos_avg_loc = {}
pos_touch   = Counter()
for s in seqs:
    for pos, locs in s["pos_locs"].items():
        pos_avg_loc.setdefault(pos, []).extend(locs)
        pos_touch[pos] += len(locs)

for pos in pos_avg_loc:
    locs = pos_avg_loc[pos]
    pos_avg_loc[pos] = (np.mean([l[0] for l in locs]),
                        np.mean([l[1] for l in locs]))

# ── FIGURE — 3 panels on VerticalPitch ───────────────────────────────────────
BG = "#0d1117"
fig = plt.figure(figsize=(20, 14), facecolor=BG)

# Title
fig.text(0.5, 0.97, f"PSV Eindhoven W — Build-Up Patterns",
         ha="center", va="top", fontsize=20, fontweight="bold",
         color=TEAM_COLOR,
         path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
fig.text(0.5, 0.935, "Build-up sequences: own half → attacking third  |  All available matches",
         ha="center", va="top", fontsize=11, color="#aaaaaa")

pitch_kw = dict(pitch_type="statsbomb", pitch_color=BG,
                line_color="#334466", linewidth=1.2)

# ── Panel 1: Individual pass arrows coloured by position group ────────────────
ax1 = fig.add_axes([0.02, 0.06, 0.30, 0.84])
pitch1 = VerticalPitch(**pitch_kw, half=False)
pitch1.draw(ax=ax1)

# sort arrows so attacking positions draw on top
grp_order = {"GK":0,"DEF":1,"WB":2,"DM":3,"MID":4,"ATT":5}
all_arrows_sorted = sorted(all_arrows,
                            key=lambda a: grp_order.get(POS_GROUP.get(a[4],"MID"),3))

for (x0, y0, x1, y1, pos) in all_arrows_sorted:
    grp   = POS_GROUP.get(pos, "MID")
    color = GROUP_COLOR[grp]
    # VerticalPitch: x stays x (pitch length), y stays y (pitch width)
    ax1.annotate("",
        xy=(y1, x1), xytext=(y0, x0),   # VerticalPitch swaps axes for drawing
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=0.9, alpha=0.55, mutation_scale=7,
                        connectionstyle="arc3,rad=0.0"))

ax1.set_title("Pass Arrows\nby Position Group",
              color="white", fontsize=11, fontweight="bold", pad=10)

# ── Panel 2: Position node network at avg locations ───────────────────────────
ax2 = fig.add_axes([0.35, 0.06, 0.30, 0.84])
pitch2 = VerticalPitch(**pitch_kw, half=False)
pitch2.draw(ax=ax2)

max_pair  = max(pair_counts.values()) if pair_counts else 1
max_touch = max(pos_touch.values())   if pos_touch   else 1

# draw edges first
for (src, dst), cnt in pair_counts.items():
    if src not in pos_avg_loc or dst not in pos_avg_loc:
        continue
    x0, y0 = pos_avg_loc[src]
    x1, y1 = pos_avg_loc[dst]
    lw      = 0.5 + 5.5 * cnt / max_pair
    alpha   = 0.20 + 0.70 * cnt / max_pair
    ax2.annotate("",
        xy=(y1, x1), xytext=(y0, x0),
        arrowprops=dict(arrowstyle="-|>", color=TEAM_COLOR,
                        lw=lw, alpha=alpha, mutation_scale=10,
                        connectionstyle="arc3,rad=0.15"))

# draw nodes
for pos, (px, py) in pos_avg_loc.items():
    grp   = POS_GROUP.get(pos, "MID")
    color = GROUP_COLOR[grp]
    size  = 120 + 550 * pos_touch[pos] / max_touch
    ax2.scatter(py, px, s=size, color=color, zorder=5,
                edgecolors="white", linewidths=1.2)
    ax2.text(py, px, pos, ha="center", va="center",
             fontsize=7, fontweight="bold", color="white", zorder=6)

    # touch count badge
    ax2.text(py + 2.5, px + 2.5, str(pos_touch[pos]),
             ha="center", va="center", fontsize=5.5,
             color="#cccccc", zorder=6)

ax2.set_title("Positional Network\n(node = avg location · size = touches · arrow = passes)",
              color="white", fontsize=11, fontweight="bold", pad=10)

# ── Panel 3: Pass density heatmap + start of sequences ───────────────────────
ax3 = fig.add_axes([0.68, 0.06, 0.30, 0.84])
pitch3 = VerticalPitch(**pitch_kw, half=False)
pitch3.draw(ax=ax3)

# kernel density of pass start locations
pass_xs = [a[0] for a in all_arrows]
pass_ys = [a[1] for a in all_arrows]

if pass_xs:
    psv_cmap = LinearSegmentedColormap.from_list(
        "psv", ["#0d1117", "#330000", TEAM_COLOR, "#ffffff"], N=256)
    pitch3.kdeplot(pass_ys, pass_xs, ax=ax3,
                   cmap=psv_cmap, levels=60,
                   fill=True, alpha=0.75, zorder=2)

# scatter sequence start points (where PSV first gets ball in own half)
start_xs = [s["xs"][0] for s in seqs]
start_ys = [s["ys"][0] for s in seqs]
ax3.scatter(start_ys, start_xs, s=55, color="white", alpha=0.9,
            edgecolors=TEAM_COLOR, linewidths=1.5, zorder=5,
            label="Sequence start")

# mark halfway line label
ax3.axhline(60, color="#ffffff", lw=0.8, ls="--", alpha=0.25, zorder=1)
ax3.axhline(80, color=TEAM_COLOR, lw=0.8, ls="--", alpha=0.4, zorder=1)

ax3.set_title("Pass Density Heatmap\n(⚪ = sequence start location)",
              color="white", fontsize=11, fontweight="bold", pad=10)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLOR.items()]
fig.legend(handles=patches, loc="lower center", ncol=6,
           fontsize=10, frameon=False, labelcolor="white",
           bbox_to_anchor=(0.5, 0.01))

out = OUT / "psv_buildup_verticalpitch.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved {out}")
