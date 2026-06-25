"""
Formation Build-Up Patterns
----------------------------
For each team, maps every player to their starting position, then traces
the POSITIONAL sequence (e.g. GK → LCB → RCB → CDM → RWB → RW → ST)
through qualifying build-up possessions (starts in own half, reaches
attacking third).  Consecutive repeats of the same position are collapsed.

Outputs
-------
  01  Pitch flow diagram  — arrows between positions weighted by transition freq
  02  Top chains          — most common positional chain strings, team-by-team
  03  Transition heatmap  — position → position pass matrix (build-up zone only)
"""

import json, math
from pathlib import Path
from collections import defaultdict, Counter
from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
from mplsoccer import Pitch, VerticalPitch

# ── config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_FILES = [
    "2025-09-06_PSV_Eindhoven_W_vs_Excelsior_W_4014321.json",
    "2025-10-12_Heerenveen_W_vs_HERA_United_W_4014348.json",
    "2025-11-16_Heerenveen_W_vs_PEC_Zwolle_W_4014359.json",
]
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

TEAM_COLORS = {
    "PSV Eindhoven W": "#E3000F",
    "Excelsior W":     "#B8860B",
    "Heerenveen W":    "#003DA5",
    "HERA United W":   "#7B2D8B",
    "PEC Zwolle W":    "#009FE3",
}

# ── position id → short label ─────────────────────────────────────────────────
POS_LABEL = {
    1: "GK",
    2: "RB",   3: "RCB",  4: "CB",   5: "LCB",  6: "LB",
    7: "RWB",  8: "LWB",
    9: "RDM",  10: "CDM", 11: "LDM",
    12: "RM",  13: "RCM", 14: "CM",  15: "LCM", 16: "LM",
    17: "RW",  18: "RAM", 19: "CAM", 20: "LAM", 21: "LW",
    22: "RCF", 23: "ST",  24: "LCF", 25: "SS",
}

# Position group for colour coding
POS_GROUP = {
    "GK":  "GK",
    "RB": "DEF", "RCB": "DEF", "CB": "DEF", "LCB": "DEF", "LB": "DEF",
    "RWB": "WB",  "LWB": "WB",
    "RDM": "MID", "CDM": "MID", "LDM": "MID",
    "RM":  "MID", "RCM": "MID", "CM":  "MID", "LCM": "MID", "LM": "MID",
    "RW":  "ATT", "RAM": "ATT", "CAM": "ATT", "LAM": "ATT", "LW": "ATT",
    "RCF": "ATT", "ST":  "ATT", "LCF": "ATT", "SS":  "ATT",
}
GROUP_COLOR = {"GK": "#f5a623", "DEF": "#4a90d9", "WB": "#7ed321",
               "MID": "#9b59b6", "ATT": "#e74c3c"}

# Approximate pitch coordinates per position (StatsBomb 120×80, attack→right)
POS_XY = {
    "GK":  ( 4, 40),
    "RB":  (25, 64), "RCB": (20, 56), "CB":  (20, 40), "LCB": (20, 24), "LB":  (25, 16),
    "RWB": (42, 74), "LWB": (42,  6),
    "RDM": (42, 60), "CDM": (42, 40), "LDM": (42, 20),
    "RM":  (55, 72), "RCM": (52, 58), "CM":  (52, 40), "LCM": (52, 22), "LM":  (55,  8),
    "RW":  (82, 70), "RAM": (72, 60), "CAM": (72, 40), "LAM": (72, 20), "LW":  (82, 10),
    "RCF": (96, 54), "ST":  (100,40), "LCF": (96, 26), "SS":  (88, 40),
}

ON_BALL      = {"Pass", "Ball Receipt*", "Dribble", "Shot"}  # no Carry
PASS_ACTS    = {"Pass"}

def norm_x(x, d): return 120 - x if d == "right_to_left" else x
def norm_y(y, d): return  80 - y if d == "right_to_left" else y

def event_pos(e):
    """Get position label from the event itself (always accurate, handles subs)."""
    pid = e.get("position", {}).get("id")
    return POS_LABEL.get(pid, "?") if pid else "?"

# ── load events ───────────────────────────────────────────────────────────────
all_events = {}
for fname in DATA_FILES:
    with open(BASE / fname) as f:
        all_events[fname] = json.load(f)

# ── build player → position map (respects substitutions & tactical shifts) ───
def build_pos_map(events, team_name):
    """Returns {player_name: pos_label} for team_name, tracking lineup changes."""
    pos_map = {}

    for e in events:
        if e.get("team", {}).get("name") != team_name:
            continue

        if e["type"]["name"] == "Starting XI":
            for p in e["tactics"]["lineup"]:
                name = p["player"]["name"]
                pid  = p["position"]["id"]
                pos_map[name] = POS_LABEL.get(pid, f"P{pid}")

        elif e["type"]["name"] in ("Substitution", "Tactical Shift"):
            # StatsBomb substitution: player goes off, replacement comes on
            if e["type"]["name"] == "Substitution":
                sub = e.get("substitution", {})
                off = e.get("player", {}).get("name")
                on  = sub.get("replacement", {}).get("name")
                # replacement inherits the outgoing player's position
                if off and on and off in pos_map:
                    pos_map[on] = pos_map[off]
                    pos_map.pop(off, None)

    return pos_map


# ── extract build-up sequences with position chains ──────────────────────────
def get_pos_sequences(events, team_name):
    chains = defaultdict(list)
    for e in events:
        if e.get("team", {}).get("name") == team_name:
            chains[e["possession"]].append(e)

    sequences = []
    for poss_id, evs in chains.items():
        on_ball = [
            e for e in evs
            if e["type"]["name"] in ON_BALL
            and e.get("location")
            and e.get("player")
        ]
        if len(on_ball) < 2:
            continue

        xs = [norm_x(e["location"][0], e.get("attacking_direction", "left_to_right"))
              for e in on_ball]

        if xs[0] > 60 or max(xs) < 80:
            continue

        # position chain — one entry per on-ball event, using event's own position
        pos_chain_raw = [event_pos(e) for e in on_ball]

        # drop unknowns then collapse consecutive identical positions
        pos_chain_raw = [p for p in pos_chain_raw if p != "?"]
        pos_chain = [k for k, _ in groupby(pos_chain_raw)]

        if len(pos_chain) < 2:
            continue

        # pass-level transitions (passer pos → receiver pos) for heatmap
        pass_pairs = []
        pass_evs = [e for e in on_ball if e["type"]["name"] == "Pass"]
        for p in pass_evs:
            p_pos = event_pos(p)
            # find the next Ball Receipt* in the sequence for receiver position
            idx = on_ball.index(p)
            for nxt in on_ball[idx + 1:]:
                if nxt["type"]["name"] == "Ball Receipt*":
                    r_pos = event_pos(nxt)
                    if p_pos != "?" and r_pos != "?":
                        pass_pairs.append((p_pos, r_pos))
                    break

        sequences.append({
            "possession": poss_id,
            "team":       team_name,
            "chain":      pos_chain,
            "chain_str":  " → ".join(pos_chain),
            "pass_pairs": pass_pairs,
            "start_x":    xs[0],
            "max_x":      max(xs),
            "n_positions": len(set(pos_chain)),
        })

    return sequences


# ── aggregate per team ────────────────────────────────────────────────────────
team_seqs      = {}   # team -> list of seq dicts
team_formation = {}   # team -> formation int (from first match)
team_pos_map   = {}   # team -> pos_map

for fname, events in all_events.items():
    teams = {e["team"]["name"] for e in events if e.get("team")}

    # grab formations from Starting XI
    for e in events:
        if e["type"]["name"] == "Starting XI":
            t = e["team"]["name"]
            if t not in team_formation:
                team_formation[t] = e["tactics"]["formation"]

    for team in teams:
        seqs = get_pos_sequences(events, team)
        team_seqs.setdefault(team, []).extend(seqs)

teams_sorted = sorted(team_seqs)

# ── print top chains per team ─────────────────────────────────────────────────
print(f"\n{'':=<80}")
for team in teams_sorted:
    seqs = team_seqs[team]
    form = team_formation.get(team, "?")
    print(f"\n  {team}  [{form}]  ({len(seqs)} build-up sequences)")
    chain_counts = Counter(s["chain_str"] for s in seqs)
    for chain, cnt in chain_counts.most_common(10):
        print(f"    {cnt:>3}×  {chain}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Pitch flow per team
# ─────────────────────────────────────────────────────────────────────────────
ncols = 3
nrows = math.ceil(len(teams_sorted) / ncols)

fig1 = plt.figure(figsize=(7 * ncols, 5.5 * nrows), facecolor="#1a1a2e")
fig1.suptitle(
    "Build-Up Positional Flow  (own half → attacking third)",
    fontsize=14, fontweight="bold", color="white", y=1.01,
)

for idx, team in enumerate(teams_sorted):
    seqs  = team_seqs[team]
    color = TEAM_COLORS.get(team, "#ffffff")
    form  = team_formation.get(team, "")

    # aggregate pass transitions
    pair_counts = Counter(
        pair for s in seqs for pair in s["pass_pairs"]
    )
    # positions actually used in this team's sequences
    used_pos = {pos for s in seqs for pos in s["chain"]} - {"?"}

    ax = fig1.add_subplot(nrows, ncols, idx + 1)
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#444466", linewidth=1)
    pitch.draw(ax=ax)

    if not pair_counts:
        ax.set_title(f"{team}\n[{form}]", color=color, fontsize=9, fontweight="bold")
        continue

    max_count = max(pair_counts.values())

    # draw arrows
    for (src, dst), cnt in pair_counts.items():
        if src not in POS_XY or dst not in POS_XY or src == dst:
            continue
        x0, y0 = POS_XY[src]
        x1, y1 = POS_XY[dst]
        alpha  = 0.25 + 0.65 * cnt / max_count
        lw     = 0.8  + 3.5  * cnt / max_count
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                connectionstyle="arc3,rad=0.12",
                mutation_scale=12,
            ),
        )

    # draw position nodes
    pos_touch = Counter(pos for s in seqs for pos in s["chain"] if pos != "?")
    max_touch  = max(pos_touch.values()) if pos_touch else 1

    for pos in used_pos:
        if pos not in POS_XY:
            continue
        x, y     = POS_XY[pos]
        grp       = POS_GROUP.get(pos, "MID")
        node_col  = GROUP_COLOR[grp]
        size      = 80 + 300 * pos_touch.get(pos, 0) / max_touch
        ax.scatter(x, y, s=size, color=node_col, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(x, y, pos, ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold", zorder=6)

    ax.set_title(f"{team}  [{form}]", color=color,
                 fontsize=9, fontweight="bold", pad=6)

# legend
legend_patches = [mpatches.Patch(color=c, label=g)
                  for g, c in GROUP_COLOR.items()]
fig1.legend(handles=legend_patches, loc="lower center",
            ncol=5, fontsize=9, frameon=False,
            labelcolor="white", bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out1 = OUT / "formation_flow.png"
plt.savefig(out1, dpi=160, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print(f"\nSaved {out1.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Top position chains as horizontal chain diagrams
# ─────────────────────────────────────────────────────────────────────────────
TOP_N = 8   # chains per team

fig2_rows = len(teams_sorted)
fig2, axes2 = plt.subplots(fig2_rows, 1,
                             figsize=(14, 2.2 * fig2_rows),
                             facecolor="white")
if fig2_rows == 1:
    axes2 = [axes2]

fig2.suptitle("Most Common Positional Build-Up Chains (own half → att. third)",
              fontsize=13, fontweight="bold", color="#1a1a1a", y=1.01)

for ax, team in zip(axes2, teams_sorted):
    seqs       = team_seqs[team]
    form       = team_formation.get(team, "")
    team_color = TEAM_COLORS.get(team, "#555")
    chain_counts = Counter(s["chain_str"] for s in seqs).most_common(TOP_N)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(chain_counts) - 0.5)
    ax.axis("off")

    # team label on left
    ax.text(-0.01, (len(chain_counts) - 1) / 2,
            f"{team}\n[{form}]",
            ha="right", va="center", fontsize=9, fontweight="bold",
            color=team_color, transform=ax.transData)

    for row, (chain_str, cnt) in enumerate(reversed(chain_counts)):
        nodes = chain_str.split(" → ")
        n     = len(nodes)
        # space nodes evenly across [0.02, 0.98]
        xs    = np.linspace(0.02, 0.96, n)
        y     = row

        # connector line
        ax.plot(xs, [y] * n, color="#dddddd", lw=1.2, zorder=1)

        for xi, pos in zip(xs, nodes):
            grp     = POS_GROUP.get(pos, "MID")
            bg_col  = GROUP_COLOR[grp]
            # node circle
            circ = plt.Circle((xi, y), 0.012, color=bg_col, zorder=3,
                               transform=ax.transData)
            ax.add_patch(circ)
            ax.text(xi, y, pos, ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=4)

        # frequency badge
        ax.text(0.985, y, f"{cnt}×",
                ha="left", va="center", fontsize=7.5,
                color="#888888", fontweight="bold")

    ax.set_facecolor("white")

plt.tight_layout()
out2 = OUT / "formation_chains.png"
plt.savefig(out2, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out2.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Position → position transition heatmap per team
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
fig3.suptitle("Build-Up Pass Transition Matrix  (position → position)",
              fontsize=13, fontweight="bold", color="#1a1a1a", y=1.01)

for idx, team in enumerate(teams_sorted):
    ax = axes3[idx // 3, idx % 3]
    seqs = team_seqs[team]
    form = team_formation.get(team, "")

    pair_counts = Counter(
        pair for s in seqs for pair in s["pass_pairs"]
    )
    if not pair_counts:
        ax.set_title(f"{team} [{form}]")
        continue

    # positions that appear
    all_pos = sorted({p for pair in pair_counts for p in pair} - {"?"})
    n = len(all_pos)
    pos_idx = {p: i for i, p in enumerate(all_pos)}

    mat = np.zeros((n, n))
    for (src, dst), cnt in pair_counts.items():
        if src in pos_idx and dst in pos_idx:
            mat[pos_idx[src], pos_idx[dst]] = cnt

    # row-normalise to % of passes FROM each position
    row_sum = mat.sum(axis=1, keepdims=True)
    mat_pct = np.where(row_sum > 0, mat / row_sum * 100, 0)

    im = ax.imshow(mat_pct, cmap="Blues", aspect="auto", vmin=0, vmax=60)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_pos, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(all_pos, fontsize=7.5)
    ax.set_xlabel("Receives pass →", fontsize=8)
    ax.set_ylabel("Plays pass ↓", fontsize=8)
    ax.set_title(f"{team}  [{form}]",
                 fontsize=9, fontweight="bold",
                 color=TEAM_COLORS.get(team, "#333"))

    # annotate cells
    for i in range(n):
        for j in range(n):
            val = mat_pct[i, j]
            if val > 5:
                ax.text(j, i, f"{val:.0f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if val > 35 else "#222")

    fig3.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                  label="% of passes from position")

# hide unused subplot
for j in range(len(teams_sorted), 6):
    axes3[j // 3, j % 3].set_visible(False)

plt.tight_layout()
out3 = OUT / "formation_transition_matrix.png"
plt.savefig(out3, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out3.name}")

print(f"\nAll outputs → {OUT}")
