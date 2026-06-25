"""
Passing Sequences: Build-Up to Attack
--------------------------------------
For each team, find possession sequences that originate in the defensive /
middle third and reach the attacking third.  For each qualifying sequence,
record how many DISTINCT players were involved (passed, carried, received).

Outputs
-------
  - Summary table: avg players per sequence, total sequences, etc.
  - Distribution chart: histogram of player-count per sequence per team
  - Dot-plot / strip chart: individual sequences plotted by player count
  - Example sequence print-out for inspection
"""

import json, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

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

# Zones (normalised x, 0 = own goal, 120 = opp goal)
DEF_THIRD_END  = 40    # sequence must START at or before this x
ATT_THIRD_START = 80   # sequence must REACH at or beyond this x

ON_BALL = {"Pass", "Carry", "Ball Receipt*", "Dribble", "Shot"}

# ── coordinate helper ─────────────────────────────────────────────────────────
def norm_x(x, direction):
    """Normalise so the team always attacks toward x = 120."""
    return 120 - x if direction == "right_to_left" else x

# ── extract qualifying sequences ──────────────────────────────────────────────
def get_sequences(events, team_name):
    """
    Return a list of dicts, one per qualifying possession sequence.
    A sequence qualifies when:
      - first on-ball action is in own defensive half (x_norm <= 60)
      - sequence reaches the attacking third (x_norm >= 80) at some point
    """
    # group all events by (possession id, possession team id)
    chains = defaultdict(list)
    for e in events:
        t = e.get("team", {}).get("name")
        if t == team_name:
            chains[e["possession"]].append(e)

    sequences = []
    for poss_id, evs in chains.items():
        # only on-ball events with a location
        on_ball = [
            e for e in evs
            if e["type"]["name"] in ON_BALL and e.get("location") and e.get("player")
        ]
        if len(on_ball) < 2:
            continue

        # normalised x for each event
        xs = [
            norm_x(e["location"][0], e.get("attacking_direction", "left_to_right"))
            for e in on_ball
        ]

        # must START in own half (first event x <= 60) and REACH att third
        if xs[0] > 60:
            continue
        if max(xs) < ATT_THIRD_START:
            continue

        # collect unique players in the sequence
        players = list(dict.fromkeys(
            e["player"]["name"] for e in on_ball
        ))   # ordered, deduplicated

        sequences.append({
            "possession":    poss_id,
            "team":          team_name,
            "n_players":     len(players),
            "n_events":      len(on_ball),
            "start_x":       xs[0],
            "max_x":         max(xs),
            "players":       players,
            "events":        on_ball,
            "xs":            xs,
        })

    return sequences


# ── load data & compute ───────────────────────────────────────────────────────
all_sequences = defaultdict(list)   # team -> list of sequence dicts

for fname in DATA_FILES:
    with open(BASE / fname) as f:
        events = json.load(f)
    teams = {e["team"]["name"] for e in events if e.get("team")}
    for team in teams:
        seqs = get_sequences(events, team)
        all_sequences[team].extend(seqs)

teams_sorted = sorted(all_sequences)

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Team':<22} {'Seqs':>5} {'Avg players':>12} {'Median':>8} "
      f"{'Max':>5} {'2-plyr%':>9} {'4+plyr%':>9}")
print("-" * 75)

summary = {}
for team in teams_sorted:
    seqs = all_sequences[team]
    n    = [s["n_players"] for s in seqs]
    avg  = np.mean(n)
    med  = np.median(n)
    mx   = max(n)
    p2   = 100 * sum(1 for x in n if x == 2) / len(n)
    p4p  = 100 * sum(1 for x in n if x >= 4) / len(n)
    summary[team] = dict(n_seqs=len(seqs), avg=avg, median=med, max=mx, pct_2=p2, pct_4plus=p4p)
    print(f"{team:<22} {len(seqs):>5} {avg:>12.2f} {med:>8.1f} {mx:>5} {p2:>8.1f}% {p4p:>8.1f}%")

# ── print one example sequence per team ──────────────────────────────────────
print("\n── Example sequences (longest player chain per team) ────────────────")
for team in teams_sorted:
    seqs   = all_sequences[team]
    best   = max(seqs, key=lambda s: s["n_players"])
    color  = TEAM_COLORS.get(team, "")
    print(f"\n  {team}  (possession #{best['possession']}, "
          f"{best['n_players']} players, {best['n_events']} events, "
          f"x: {best['start_x']:.0f} → {best['max_x']:.0f})")
    for i, (p, x) in enumerate(zip(best["xs"], [e["player"]["name"] for e in best["events"]])):
        print(f"    {i+1:>2}.  {x:<25s}  x={p:.1f}")

# ── FIGURE 1: histogram of n_players per sequence ────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5), facecolor="white")
ax1.set_facecolor("#f8f8f8")

max_players = max(s["n_players"] for seqs in all_sequences.values() for s in seqs)
bins = np.arange(1, max_players + 2) - 0.5

width  = 0.8 / len(teams_sorted)
x_base = np.arange(1, max_players + 1)

for i, team in enumerate(teams_sorted):
    n_list = [s["n_players"] for s in all_sequences[team]]
    counts = [n_list.count(v) for v in x_base]
    # normalise to percentage
    pct    = [100 * c / len(n_list) for c in counts]
    offset = (i - len(teams_sorted) / 2 + 0.5) * width
    color  = TEAM_COLORS.get(team, "#555")
    ax1.bar(x_base + offset, pct, width=width * 0.9,
            color=color, label=team, alpha=0.88, edgecolor="white")

ax1.set_xlabel("Number of distinct players in sequence", fontsize=11)
ax1.set_ylabel("% of qualifying sequences", fontsize=11)
ax1.set_title(
    "How Many Players Are Involved in Build-Up → Attack Sequences?\n"
    "(Sequences starting in own half that reach the attacking third)",
    fontsize=12, fontweight="bold", color="#1a1a1a",
)
ax1.set_xticks(x_base)
ax1.set_xticklabels([str(v) for v in x_base])
ax1.legend(fontsize=8.5, frameon=True, framealpha=0.9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", color="#e0e0e0", linewidth=0.7)

plt.tight_layout()
out1 = OUT / "sequences_histogram.png"
plt.savefig(out1, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved {out1.name}")

# ── FIGURE 2: avg players bar chart + median dot ─────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5), facecolor="white")
ax2.set_facecolor("#f8f8f8")

colors  = [TEAM_COLORS.get(t, "#555") for t in teams_sorted]
avgs    = [summary[t]["avg"]    for t in teams_sorted]
medians = [summary[t]["median"] for t in teams_sorted]
counts  = [summary[t]["n_seqs"] for t in teams_sorted]
short   = [t.replace(" W", "") for t in teams_sorted]

bars = ax2.bar(range(len(teams_sorted)), avgs, color=colors,
               edgecolor="white", width=0.6, alpha=0.88, zorder=3)

# median as a horizontal tick
for i, med in enumerate(medians):
    ax2.hlines(med, i - 0.25, i + 0.25, colors="#333333",
               linewidth=2.5, zorder=4, label="Median" if i == 0 else "")

# annotate bar with avg and sequence count
for i, (bar, avg, cnt) in enumerate(zip(bars, avgs, counts)):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.04,
             f"{avg:.2f}\n({cnt} seqs)",
             ha="center", va="bottom", fontsize=8, color="#333", fontweight="bold")

ax2.set_xticks(range(len(teams_sorted)))
ax2.set_xticklabels(short, fontsize=9.5)
ax2.set_ylabel("Avg distinct players per sequence", fontsize=10)
ax2.set_title(
    "Average Players Used: Build-Up → Attack Sequences\n"
    "Black tick = median",
    fontsize=12, fontweight="bold", color="#1a1a1a",
)
ax2.set_ylim(0, max(avgs) * 1.3)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=0)
ax2.legend(fontsize=9, frameon=False)

plt.tight_layout()
out2 = OUT / "sequences_avg_players.png"
plt.savefig(out2, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out2.name}")

# ── FIGURE 3: strip / dot plot (each sequence as a dot) ──────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor="white")
ax3.set_facecolor("#f9f9f9")

for i, team in enumerate(teams_sorted):
    color  = TEAM_COLORS.get(team, "#555")
    n_list = [s["n_players"] for s in all_sequences[team]]
    # jitter to avoid overplotting
    rng    = np.random.default_rng(seed=i)
    jitter = rng.uniform(-0.18, 0.18, size=len(n_list))
    ax3.scatter(np.array(n_list) + jitter,
                [i] * len(n_list),
                color=color, alpha=0.55, s=28, zorder=3,
                edgecolors="white", linewidths=0.4)
    # mean marker
    ax3.scatter([np.mean(n_list)], [i],
                color=color, s=120, zorder=5,
                edgecolors="#333333", linewidths=1.2, marker="D")

ax3.set_yticks(range(len(teams_sorted)))
ax3.set_yticklabels([t.replace(" W", "") for t in teams_sorted], fontsize=10)
ax3.set_xlabel("Distinct players in sequence", fontsize=11)
ax3.set_title(
    "Distribution of Player Involvement per Build-Up → Attack Sequence\n"
    "Each dot = one sequence  |  ◆ = mean",
    fontsize=12, fontweight="bold", color="#1a1a1a",
)
ax3.grid(axis="x", color="#e0e0e0", linewidth=0.7, zorder=0)
ax3.spines[["top", "right"]].set_visible(False)

# vertical mean reference line across all teams
grand_mean = np.mean([s["n_players"]
                       for seqs in all_sequences.values() for s in seqs])
ax3.axvline(grand_mean, color="#999999", linestyle="--", linewidth=1,
            label=f"Overall mean = {grand_mean:.2f}")
ax3.legend(fontsize=9, frameon=False)

plt.tight_layout()
out3 = OUT / "sequences_strip.png"
plt.savefig(out3, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out3.name}")

# ── FIGURE 4: stacked % bars (2-player, 3-player, 4+) ────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5), facecolor="white")
ax4.set_facecolor("#f8f8f8")

categories = {
    "2 players":  lambda x: x == 2,
    "3 players":  lambda x: x == 3,
    "4 players":  lambda x: x == 4,
    "5+ players": lambda x: x >= 5,
}
cat_colors = ["#AED6F1", "#2980B9", "#1A5276", "#0A2333"]

bottoms = np.zeros(len(teams_sorted))
for (label, fn), col in zip(categories.items(), cat_colors):
    vals = []
    for team in teams_sorted:
        n_list = [s["n_players"] for s in all_sequences[team]]
        vals.append(100 * sum(fn(x) for x in n_list) / len(n_list))
    vals = np.array(vals)
    ax4.bar(range(len(teams_sorted)), vals, bottom=bottoms,
            color=col, label=label, edgecolor="white", width=0.55, zorder=3)
    # label inside bar if segment is wide enough
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 5:
            ax4.text(i, b + v / 2, f"{v:.0f}%",
                     ha="center", va="center", fontsize=8.5,
                     color="white", fontweight="bold")
    bottoms += vals

ax4.set_xticks(range(len(teams_sorted)))
ax4.set_xticklabels([t.replace(" W", "") for t in teams_sorted], fontsize=9.5)
ax4.set_ylabel("% of sequences", fontsize=10)
ax4.set_ylim(0, 105)
ax4.set_title(
    "Player Involvement Breakdown — Build-Up → Attack Sequences",
    fontsize=12, fontweight="bold", color="#1a1a1a",
)
ax4.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
ax4.spines[["top", "right"]].set_visible(False)
ax4.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=0)

plt.tight_layout()
out4 = OUT / "sequences_stacked.png"
plt.savefig(out4, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out4.name}")

print(f"\nAll outputs in: {OUT}")
