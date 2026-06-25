"""
Coach/Team Profiling with Data
Inspired by: https://analyticsfc.co.uk/blog/2021/03/22/profiling-coaches-with-data/

Metrics
-------
  High Press        – pressing intensity (inverse PPDA); lower PPDA = more pressing
  Counter Play      – proportion of possessions starting as direct counter-attacks
  Low Block         – share of defensive actions applied in own half (deep defending)
  Long Balls        – directness from defensive areas (high/long passes from own half)
  Deep Circulation  – proportion of passes in own defensive third that stay short / go back
  Wing Play         – concentration of passing + carrying actions in wide channels
"""

import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── config ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
OUT  = BASE / "output"
OUT.mkdir(exist_ok=True)

DATA_FILES = [
    "2025-09-06_PSV_Eindhoven_W_vs_Excelsior_W_4014321.json",
    "2025-10-12_Heerenveen_W_vs_HERA_United_W_4014348.json",
    "2025-11-16_Heerenveen_W_vs_PEC_Zwolle_W_4014359.json",
]

TEAM_COLORS = {
    "PSV Eindhoven W": "#E3000F",
    "Excelsior W":     "#B8860B",
    "Heerenveen W":    "#003DA5",
    "HERA United W":   "#7B2D8B",
    "PEC Zwolle W":    "#009FE3",
}

LABELS = [
    "High Press",
    "Counter Play",
    "Low Block",
    "Long Balls",
    "Deep Circulation",
    "Wing Play",
]

METRIC_DESC = {
    "High Press":        "Pressing intensity\n(inverse PPDA)",
    "Counter Play":      "Share of possessions\nstarting as counters",
    "Low Block":         "Defensive actions\napplied in own half",
    "Long Balls":        "High/long passes\nfrom defensive areas",
    "Deep Circulation":  "Short/back passes\nin defensive third",
    "Wing Play":         "Passes & carries\nin wide channels",
}

# ── coordinate helpers ────────────────────────────────────────────────────────
# StatsBomb pitch: 120 × 80 yds; attacking direction varies per half.
def nx(x, d): return 120 - x if d == "right_to_left" else x
def ny(y, d): return  80 - y if d == "right_to_left" else y

# ── metric computation ────────────────────────────────────────────────────────
def raw_metrics(events, team_name):
    team_ev = [e for e in events if e.get("team", {}).get("name") == team_name]
    opp_ev  = [e for e in events
                if e.get("team") and e["team"]["name"] != team_name]

    # 1 · HIGH PRESS — PPDA ─────────────────────────────────────────────────
    # opponent passes in their own half / team defensive actions in that same zone
    opp_passes_oh = 0
    for e in opp_ev:
        if e["type"]["name"] != "Pass" or not e.get("location"):
            continue
        d = e.get("attacking_direction", "left_to_right")
        if nx(e["location"][0], d) < 60:          # opponent's own half
            opp_passes_oh += 1

    team_def_oh = 0
    for e in team_ev:
        if e["type"]["name"] not in ("Pressure", "Interception", "Block"):
            continue
        if not e.get("location"):
            continue
        d = e.get("attacking_direction", "left_to_right")
        if nx(e["location"][0], d) >= 60:          # team's attacking half
            team_def_oh += 1

    ppda = opp_passes_oh / team_def_oh if team_def_oh else 20.0

    # 2 · COUNTER PLAY ──────────────────────────────────────────────────────
    all_poss     = {(e["possession"], e["possession_team"]["id"]) for e in team_ev}
    counter_poss = {(e["possession"], e["possession_team"]["id"])
                    for e in team_ev if e["play_pattern"]["name"] == "From Counter"}
    counter_pct = len(counter_poss) / len(all_poss) if all_poss else 0

    # 3 · LOW BLOCK ─────────────────────────────────────────────────────────
    press_total = press_own = 0
    for e in team_ev:
        if e["type"]["name"] != "Pressure" or not e.get("location"):
            continue
        d = e.get("attacking_direction", "left_to_right")
        press_total += 1
        if nx(e["location"][0], d) < 60:
            press_own += 1
    low_block_pct = press_own / press_total if press_total else 0

    # 4 · LONG BALLS ────────────────────────────────────────────────────────
    pass_own_total = long_count = 0
    for e in team_ev:
        if e["type"]["name"] != "Pass" or not e.get("location"):
            continue
        d = e.get("attacking_direction", "left_to_right")
        if nx(e["location"][0], d) >= 60:
            continue
        p = e.get("pass", {})
        pass_own_total += 1
        if p.get("height", {}).get("name") == "High Pass" or p.get("length", 0) >= 32:
            long_count += 1
    long_pct = long_count / pass_own_total if pass_own_total else 0

    # 5 · DEEP CIRCULATION ──────────────────────────────────────────────────
    def3_total = def3_circ = 0
    for e in team_ev:
        if e["type"]["name"] != "Pass" or not e.get("location"):
            continue
        d = e.get("attacking_direction", "left_to_right")
        x0 = nx(e["location"][0], d)
        if x0 >= 40:
            continue
        end = e.get("pass", {}).get("end_location")
        if not end:
            continue
        x1 = nx(end[0], d)
        def3_total += 1
        if x1 <= x0 + 5:          # stays put or goes backward
            def3_circ += 1
    deep_circ_pct = def3_circ / def3_total if def3_total else 0

    # 6 · WING PLAY ─────────────────────────────────────────────────────────
    # passes whose start OR end falls in wide channel (y < 20 or y > 60)
    pass_total_w = pass_wide = 0
    for e in team_ev:
        if e["type"]["name"] != "Pass" or not e.get("location"):
            continue
        d   = e.get("attacking_direction", "left_to_right")
        y0  = ny(e["location"][1], d)
        end = e.get("pass", {}).get("end_location")
        y1  = ny(end[1], d) if end else None
        pass_total_w += 1
        if y0 < 20 or y0 > 60 or (y1 is not None and (y1 < 20 or y1 > 60)):
            pass_wide += 1
    # also count carries in wide channels
    carry_total = carry_wide = 0
    for e in team_ev:
        if e["type"]["name"] != "Carry" or not e.get("location"):
            continue
        d   = e.get("attacking_direction", "left_to_right")
        y0  = ny(e["location"][1], d)
        end = e.get("carry", {}).get("end_location")
        y1  = ny(end[1], d) if end else None
        carry_total += 1
        if y0 < 20 or y0 > 60 or (y1 is not None and (y1 < 20 or y1 > 60)):
            carry_wide += 1
    total_actions = pass_total_w + carry_total
    wide_actions  = pass_wide   + carry_wide
    wing_pct = wide_actions / total_actions if total_actions else 0

    return dict(
        ppda          = ppda,
        counter_pct   = counter_pct,
        low_block_pct = low_block_pct,
        long_pct      = long_pct,
        deep_circ_pct = deep_circ_pct,
        wing_pct      = wing_pct,
    )


def scale_metrics(raw):
    """Convert raw percentages/values to 0–1 scores using observed ranges."""
    # High Press: PPDA 1.5 → score 1.0 (intense),  PPDA 7.0 → score 0.0
    high_press = max(0.0, min(1.0, (7.0 - raw["ppda"]) / 5.5))

    # Counter Play: 0 % → 0.0,  3 % → 1.0
    counter = min(1.0, raw["counter_pct"] / 0.030)

    # Low Block: 35 % → 0.0,  75 % → 1.0
    low_block = max(0.0, min(1.0, (raw["low_block_pct"] - 0.35) / 0.40))

    # Long Balls: 15 % → 0.0,  40 % → 1.0
    long_balls = max(0.0, min(1.0, (raw["long_pct"] - 0.15) / 0.25))

    # Deep Circulation: 38 % → 0.0,  65 % → 1.0
    deep_circ = max(0.0, min(1.0, (raw["deep_circ_pct"] - 0.38) / 0.27))

    # Wing Play: 40 % → 0.0,  65 % → 1.0
    wing = max(0.0, min(1.0, (raw["wing_pct"] - 0.40) / 0.25))

    return {
        "High Press":        round(high_press, 3),
        "Counter Play":      round(counter,    3),
        "Low Block":         round(low_block,  3),
        "Long Balls":        round(long_balls, 3),
        "Deep Circulation":  round(deep_circ,  3),
        "Wing Play":         round(wing,       3),
    }


# ── load & aggregate ─────────────────────────────────────────────────────────
team_raw_list   = {}   # team -> [raw_dict, ...]
for fname in DATA_FILES:
    with open(BASE / fname) as f:
        events = json.load(f)
    teams = {e["team"]["name"] for e in events if e.get("team")}
    for t in teams:
        r = raw_metrics(events, t)
        team_raw_list.setdefault(t, []).append(r)

# Average raws, then scale once
team_metrics = {}
team_raw_avg = {}
for t, raws in team_raw_list.items():
    avg_raw = {k: sum(r[k] for r in raws) / len(raws) for k in raws[0]}
    team_raw_avg[t] = avg_raw
    team_metrics[t] = scale_metrics(avg_raw)

teams_sorted = sorted(team_metrics)

print("Raw averages:")
for t in teams_sorted:
    r = team_raw_avg[t]
    print(f"  {t}: PPDA={r['ppda']:.2f}  counter={r['counter_pct']:.2%}  "
          f"lowblock={r['low_block_pct']:.2%}  long={r['long_pct']:.2%}  "
          f"deepcirc={r['deep_circ_pct']:.2%}  wing={r['wing_pct']:.2%}")

print("\nScaled metrics:")
for t in teams_sorted:
    print(f"  {t}: {team_metrics[t]}")

# ── CHART HELPERS ─────────────────────────────────────────────────────────────
N      = len(LABELS)
angles = [n / N * 2 * math.pi for n in range(N)]
angles_closed = angles + angles[:1]

def _radar(ax, vals, color, alpha_fill=0.20, lw=2.0, label=None):
    v = list(vals) + [vals[0]]
    ax.plot(angles_closed, v, color=color, linewidth=lw, zorder=3, label=label)
    ax.fill(angles_closed, v, color=color, alpha=alpha_fill, zorder=2)
    ax.scatter(angles, vals, color=color, s=35, zorder=4, edgecolors="white", linewidths=0.8)

def _style_radar(ax, title=None, color=None):
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(LABELS, size=8.5, fontweight="bold", color="#222222")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25", "50", "75", "100"], size=6, color="#aaaaaa")
    ax.tick_params(axis="y", pad=2)
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.5, zorder=1)
    ax.spines["polar"].set_color("#cccccc")
    ax.set_facecolor("#fafafa")
    if title:
        ax.set_title(title, size=10, fontweight="bold",
                     color=color or "#333333", pad=18)

# ── FIGURE 1 — Individual radar per team ─────────────────────────────────────
ncols = 3
nrows = math.ceil(len(teams_sorted) / ncols)
fig1 = plt.figure(figsize=(15, 5 * nrows), facecolor="#ffffff")
fig1.suptitle(
    "Team Tactical Profiles — Dutch Women's Football\n"
    "Metrics inspired by Analytics FC · Profiling Coaches with Data",
    fontsize=13, fontweight="bold", color="#1a1a1a", y=1.00,
)

for i, team in enumerate(teams_sorted):
    ax = fig1.add_subplot(nrows, ncols, i + 1, projection="polar")
    color = TEAM_COLORS.get(team, "#555555")
    vals  = [team_metrics[team][lbl] for lbl in LABELS]
    _radar(ax, vals, color, alpha_fill=0.25, lw=2.2)
    _style_radar(ax, title=team, color=color)
    # annotate values
    for angle, val in zip(angles, vals):
        ax.text(angle, val + 0.10, f"{val:.2f}",
                ha="center", va="center", fontsize=6.5, color=color,
                fontweight="bold")

for j in range(i + 1, nrows * ncols):
    fig1.add_subplot(nrows, ncols, j + 1).set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = OUT / "01_team_profiles_individual.png"
plt.savefig(out1, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved {out1.name}")

# ── FIGURE 2 — All teams overlaid on one radar ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"},
                          facecolor="white")
for team in teams_sorted:
    color = TEAM_COLORS.get(team, "#555555")
    vals  = [team_metrics[team][lbl] for lbl in LABELS]
    _radar(ax2, vals, color, alpha_fill=0.10, lw=2.2, label=team)

_style_radar(ax2, title="Tactical Style Comparison\nDutch Women's Football")

handles = [mpatches.Patch(color=TEAM_COLORS.get(t, "#555"), label=t)
           for t in teams_sorted]
ax2.legend(handles=handles, loc="upper right",
           bbox_to_anchor=(1.42, 1.18), fontsize=9.5,
           frameon=True, framealpha=0.9, edgecolor="#dddddd")

plt.tight_layout()
out2 = OUT / "02_team_profiles_comparison.png"
plt.savefig(out2, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out2.name}")

# ── FIGURE 3 — Bar chart breakdown per metric ─────────────────────────────────
fig3, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
fig3.suptitle("Metric-by-Metric Breakdown", fontsize=13,
              fontweight="bold", color="#1a1a1a", y=1.01)

short_names = {
    "PSV Eindhoven W": "PSV\nEindhoven W",
    "Excelsior W":     "Excelsior W",
    "Heerenveen W":    "Heerenveen W",
    "HERA United W":   "HERA\nUnited W",
    "PEC Zwolle W":    "PEC\nZwolle W",
}

for idx, lbl in enumerate(LABELS):
    ax = axes[idx // 3][idx % 3]
    vals   = [team_metrics[t][lbl] for t in teams_sorted]
    colors = [TEAM_COLORS.get(t, "#555") for t in teams_sorted]
    xlabels = [short_names.get(t, t) for t in teams_sorted]
    bars = ax.bar(range(len(vals)), vals, color=colors,
                  edgecolor="white", width=0.6, linewidth=1.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score (0–1)", fontsize=8, color="#888")
    ax.set_title(lbl, fontweight="bold", fontsize=10, color="#222222", pad=6)
    ax.set_facecolor("#f8f8f8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=2)
    # subtitle
    ax.text(0.5, -0.22, METRIC_DESC[lbl],
            ha="center", va="top", transform=ax.transAxes,
            fontsize=7, color="#888888", fontstyle="italic")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)

plt.tight_layout()
out3 = OUT / "03_metric_breakdown.png"
plt.savefig(out3, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out3.name}")

# ── FIGURE 4 — Heatmap / summary table ──────────────────────────────────────
df = pd.DataFrame(
    {t: [team_metrics[t][lbl] for lbl in LABELS] for t in teams_sorted},
    index=LABELS,
).T   # teams as rows, metrics as cols

fig4, ax4 = plt.subplots(figsize=(11, 4), facecolor="white")
ax4.set_facecolor("white")

# draw heatmap cells
for ci, lbl in enumerate(LABELS):
    for ri, team in enumerate(teams_sorted):
        val = df.loc[team, lbl]
        bg  = plt.cm.RdYlGn(val)
        rect = plt.Rectangle([ci - 0.5, ri - 0.5], 1, 1,
                              color=bg, zorder=1)
        ax4.add_patch(rect)
        ax4.text(ci, ri, f"{val:.2f}",
                 ha="center", va="center", fontsize=9,
                 fontweight="bold",
                 color="white" if val < 0.25 or val > 0.75 else "#333333",
                 zorder=2)

ax4.set_xticks(range(len(LABELS)))
ax4.set_xticklabels(LABELS, fontsize=9.5, fontweight="bold", color="#222")
ax4.set_yticks(range(len(teams_sorted)))
ax4.set_yticklabels(teams_sorted, fontsize=9)

# colour the y-tick labels by team colour
for tick, team in zip(ax4.get_yticklabels(), teams_sorted):
    tick.set_color(TEAM_COLORS.get(team, "#333"))
    tick.set_fontweight("bold")

ax4.set_xlim(-0.5, len(LABELS) - 0.5)
ax4.set_ylim(-0.5, len(teams_sorted) - 0.5)
ax4.set_title("Tactical Profile Summary — Score Heatmap (0 = low · 1 = high)",
              fontsize=12, fontweight="bold", pad=12, color="#1a1a1a")
ax4.tick_params(length=0)
for spine in ax4.spines.values():
    spine.set_visible(False)

# colour bar
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = fig4.colorbar(sm, ax=ax4, orientation="vertical",
                     fraction=0.025, pad=0.02)
cbar.set_label("Score", fontsize=9)

plt.tight_layout()
out4 = OUT / "04_metrics_heatmap.png"
plt.savefig(out4, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out4.name}")

print(f"\nAll outputs in: {OUT}")
