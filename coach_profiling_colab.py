# @title 🏟️ Team Tactical Profiling — Dutch Women's Football
# Inspired by Analytics FC: Profiling Coaches with Data
# Loads all JSON match files from Google Drive → /Football Manager/Events/

# ── Cell 1: Mount Google Drive & install deps ────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run(['pip', 'install', 'mplsoccer', '-q'])

# ── Cell 2: Imports ──────────────────────────────────────────────────────────
import json, math, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from IPython.display import display

# ── Cell 3: Load all JSON files from Drive ───────────────────────────────────
EVENTS_DIR = Path('/content/drive/MyDrive/Football Manager/Events')

json_files = sorted(EVENTS_DIR.glob('*.json'))
print(f"Found {len(json_files)} match file(s):")
for f in json_files:
    print(f"  {f.name}")

all_events = {}
for f in json_files:
    with open(f) as fh:
        all_events[f.name] = json.load(fh)
    print(f"  Loaded {f.name}  ({len(all_events[f.name])} events)")

# ── Cell 4: Team colours (extend as needed) ──────────────────────────────────
TEAM_COLORS = {
    "PSV Eindhoven W": "#E3000F",
    "Excelsior W":     "#B8860B",
    "Heerenveen W":    "#003DA5",
    "HERA United W":   "#7B2D8B",
    "PEC Zwolle W":    "#009FE3",
    # add more teams here if needed
}
DEFAULT_PALETTE = [
    "#E74C3C","#3498DB","#2ECC71","#F39C12",
    "#9B59B6","#1ABC9C","#E67E22","#34495E",
]

def team_color(name):
    if name in TEAM_COLORS:
        return TEAM_COLORS[name]
    # auto-assign from palette based on hash
    return DEFAULT_PALETTE[hash(name) % len(DEFAULT_PALETTE)]

# ── Cell 5: Metric definitions ───────────────────────────────────────────────
LABELS = [
    "High Press",
    "Counter Play",
    "Low Block",
    "Long Balls",
    "Deep Circulation",
    "Wing Play",
    "Territory",
    "GK Build-Up",
    "Crossing",
]

METRIC_DESC = {
    "High Press":        "Pressing intensity\n(inverse PPDA)",
    "Counter Play":      "Share of possessions\nstarting as counters",
    "Low Block":         "Defensive actions\napplied in own half",
    "Long Balls":        "High/long passes\nfrom defensive areas",
    "Deep Circulation":  "Short/back passes\nin defensive third",
    "Wing Play":         "Passes & carries\nin wide channels",
    "Territory":         "Average pitch position\nof on-ball actions",
    "GK Build-Up":       "GK short/ground passes\nvs. total GK passes",
    "Crossing":          "Crosses as share\nof total passes",
}

# ── Cell 6: Coordinate helpers ───────────────────────────────────────────────
# StatsBomb pitch: 120 × 80 yards; attacking direction varies per half.
def nx(x, d): return 120 - x if d == "right_to_left" else x
def ny(y, d): return  80 - y if d == "right_to_left" else y

# ── Cell 7: Compute raw metrics ──────────────────────────────────────────────
def raw_metrics(events, team_name):
    team_ev = [e for e in events if e.get("team", {}).get("name") == team_name]
    opp_ev  = [e for e in events
                if e.get("team") and e["team"]["name"] != team_name]

    # 1 · HIGH PRESS — PPDA
    opp_passes_oh = sum(
        1 for e in opp_ev
        if e["type"]["name"] == "Pass"
        and e.get("location")
        and nx(e["location"][0], e.get("attacking_direction", "left_to_right")) < 60
    )
    team_def_oh = sum(
        1 for e in team_ev
        if e["type"]["name"] in ("Pressure", "Interception", "Block")
        and e.get("location")
        and nx(e["location"][0], e.get("attacking_direction", "left_to_right")) >= 60
    )
    ppda = opp_passes_oh / team_def_oh if team_def_oh else 20.0

    # 2 · COUNTER PLAY
    all_poss     = {(e["possession"], e["possession_team"]["id"]) for e in team_ev}
    counter_poss = {(e["possession"], e["possession_team"]["id"])
                    for e in team_ev if e["play_pattern"]["name"] == "From Counter"}
    counter_pct = len(counter_poss) / len(all_poss) if all_poss else 0.0

    # 3 · LOW BLOCK
    press_total = press_own = 0
    for e in team_ev:
        if e["type"]["name"] != "Pressure" or not e.get("location"):
            continue
        d = e.get("attacking_direction", "left_to_right")
        press_total += 1
        if nx(e["location"][0], d) < 60:
            press_own += 1
    low_block_pct = press_own / press_total if press_total else 0.0

    # 4 · LONG BALLS
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
    long_pct = long_count / pass_own_total if pass_own_total else 0.0

    # 5 · DEEP CIRCULATION
    def3_total = def3_circ = 0
    for e in team_ev:
        if e["type"]["name"] != "Pass" or not e.get("location"):
            continue
        d  = e.get("attacking_direction", "left_to_right")
        x0 = nx(e["location"][0], d)
        if x0 >= 40:
            continue
        end = e.get("pass", {}).get("end_location")
        if not end:
            continue
        def3_total += 1
        if nx(end[0], d) <= x0 + 5:
            def3_circ += 1
    deep_circ_pct = def3_circ / def3_total if def3_total else 0.0

    # 6 · WING PLAY (passes + carries in wide channels y<20 or y>60)
    pass_total_w = pass_wide = 0
    for e in team_ev:
        if e["type"]["name"] != "Pass" or not e.get("location"):
            continue
        d  = e.get("attacking_direction", "left_to_right")
        y0 = ny(e["location"][1], d)
        end = e.get("pass", {}).get("end_location")
        y1  = ny(end[1], d) if end else None
        pass_total_w += 1
        if y0 < 20 or y0 > 60 or (y1 is not None and (y1 < 20 or y1 > 60)):
            pass_wide += 1
    carry_total = carry_wide = 0
    for e in team_ev:
        if e["type"]["name"] != "Carry" or not e.get("location"):
            continue
        d  = e.get("attacking_direction", "left_to_right")
        y0 = ny(e["location"][1], d)
        end = e.get("carry", {}).get("end_location")
        y1  = ny(end[1], d) if end else None
        carry_total += 1
        if y0 < 20 or y0 > 60 or (y1 is not None and (y1 < 20 or y1 > 60)):
            carry_wide += 1
    wing_pct = (pass_wide + carry_wide) / (pass_total_w + carry_total) \
               if (pass_total_w + carry_total) else 0.0

    # 7 · TERRITORY
    ON_BALL = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*"}
    x_vals = [
        nx(e["location"][0], e.get("attacking_direction", "left_to_right"))
        for e in team_ev
        if e["type"]["name"] in ON_BALL and e.get("location")
    ]
    territory_avg_x = sum(x_vals) / len(x_vals) if x_vals else 60.0

    # 8 · GK BUILD-UP
    gk_passes = [e for e in team_ev
                 if e["type"]["name"] == "Pass"
                 and e.get("position", {}).get("name") == "Goalkeeper"]
    gk_short  = [p for p in gk_passes
                 if p.get("pass", {}).get("height", {}).get("name") in ("Ground Pass", "Low Pass")
                 and p.get("pass", {}).get("length", 99) < 35]
    gk_buildup_ratio = len(gk_short) / len(gk_passes) if gk_passes else 0.0

    # 9 · CROSSING
    passes_all = [e for e in team_ev if e["type"]["name"] == "Pass"]
    crosses    = [p for p in passes_all if p.get("pass", {}).get("cross")]
    cross_pct  = len(crosses) / len(passes_all) if passes_all else 0.0

    return dict(
        ppda             = ppda,
        counter_pct      = counter_pct,
        low_block_pct    = low_block_pct,
        long_pct         = long_pct,
        deep_circ_pct    = deep_circ_pct,
        wing_pct         = wing_pct,
        territory_avg_x  = territory_avg_x,
        gk_buildup_ratio = gk_buildup_ratio,
        cross_pct        = cross_pct,
    )


def scale_metrics(raw):
    """Map raw values to 0–1 scores using observed reference ranges."""
    return {
        "High Press":       round(max(0., min(1., (7.0  - raw["ppda"])             / 5.5)),  3),
        "Counter Play":     round(min(1.,             raw["counter_pct"]            / 0.030), 3),
        "Low Block":        round(max(0., min(1., (raw["low_block_pct"]   - 0.35)  / 0.40)), 3),
        "Long Balls":       round(max(0., min(1., (raw["long_pct"]        - 0.15)  / 0.25)), 3),
        "Deep Circulation": round(max(0., min(1., (raw["deep_circ_pct"]   - 0.38)  / 0.27)), 3),
        "Wing Play":        round(max(0., min(1., (raw["wing_pct"]        - 0.40)  / 0.25)), 3),
        "Territory":        round(max(0., min(1., (raw["territory_avg_x"] - 45.0)  / 20.0)), 3),
        "GK Build-Up":      round(max(0., min(1., (raw["gk_buildup_ratio"]- 0.20)  / 0.60)), 3),
        "Crossing":         round(min(1.,             raw["cross_pct"]             / 0.05),  3),
    }


# ── Cell 8: Aggregate across all matches ─────────────────────────────────────
team_raw_list = {}
for fname, events in all_events.items():
    teams = {e["team"]["name"] for e in events if e.get("team")}
    for t in teams:
        team_raw_list.setdefault(t, []).append(raw_metrics(events, t))

team_metrics = {}
team_raw_avg = {}
for t, raws in team_raw_list.items():
    avg_raw = {k: sum(r[k] for r in raws) / len(raws) for k in raws[0]}
    team_raw_avg[t] = avg_raw
    team_metrics[t] = scale_metrics(avg_raw)

teams_sorted = sorted(team_metrics)

print("\n── Scaled Metrics ──────────────────────────────────────────────────")
df_display = pd.DataFrame(team_metrics).T[LABELS]
display(df_display.style
        .format("{:.2f}")
        .background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=1)
        .set_caption("Tactical Profile Scores  (0 = low · 1 = high)"))

# ── Cell 9: Chart helpers ─────────────────────────────────────────────────────
N      = len(LABELS)
angles = [n / N * 2 * math.pi for n in range(N)]
angles_c = angles + angles[:1]

def _radar(ax, vals, color, alpha_fill=0.20, lw=2.0, label=None):
    v = list(vals) + [vals[0]]
    ax.plot(angles_c, v, color=color, linewidth=lw, zorder=3, label=label)
    ax.fill(angles_c, v, color=color, alpha=alpha_fill, zorder=2)
    ax.scatter(angles, vals, color=color, s=35, zorder=4,
               edgecolors="white", linewidths=0.8)

def _style_radar(ax, title=None, color=None):
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(LABELS, size=8, fontweight="bold", color="#222")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25", "50", "75", "100"], size=6, color="#aaa")
    ax.tick_params(axis="y", pad=2)
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.5, zorder=1)
    ax.spines["polar"].set_color("#cccccc")
    ax.set_facecolor("#fafafa")
    if title:
        ax.set_title(title, size=10, fontweight="bold",
                     color=color or "#333", pad=18)

# ── Cell 10: Fig 1 — Individual radar per team ───────────────────────────────
ncols = 3
nrows = math.ceil(len(teams_sorted) / ncols)

fig1 = plt.figure(figsize=(15, 5 * nrows), facecolor="white")
fig1.suptitle(
    "Team Tactical Profiles — Dutch Women's Football\n"
    "Metrics inspired by Analytics FC · Profiling Coaches with Data",
    fontsize=13, fontweight="bold", color="#1a1a1a", y=1.00,
)

for i, team in enumerate(teams_sorted):
    ax = fig1.add_subplot(nrows, ncols, i + 1, projection="polar")
    color = team_color(team)
    vals  = [team_metrics[team][lbl] for lbl in LABELS]
    _radar(ax, vals, color, alpha_fill=0.25, lw=2.2)
    _style_radar(ax, title=team, color=color)
    for angle, val in zip(angles, vals):
        ax.text(angle, val + 0.10, f"{val:.2f}",
                ha="center", va="center", fontsize=6.5,
                color=color, fontweight="bold")

for j in range(i + 1, nrows * ncols):
    fig1.add_subplot(nrows, ncols, j + 1).set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("team_profiles_individual.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.show()

# ── Cell 11: Fig 2 — All teams overlaid ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"},
                          facecolor="white")
for team in teams_sorted:
    color = team_color(team)
    vals  = [team_metrics[team][lbl] for lbl in LABELS]
    _radar(ax2, vals, color, alpha_fill=0.10, lw=2.2, label=team)

_style_radar(ax2, title="Tactical Style Comparison\nDutch Women's Football")
handles = [mpatches.Patch(color=team_color(t), label=t) for t in teams_sorted]
ax2.legend(handles=handles, loc="upper right",
           bbox_to_anchor=(1.45, 1.18), fontsize=9.5,
           frameon=True, framealpha=0.9, edgecolor="#ddd")

plt.tight_layout()
plt.savefig("team_profiles_comparison.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.show()

# ── Cell 12: Fig 3 — Metric breakdown bars (3×3) ─────────────────────────────
fig3, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor="white")
fig3.suptitle("Metric-by-Metric Breakdown", fontsize=13,
              fontweight="bold", color="#1a1a1a", y=1.01)

def short(name):
    return name.replace(" W", "").replace(" United", "\nUnited")

for idx, lbl in enumerate(LABELS):
    ax = axes[idx // 3, idx % 3]
    vals   = [team_metrics[t][lbl] for t in teams_sorted]
    colors = [team_color(t) for t in teams_sorted]
    bars = ax.bar(range(len(vals)), vals, color=colors,
                  edgecolor="white", width=0.6, linewidth=1.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([short(t) for t in teams_sorted], fontsize=7.5)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score (0–1)", fontsize=8, color="#888")
    ax.set_title(lbl, fontweight="bold", fontsize=10, color="#222", pad=6)
    ax.set_facecolor("#f8f8f8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.axhline(0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=2)
    ax.text(0.5, -0.22, METRIC_DESC[lbl], ha="center", va="top",
            transform=ax.transAxes, fontsize=7, color="#888", fontstyle="italic")
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.5, zorder=0)

plt.tight_layout()
plt.savefig("metric_breakdown.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.show()

# ── Cell 13: Fig 4 — Heatmap summary ─────────────────────────────────────────
df = pd.DataFrame(
    {t: [team_metrics[t][lbl] for lbl in LABELS] for t in teams_sorted},
    index=LABELS,
).T

fig4, ax4 = plt.subplots(figsize=(13, 0.7 * len(teams_sorted) + 2), facecolor="white")
ax4.set_facecolor("white")

for ci, lbl in enumerate(LABELS):
    for ri, team in enumerate(teams_sorted):
        val = df.loc[team, lbl]
        bg  = plt.cm.RdYlGn(val)
        ax4.add_patch(plt.Rectangle([ci - 0.5, ri - 0.5], 1, 1, color=bg, zorder=1))
        ax4.text(ci, ri, f"{val:.2f}", ha="center", va="center", fontsize=9,
                 fontweight="bold",
                 color="white" if val < 0.25 or val > 0.75 else "#333",
                 zorder=2)

ax4.set_xticks(range(len(LABELS)))
ax4.set_xticklabels(LABELS, fontsize=9.5, fontweight="bold", color="#222")
ax4.set_yticks(range(len(teams_sorted)))
ax4.set_yticklabels(teams_sorted, fontsize=9)
for tick, team in zip(ax4.get_yticklabels(), teams_sorted):
    tick.set_color(team_color(team))
    tick.set_fontweight("bold")

ax4.set_xlim(-0.5, len(LABELS) - 0.5)
ax4.set_ylim(-0.5, len(teams_sorted) - 0.5)
ax4.set_title("Tactical Profile Summary — Score Heatmap  (0 = low · 1 = high)",
              fontsize=12, fontweight="bold", pad=12, color="#1a1a1a")
ax4.tick_params(length=0)
for spine in ax4.spines.values():
    spine.set_visible(False)

sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(0, 1))
sm.set_array([])
fig4.colorbar(sm, ax=ax4, orientation="vertical", fraction=0.015, pad=0.02).set_label("Score", fontsize=9)

plt.tight_layout()
plt.savefig("metrics_heatmap.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.show()

print("\n✅ All 4 charts saved to the Colab working directory.")
print("   To save to Drive, copy them:")
print("   !cp *.png '/content/drive/MyDrive/Football Manager/Output/'")
