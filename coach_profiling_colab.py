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

print("\n✅ Charts 1–4 saved to the Colab working directory.")

# ── Cell 14: Percentile ranks ─────────────────────────────────────────────────
# For each metric, rank each team relative to all teams in the dataset (0–100).
# scipy percentileofscore uses 'rank' method: average of lower/upper bounds.
from scipy.stats import percentileofscore

df_scores = pd.DataFrame(team_metrics).T[LABELS]   # teams × metrics, values 0–1

df_pct = df_scores.copy()
for col in LABELS:
    col_vals = df_scores[col].values.tolist()
    n = len(col_vals)
    df_pct[col] = [
        # scale 0-100 rank into 1-99 range
        round(1 + (percentileofscore(col_vals, v, kind="rank") / 100) * 98, 1)
        for v in col_vals
    ]

print("\n── Percentile Ranks (1 = lowest · 99 = highest in this dataset) ──")
display(df_pct.style
        .format("{:.0f}")
        .background_gradient(cmap="RdYlGn", axis=None, vmin=1, vmax=99)
        .set_caption("Percentile Ranks across all teams (1–99)"))

# ── Cell 15: Pizza plots — one per team ───────────────────────────────────────
from mplsoccer import PyPizza, add_image
from matplotlib.colors import to_rgba

# Slice colours based on metric category
SLICE_COLORS = {
    "High Press":        "#E74C3C",   # red   – pressing
    "Counter Play":      "#E67E22",   # orange – transition
    "Low Block":         "#F1C40F",   # yellow – defending
    "Long Balls":        "#F1C40F",
    "Deep Circulation":  "#2ECC71",   # green  – build-up
    "Wing Play":         "#2ECC71",
    "Territory":         "#3498DB",   # blue   – positional
    "GK Build-Up":       "#3498DB",
    "Crossing":          "#9B59B6",   # purple – delivery
}
slice_colors = [SLICE_COLORS[lbl] for lbl in LABELS]
text_colors  = ["white"] * len(LABELS)

baker = PyPizza(
    params            = LABELS,
    background_color  = "#FFFFFF",
    straight_line_color= "#CCCCCC",
    straight_line_lw  = 1,
    last_circle_color = "#CCCCCC",
    last_circle_lw    = 2,
    other_circle_lw   = 1,
    other_circle_color= "#EEEEEE",
    inner_circle_size = 10,
)

for team in teams_sorted:
    color  = team_color(team)
    values = df_pct.loc[team, LABELS].tolist()
    values_int = [int(v) for v in values]

    fig_p, ax_p = baker.make_pizza(
        values_int,
        figsize           = (8, 8),
        color_blank_space = "same",
        slice_colors      = slice_colors,
        value_colors      = text_colors,
        value_bck_colors  = slice_colors,
        blank_alpha        = 0.4,
        kwargs_slices     = dict(edgecolor="#FFFFFF", zorder=2, linewidth=1),
        kwargs_params     = dict(color="#222222", fontsize=10,
                                  fontweight="bold", va="center"),
        kwargs_values     = dict(color="#FFFFFF", fontsize=10,
                                  fontweight="bold", zorder=3,
                                  bbox=dict(edgecolor="#FFFFFF", facecolor=color,
                                            boxstyle="round,pad=0.2", lw=1.5)),
    )

    # Title
    fig_p.text(0.515, 0.975, team,
               size=16, fontweight="bold", color=color,
               ha="center", va="top")
    fig_p.text(0.515, 0.945,
               "Tactical Profile  |  Percentile vs. dataset",
               size=10, color="#555555", ha="center", va="top")

    # Subtitle credits
    fig_p.text(0.515, 0.02,
               "Metrics inspired by Analytics FC · Profiling Coaches with Data",
               size=7, color="#aaaaaa", ha="center", va="bottom")

    fname_out = f"pizza_{team.replace(' ', '_')}.png"
    plt.savefig(fname_out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"  Saved {fname_out}")

# ── Cell 16: Pizza comparison — all teams small multiples ────────────────────
ncols_p = 3
nrows_p = math.ceil(len(teams_sorted) / ncols_p)
fig_all = plt.figure(figsize=(6 * ncols_p, 6.5 * nrows_p), facecolor="white")
fig_all.suptitle(
    "Tactical Pizza Plots — Percentile Ranks\nDutch Women's Football",
    fontsize=14, fontweight="bold", color="#1a1a1a", y=1.01,
)

for i, team in enumerate(teams_sorted):
    color      = team_color(team)
    values_int = [int(v) for v in df_pct.loc[team, LABELS].tolist()]

    baker_small = PyPizza(
        params             = LABELS,
        background_color   = "#FFFFFF",
        straight_line_color= "#CCCCCC",
        straight_line_lw   = 0.8,
        last_circle_color  = "#CCCCCC",
        last_circle_lw     = 1.5,
        other_circle_lw    = 0.8,
        other_circle_color = "#EEEEEE",
        inner_circle_size  = 10,
    )

    # PyPizza needs its own figure; we'll draw it then transfer the axes
    fig_tmp, ax_tmp = baker_small.make_pizza(
        values_int,
        figsize            = (5, 5),
        color_blank_space  = "same",
        slice_colors       = slice_colors,
        value_colors       = text_colors,
        value_bck_colors   = slice_colors,
        blank_alpha         = 0.4,
        kwargs_slices      = dict(edgecolor="#FFFFFF", zorder=2, linewidth=0.8),
        kwargs_params      = dict(color="#222222", fontsize=7.5,
                                   fontweight="bold", va="center"),
        kwargs_values      = dict(color="#FFFFFF", fontsize=8,
                                   fontweight="bold", zorder=3,
                                   bbox=dict(edgecolor="#FFFFFF", facecolor=color,
                                             boxstyle="round,pad=0.15", lw=1)),
    )
    fig_tmp.text(0.5, 0.97, team, size=10, fontweight="bold",
                 color=color, ha="center", va="top")

    fname_tmp = f"_pizza_tmp_{i}.png"
    fig_tmp.savefig(fname_tmp, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig_tmp)

    # load image into grid figure
    from matplotlib.image import imread
    img = imread(fname_tmp)
    ax_sub = fig_all.add_subplot(nrows_p, ncols_p, i + 1)
    ax_sub.imshow(img)
    ax_sub.axis("off")

for j in range(i + 1, nrows_p * ncols_p):
    fig_all.add_subplot(nrows_p, ncols_p, j + 1).set_visible(False)

plt.tight_layout()
plt.savefig("pizza_all_teams.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

# tidy temp files
import os
for i in range(len(teams_sorted)):
    try: os.remove(f"_pizza_tmp_{i}.png")
    except: pass

print("\n✅ Pizza charts saved.")

# ── Cell 17: Percentile rank bar charts — one per team ───────────────────────
# Horizontal bars, metric label on left, percentile value on right,
# background track shows the full 0–100 range.

BAR_COLORS = {
    "High Press":        "#E74C3C",
    "Counter Play":      "#E67E22",
    "Low Block":         "#F1C40F",
    "Long Balls":        "#F1C40F",
    "Deep Circulation":  "#2ECC71",
    "Wing Play":         "#2ECC71",
    "Territory":         "#3498DB",
    "GK Build-Up":       "#3498DB",
    "Crossing":          "#9B59B6",
}

for team in teams_sorted:
    color  = team_color(team)
    vals   = df_pct.loc[team, LABELS].tolist()
    bar_c  = [BAR_COLORS[lbl] for lbl in LABELS]

    fig_b, ax_b = plt.subplots(figsize=(7, 5.5), facecolor="#F5F5F5")
    ax_b.set_facecolor("#F5F5F5")

    y_pos = range(len(LABELS) - 1, -1, -1)   # top-to-bottom label order

    # Background track (full 100)
    for y in y_pos:
        ax_b.barh(y, 100, color="#E0E0E0", height=0.6, zorder=1)

    # Actual percentile bar
    for y, val, bc in zip(y_pos, vals, bar_c):
        ax_b.barh(y, val, color=bc, height=0.6, zorder=2, alpha=0.90)
        # value label inside or outside depending on space
        x_txt = val - 3 if val >= 15 else val + 2
        ha_txt = "right" if val >= 15 else "left"
        txt_col = "white" if val >= 15 else "#333333"
        ax_b.text(x_txt, y, f"{int(val)}",
                  va="center", ha=ha_txt, fontsize=9.5,
                  fontweight="bold", color=txt_col, zorder=3)

    # Reference lines
    for xref in [25, 50, 75]:
        ax_b.axvline(xref, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)

    ax_b.set_yticks(list(y_pos))
    ax_b.set_yticklabels(list(reversed(LABELS)), fontsize=9.5, fontweight="bold", color="#222")
    ax_b.set_xlim(0, 102)
    ax_b.set_xlabel("Percentile Rank", fontsize=9, color="#666")
    ax_b.set_xticks([1, 25, 50, 75, 99])
    ax_b.xaxis.set_tick_params(labelsize=8, colors="#888")
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.spines["bottom"].set_color("#cccccc")
    ax_b.tick_params(axis="y", length=0)

    # Titles
    fig_b.text(0.13, 0.97, team,
               fontsize=14, fontweight="bold", color=color,
               va="top", ha="left")
    fig_b.text(0.13, 0.925,
               "Tactical Profile  ·  Percentile vs. dataset",
               fontsize=9, color="#888888", va="top", ha="left")
    fig_b.text(0.98, 0.01,
               "Inspired by Analytics FC · Profiling Coaches with Data",
               fontsize=6.5, color="#bbbbbb", va="bottom", ha="right")

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    fname_b = f"bar_{team.replace(' ', '_')}.png"
    plt.savefig(fname_b, dpi=160, bbox_inches="tight", facecolor="#F5F5F5")
    plt.show()
    print(f"  Saved {fname_b}")

# ── Cell 18: Bar chart small-multiples grid ───────────────────────────────────
ncols_b = 3
nrows_b = math.ceil(len(teams_sorted) / ncols_b)
fig_bg  = plt.figure(figsize=(7 * ncols_b, 6 * nrows_b), facecolor="white")
fig_bg.suptitle(
    "Tactical Percentile Ranks — Dutch Women's Football",
    fontsize=15, fontweight="bold", color="#1a1a1a", y=1.01,
)

for i, team in enumerate(teams_sorted):
    color  = team_color(team)
    vals   = df_pct.loc[team, LABELS].tolist()
    bar_c  = [BAR_COLORS[lbl] for lbl in LABELS]
    y_pos  = range(len(LABELS) - 1, -1, -1)

    ax_s = fig_bg.add_subplot(nrows_b, ncols_b, i + 1)
    ax_s.set_facecolor("#F7F7F7")

    for y in y_pos:
        ax_s.barh(y, 100, color="#E4E4E4", height=0.6, zorder=1)

    for y, val, bc in zip(y_pos, vals, bar_c):
        ax_s.barh(y, val, color=bc, height=0.6, zorder=2, alpha=0.88)
        x_txt = val - 2 if val >= 12 else val + 1.5
        ha_txt = "right" if val >= 12 else "left"
        txt_col = "white" if val >= 12 else "#444"
        ax_s.text(x_txt, y, f"{int(val)}",
                  va="center", ha=ha_txt, fontsize=8,
                  fontweight="bold", color=txt_col, zorder=3)

    for xref in [25, 50, 75]:
        ax_s.axvline(xref, color="#cccccc", linewidth=0.6, linestyle="--", zorder=0)

    ax_s.set_yticks(list(y_pos))
    ax_s.set_yticklabels(list(reversed(LABELS)), fontsize=8, color="#333")
    ax_s.set_xlim(0, 102)
    ax_s.set_xticks([1, 25, 50, 75, 99])
    ax_s.xaxis.set_tick_params(labelsize=7, colors="#999")
    ax_s.spines[["top", "right", "left"]].set_visible(False)
    ax_s.spines["bottom"].set_color("#dddddd")
    ax_s.tick_params(axis="y", length=0)
    ax_s.set_title(team, fontsize=10, fontweight="bold",
                   color=color, pad=8)

for j in range(i + 1, nrows_b * ncols_b):
    fig_bg.add_subplot(nrows_b, ncols_b, j + 1).set_visible(False)

plt.tight_layout()
plt.savefig("bar_all_teams.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

print("\n✅ All charts saved to the Colab working directory.")
print("   To save to Drive, copy them:")
print("   !cp *.png '/content/drive/MyDrive/Football Manager/Output/'")
