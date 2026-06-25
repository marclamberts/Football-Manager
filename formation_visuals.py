"""
Formation Build-Up — Extended Visuals
======================================
Re-uses the sequence extraction from formation_sequences.py and produces
10 additional charts, each answering a different tactical question.

Visual  1  Pitch flow arrows                  (improved, per team)
Visual  2  Positional chain strings           (top 8 per team)
Visual  3  Position → position heatmap        (transition %)
Visual  4  Step-by-step position heatmap      (which position at step N?)
Visual  5  Who starts / who enters att. third (bar charts)
Visual  6  Position involvement frequency     (how often each pos touches ball)
Visual  7  NetworkX circular graph            (weighted position network)
Visual  8  Sankey zones                       (DEF → MID → ATT flow counts)
Visual  9  All passes on pitch               (coloured by position group)
Visual 10  Chain length distribution          (steps per sequence)
Visual 11  First-3 triplets                   (most common 3-step openings)
Visual 12  Ball progression per step          (avg x at step 1, 2, 3 …)
"""

import json, math, textwrap
from pathlib import Path
from collections import defaultdict, Counter
from itertools import groupby

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba, Normalize
import matplotlib.cm as cm
import networkx as nx
from mplsoccer import Pitch

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
GROUP_ORDER  = ["GK","DEF","WB","DM","MID","ATT"]
GROUP_COLOR  = {
    "GK":"#F5A623","DEF":"#4A90D9","WB":"#7ED321",
    "DM":"#BD10E0","MID":"#9B59B6","ATT":"#E74C3C",
}

POS_XY = {
    "GK":(4,40),
    "RB":(25,64),"RCB":(20,56),"CB":(20,40),"LCB":(20,24),"LB":(25,16),
    "RWB":(42,74),"LWB":(42,6),
    "RDM":(42,60),"CDM":(42,40),"LDM":(42,20),
    "RM":(55,72),"RCM":(52,58),"CM":(52,40),"LCM":(52,22),"LM":(55,8),
    "RW":(82,70),"RAM":(72,60),"CAM":(72,40),"LAM":(72,20),"LW":(82,10),
    "RCF":(96,54),"ST":(100,40),"LCF":(96,26),"SS":(88,40),
}

ON_BALL = {"Pass","Ball Receipt*","Dribble","Shot"}

def norm_x(x,d): return 120-x if d=="right_to_left" else x
def norm_y(y,d): return  80-y if d=="right_to_left" else y
def event_pos(e):
    pid = e.get("position",{}).get("id")
    return POS_LABEL.get(pid,"?") if pid else "?"
def pos_grp(p): return POS_GROUP.get(p,"MID")

# ── load & extract sequences ──────────────────────────────────────────────────
all_events = {}
for fname in DATA_FILES:
    with open(BASE/fname) as f:
        all_events[fname] = json.load(f)

def get_sequences(events, team_name):
    chains = defaultdict(list)
    for e in events:
        if e.get("team",{}).get("name") == team_name:
            chains[e["possession"]].append(e)

    seqs = []
    for poss_id, evs in chains.items():
        ob = [e for e in evs if e["type"]["name"] in ON_BALL
              and e.get("location") and e.get("player")]
        if len(ob) < 2: continue

        xs = [norm_x(e["location"][0], e.get("attacking_direction","left_to_right")) for e in ob]
        ys = [norm_y(e["location"][1], e.get("attacking_direction","left_to_right")) for e in ob]

        if xs[0] > 60 or max(xs) < 80: continue

        pos_raw = [event_pos(e) for e in ob]
        pos_raw = [p for p in pos_raw if p != "?"]
        chain   = [k for k,_ in groupby(pos_raw)]
        if len(chain) < 2: continue

        # pass pairs (passer pos → next receipt pos)
        pairs = []
        for i,e in enumerate(ob):
            if e["type"]["name"] != "Pass": continue
            p_pos = event_pos(e)
            if p_pos == "?": continue
            for nxt in ob[i+1:]:
                if nxt["type"]["name"] == "Ball Receipt*":
                    r_pos = event_pos(nxt)
                    if r_pos != "?": pairs.append((p_pos, r_pos))
                    break

        # who enters the attacking third (position of last event before x crosses 80)
        att_entry_pos = None
        for e, x in zip(ob, xs):
            if x >= 80:
                att_entry_pos = event_pos(e)
                if att_entry_pos != "?": break

        # pass locations in build-up zone (x < 80)
        pass_locs = [
            (norm_x(e["location"][0], e.get("attacking_direction","left_to_right")),
             norm_y(e["location"][1], e.get("attacking_direction","left_to_right")),
             event_pos(e))
            for e in ob
            if e["type"]["name"] == "Pass"
            and norm_x(e["location"][0], e.get("attacking_direction","left_to_right")) < 80
        ]

        seqs.append(dict(
            poss_id   = poss_id,
            team      = team_name,
            chain     = chain,
            chain_str = " → ".join(chain),
            pairs     = pairs,
            xs        = xs,
            ys        = ys,
            ob        = ob,
            start_pos = chain[0],
            end_pos   = chain[-1],
            att_entry = att_entry_pos,
            pass_locs = pass_locs,
            n_steps   = len(chain),
        ))
    return seqs

team_seqs      = {}
team_formation = {}

for fname, events in all_events.items():
    teams = {e["team"]["name"] for e in events if e.get("team")}
    for e in events:
        if e["type"]["name"] == "Starting XI":
            t = e["team"]["name"]
            if t not in team_formation:
                team_formation[t] = e["tactics"]["formation"]
    for team in teams:
        team_seqs.setdefault(team,[]).extend(get_sequences(events,team))

teams_sorted = sorted(team_seqs)
n_teams = len(teams_sorted)

def tc(t): return TEAM_COLORS.get(t,"#555555")
def fmt(t): return f"{t}  [{team_formation.get(t,'?')}]"
def shortname(t): return t.replace(" W","")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 4 — Step-by-step position heatmap
# Which position appears at step 1, 2, 3, 4, 5 of build-up chains?
# ═══════════════════════════════════════════════════════════════════════════════
MAX_STEP = 7
all_positions = sorted({p for seqs in team_seqs.values()
                          for s in seqs for p in s["chain"]})

fig4, axes4 = plt.subplots(1, n_teams, figsize=(4*n_teams, 7), facecolor="white")
fig4.suptitle("Which Position Has the Ball at Each Build-Up Step?\n(Step 1 = sequence start)",
              fontsize=13, fontweight="bold", y=1.01)

for ax, team in zip(axes4, teams_sorted):
    seqs = team_seqs[team]
    mat  = np.zeros((len(all_positions), MAX_STEP))
    for s in seqs:
        for step, pos in enumerate(s["chain"][:MAX_STEP]):
            if pos in all_positions:
                mat[all_positions.index(pos), step] += 1
    # normalise each step column to %
    col_sum = mat.sum(axis=0, keepdims=True)
    mat_pct = np.where(col_sum>0, mat/col_sum*100, 0)

    im = ax.imshow(mat_pct, cmap="YlOrRd", aspect="auto", vmin=0, vmax=70)
    ax.set_xticks(range(MAX_STEP))
    ax.set_xticklabels([f"Step {i+1}" for i in range(MAX_STEP)], fontsize=7, rotation=45)
    ax.set_yticks(range(len(all_positions)))
    ax.set_yticklabels(all_positions, fontsize=7.5)
    ax.set_title(fmt(team), fontsize=8.5, fontweight="bold", color=tc(team), pad=6)
    for i in range(len(all_positions)):
        for j in range(MAX_STEP):
            v = mat_pct[i,j]
            if v > 10:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if v>45 else "#333", fontweight="bold")

plt.tight_layout()
out4 = OUT/"vis4_step_heatmap.png"
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out4.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 5 — Who STARTS build-up  /  Who ENTERS attacking third
# ═══════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(2, n_teams, figsize=(3.5*n_teams, 8), facecolor="white")
fig5.suptitle("Who Starts Build-Up (top) vs. Who Enters the Attacking Third (bottom)",
              fontsize=13, fontweight="bold", y=1.01)

for col, team in enumerate(teams_sorted):
    seqs  = team_seqs[team]
    color = tc(team)

    for row, (key, title) in enumerate([("start_pos","Sequence initiator"),
                                         ("att_entry","Att. third entry")]):
        ax   = axes5[row, col]
        vals = [s[key] for s in seqs if s[key] and s[key]!="?"]
        cnt  = Counter(vals).most_common(8)
        positions = [x[0] for x in cnt]
        counts    = [x[1] for x in cnt]
        bar_cols  = [GROUP_COLOR[pos_grp(p)] for p in positions]
        bars = ax.barh(range(len(positions)), counts, color=bar_cols,
                       edgecolor="white", height=0.65)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions, fontsize=8)
        ax.set_xlabel("# sequences", fontsize=7)
        ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f9f9f9")
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(shortname(team), fontsize=9, fontweight="bold", color=color)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    str(val), va="center", fontsize=7)

    axes5[0,0].set_ylabel("Sequence\ninitiator", fontsize=8, color="#555")
    axes5[1,0].set_ylabel("Att. third\nentry pos", fontsize=8, color="#555")

plt.tight_layout()
out5 = OUT/"vis5_start_end.png"
plt.savefig(out5, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out5.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 6 — Position involvement frequency (how often in build-up chains)
# ═══════════════════════════════════════════════════════════════════════════════
fig6, ax6 = plt.subplots(figsize=(13, 6), facecolor="white")
ax6.set_facecolor("#f8f8f8")

# collect per team
pos_freq = {}
for team in teams_sorted:
    cnt = Counter(p for s in team_seqs[team] for p in s["chain"])
    total = sum(cnt.values())
    pos_freq[team] = {p: 100*v/total for p,v in cnt.items()} if total else {}

all_pos_used = sorted({p for d in pos_freq.values() for p in d},
                       key=lambda p: GROUP_ORDER.index(pos_grp(p)) if pos_grp(p) in GROUP_ORDER else 99)

x     = np.arange(len(all_pos_used))
width = 0.8 / n_teams

for i, team in enumerate(teams_sorted):
    vals   = [pos_freq[team].get(p,0) for p in all_pos_used]
    offset = (i - n_teams/2 + 0.5) * width
    ax6.bar(x + offset, vals, width=width*0.9, color=tc(team),
            label=shortname(team), alpha=0.88, edgecolor="white")

ax6.set_xticks(x)
ax6.set_xticklabels(all_pos_used, fontsize=8.5, rotation=30)
ax6.set_ylabel("% of chain touches", fontsize=10)
ax6.set_title("Position Involvement in Build-Up Sequences (% of chain touches)",
              fontsize=12, fontweight="bold")
ax6.legend(fontsize=8.5, frameon=True, framealpha=0.9)
ax6.spines[["top","right"]].set_visible(False)
ax6.grid(axis="y", color="#e0e0e0", linewidth=0.7)

# shade by group
grp_bounds = {}
for i,p in enumerate(all_pos_used):
    g = pos_grp(p)
    grp_bounds.setdefault(g,[]).append(i)
for g, idxs in grp_bounds.items():
    ax6.axvspan(min(idxs)-0.5, max(idxs)+0.5, alpha=0.06,
                color=GROUP_COLOR[g], zorder=0)
    ax6.text((min(idxs)+max(idxs))/2, ax6.get_ylim()[1]*0.96, g,
             ha="center", va="top", fontsize=7, color=GROUP_COLOR[g], fontweight="bold")

plt.tight_layout()
out6 = OUT/"vis6_pos_involvement.png"
plt.savefig(out6, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out6.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 7 — NetworkX circular position graph (per team)
# ═══════════════════════════════════════════════════════════════════════════════
fig7, axes7 = plt.subplots(2, 3, figsize=(18, 12), facecolor="#111122")
fig7.suptitle("Position Transition Network — Build-Up Sequences",
              fontsize=14, fontweight="bold", color="white", y=1.01)

for idx, team in enumerate(teams_sorted):
    ax = axes7[idx//3, idx%3]
    ax.set_facecolor("#111122")
    seqs = team_seqs[team]

    G = nx.DiGraph()
    pair_cnt = Counter(p for s in seqs for p in s["pairs"])
    pos_cnt  = Counter(p for s in seqs for p in s["chain"])

    for (src,dst), w in pair_cnt.items():
        G.add_edge(src, dst, weight=w)

    if not G.nodes():
        ax.set_title(fmt(team), color=tc(team)); continue

    # layout: arrange by group left→right
    pos_layout = {}
    groups_present = {}
    for node in G.nodes():
        g = pos_grp(node)
        groups_present.setdefault(g,[]).append(node)

    group_x = {"GK":0.05,"DEF":0.18,"WB":0.28,"DM":0.42,"MID":0.56,"ATT":0.82}
    for g, nodes in groups_present.items():
        gx = group_x.get(g, 0.5)
        ys = np.linspace(0.1, 0.9, len(nodes))
        for node, y in zip(sorted(nodes), ys):
            pos_layout[node] = (gx, y)

    max_w   = max(d["weight"] for _,_,d in G.edges(data=True))
    max_cnt = max(pos_cnt.values()) if pos_cnt else 1

    node_sizes  = [300 + 1800 * pos_cnt.get(n,0)/max_cnt for n in G.nodes()]
    node_colors = [GROUP_COLOR[pos_grp(n)] for n in G.nodes()]
    edge_widths = [0.5 + 4.5*G[u][v]["weight"]/max_w for u,v in G.edges()]
    edge_alphas = [0.2 + 0.7*G[u][v]["weight"]/max_w for u,v in G.edges()]

    nx.draw_networkx_nodes(G, pos_layout, ax=ax,
                           node_size=node_sizes, node_color=node_colors,
                           edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(G, pos_layout, ax=ax,
                            font_size=6.5, font_color="white", font_weight="bold")
    for (u,v), lw, alpha in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(G, pos_layout, edgelist=[(u,v)], ax=ax,
                               width=lw, alpha=alpha,
                               edge_color=tc(team),
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.18",
                               min_source_margin=12, min_target_margin=12)

    ax.set_title(fmt(team), fontsize=10, fontweight="bold", color=tc(team), pad=8)
    ax.axis("off")

for j in range(n_teams, 6):
    axes7[j//3, j%3].set_visible(False)

legend_patches = [mpatches.Patch(color=c,label=g) for g,c in GROUP_COLOR.items()]
fig7.legend(handles=legend_patches, loc="lower center", ncol=6,
            fontsize=9, frameon=False, labelcolor="white", bbox_to_anchor=(0.5,-0.01))
plt.tight_layout()
out7 = OUT/"vis7_network.png"
plt.savefig(out7, dpi=150, bbox_inches="tight", facecolor="#111122"); plt.close()
print(f"Saved {out7.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 8 — Sankey: DEF → DM → MID → ATT zone flow
# ═══════════════════════════════════════════════════════════════════════════════
ZONE_LABEL = {"GK":"GK","DEF":"DEF","WB":"WB","DM":"DM","MID":"MID","ATT":"ATT"}
ZONE_ORDER  = ["GK","DEF","WB","DM","MID","ATT"]
ZONE_X      = dict(zip(ZONE_ORDER, np.linspace(0.05, 0.95, len(ZONE_ORDER))))

fig8, axes8 = plt.subplots(1, n_teams, figsize=(5*n_teams, 5), facecolor="white")
fig8.suptitle("Build-Up Flow: Zone-to-Zone Pass Distribution",
              fontsize=13, fontweight="bold", y=1.01)

from matplotlib.patches import FancyArrowPatch

for ax, team in zip(axes8, teams_sorted):
    ax.set_facecolor("#fafafa")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    seqs  = team_seqs[team]
    color = tc(team)

    # count zone→zone transitions
    zone_pairs = Counter()
    for s in seqs:
        grps = [pos_grp(p) for p in s["chain"]]
        for a,b in zip(grps, grps[1:]):
            if a != b:
                zone_pairs[(a,b)] += 1

    if not zone_pairs:
        ax.set_title(shortname(team), fontsize=9, color=color); continue

    # draw zone boxes
    zone_counts = Counter(g for s in seqs for p in s["chain"] for g in [pos_grp(p)])
    max_cnt = max(zone_counts.values()) if zone_counts else 1

    for zone in ZONE_ORDER:
        zx = ZONE_X[zone]
        cnt = zone_counts.get(zone, 0)
        h   = 0.08 + 0.25*(cnt/max_cnt)
        rect = mpatches.FancyBboxPatch((zx-0.05, 0.5-h/2), 0.10, h,
                                        boxstyle="round,pad=0.01",
                                        facecolor=GROUP_COLOR[zone], alpha=0.9,
                                        edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(zx, 0.5, zone, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        ax.text(zx, 0.5-h/2-0.06, str(cnt), ha="center", va="top",
                fontsize=7, color="#555")

    # draw arrows
    max_pair = max(zone_pairs.values())
    for (z1,z2), cnt in zone_pairs.items():
        if z1 not in ZONE_X or z2 not in ZONE_X: continue
        x1, x2 = ZONE_X[z1], ZONE_X[z2]
        lw = 0.8 + 4.5*cnt/max_pair
        alpha = 0.3 + 0.6*cnt/max_pair
        rad   = -0.3 if x1 < x2 else 0.3
        ax.annotate("", xy=(x2-0.05 if x2>x1 else x2+0.05, 0.5),
                    xytext=(x1+0.05 if x2>x1 else x1-0.05, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, alpha=alpha,
                                   connectionstyle=f"arc3,rad={rad}",
                                   mutation_scale=10))
        mx = (x1+x2)/2
        ax.text(mx, 0.5+0.18*np.sign(rad), str(cnt), ha="center",
                fontsize=6.5, color=color, fontweight="bold")

    ax.set_title(shortname(team), fontsize=9, fontweight="bold", color=color, pad=6)

plt.tight_layout()
out8 = OUT/"vis8_zone_sankey.png"
plt.savefig(out8, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out8.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 9 — All build-up passes on pitch, coloured by position group
# ═══════════════════════════════════════════════════════════════════════════════
fig9, axes9 = plt.subplots(2, 3, figsize=(18, 12), facecolor="#0e1117")
fig9.suptitle("Build-Up Pass Locations by Position Group",
              fontsize=14, fontweight="bold", color="white", y=1.01)

for idx, team in enumerate(teams_sorted):
    ax = axes9[idx//3, idx%3]
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0e1117",
                  line_color="#333355", linewidth=1)
    pitch.draw(ax=ax)

    seqs = team_seqs[team]
    for s in seqs:
        for (px, py, pos) in s["pass_locs"]:
            grp = pos_grp(pos)
            col = GROUP_COLOR[grp]
            ax.scatter(px, py, c=col, s=22, alpha=0.65, zorder=3,
                       edgecolors="none")

    # shade attacking third
    ax.axvspan(80, 120, alpha=0.06, color="white", zorder=0)
    ax.text(100, 78, "Att. third", ha="center", va="top",
            color="white", alpha=0.4, fontsize=8)

    ax.set_title(fmt(team), fontsize=10, fontweight="bold",
                 color=tc(team), pad=6)

for j in range(n_teams, 6):
    axes9[j//3, j%3].set_visible(False)

legend_patches = [mpatches.Patch(color=GROUP_COLOR[g],label=g) for g in GROUP_ORDER]
fig9.legend(handles=legend_patches, loc="lower center", ncol=6,
            fontsize=9, frameon=False, labelcolor="white", bbox_to_anchor=(0.5,-0.01))
plt.tight_layout()
out9 = OUT/"vis9_pass_map.png"
plt.savefig(out9, dpi=150, bbox_inches="tight", facecolor="#0e1117"); plt.close()
print(f"Saved {out9.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 10 — Chain length distribution (how many positional steps per sequence)
# ═══════════════════════════════════════════════════════════════════════════════
fig10, ax10 = plt.subplots(figsize=(10, 5), facecolor="white")
ax10.set_facecolor("#f8f8f8")

max_len = max(s["n_steps"] for seqs in team_seqs.values() for s in seqs)
bins    = np.arange(1, max_len+2) - 0.5
width   = 0.8/n_teams

for i, team in enumerate(teams_sorted):
    lengths = [s["n_steps"] for s in team_seqs[team]]
    counts  = [lengths.count(v) for v in range(1, max_len+1)]
    pcts    = [100*c/len(lengths) for c in counts]
    offset  = (i - n_teams/2 + 0.5)*width
    ax10.bar(np.arange(1, max_len+1)+offset, pcts, width=width*0.9,
             color=tc(team), label=shortname(team), alpha=0.88, edgecolor="white")

ax10.set_xlabel("Positional steps in chain (collapsed)", fontsize=11)
ax10.set_ylabel("% of sequences", fontsize=11)
ax10.set_title("Build-Up Chain Length Distribution\n(how many distinct positions touched before reaching final third)",
               fontsize=12, fontweight="bold")
ax10.set_xticks(range(1, max_len+1))
ax10.legend(fontsize=9, frameon=True, framealpha=0.9)
ax10.spines[["top","right"]].set_visible(False)
ax10.grid(axis="y", color="#e0e0e0", linewidth=0.7)

# median lines
for team in teams_sorted:
    med = np.median([s["n_steps"] for s in team_seqs[team]])
    ax10.axvline(med, color=tc(team), linewidth=1.5, linestyle="--", alpha=0.6)

plt.tight_layout()
out10 = OUT/"vis10_chain_length.png"
plt.savefig(out10, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out10.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 11 — Most common first-3 positional triplets
# ═══════════════════════════════════════════════════════════════════════════════
fig11, axes11 = plt.subplots(1, n_teams, figsize=(4.5*n_teams, 6), facecolor="white")
fig11.suptitle("Most Common Opening 3-Step Positional Triplets",
               fontsize=13, fontweight="bold", y=1.01)

for ax, team in zip(axes11, teams_sorted):
    seqs    = team_seqs[team]
    triplets = Counter(
        " → ".join(s["chain"][:3]) for s in seqs if len(s["chain"]) >= 3
    ).most_common(8)

    labels = [t[0] for t in triplets]
    counts = [t[1] for t in triplets]
    colors = [GROUP_COLOR[pos_grp(lbl.split(" → ")[0])] for lbl in labels]
    ys     = range(len(labels)-1, -1, -1)

    ax.barh(list(ys), counts, color=colors, edgecolor="white", height=0.6, alpha=0.88)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Count", fontsize=8)
    ax.set_title(shortname(team), fontsize=9, fontweight="bold", color=tc(team))
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    ax.tick_params(labelsize=7.5)
    for y, cnt in zip(ys, counts):
        ax.text(cnt+0.05, y, str(cnt), va="center", fontsize=7)

plt.tight_layout()
out11 = OUT/"vis11_triplets.png"
plt.savefig(out11, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out11.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# VIS 12 — Ball progression: avg x position at each chain step
# ═══════════════════════════════════════════════════════════════════════════════
fig12, ax12 = plt.subplots(figsize=(10, 6), facecolor="white")
ax12.set_facecolor("#f8f8f8")

for team in teams_sorted:
    seqs = team_seqs[team]
    step_xs = defaultdict(list)
    for s in seqs:
        # x at each step using the first event of each "collapsed" position run
        # Re-trace: group ob events by collapsed position
        ob = s["ob"]
        xs_raw = [norm_x(e["location"][0], e.get("attacking_direction","left_to_right"))
                  for e in ob if e.get("location")]
        pos_raw = [event_pos(e) for e in ob if event_pos(e)!="?"]
        # zip and group
        pairs_xp = [(x,p) for x,p in zip(xs_raw, pos_raw)]
        step = 0
        prev = None
        for x, p in pairs_xp:
            if p != prev:
                step_xs[step].append(x)
                step += 1
                prev = p

    max_step = max(step_xs.keys()) if step_xs else 0
    steps = range(min(10, max_step+1))
    avgs  = [np.mean(step_xs[s]) if step_xs[s] else np.nan for s in steps]
    stds  = [np.std(step_xs[s])  if step_xs[s] else 0      for s in steps]

    ax12.plot(list(steps), avgs, color=tc(team), lw=2.2, marker="o", ms=6,
              label=shortname(team), zorder=3)
    ax12.fill_between(list(steps),
                      [a-s for a,s in zip(avgs,stds)],
                      [a+s for a,s in zip(avgs,stds)],
                      color=tc(team), alpha=0.10, zorder=2)

ax12.axhline(60, color="#888", linestyle=":", linewidth=1, label="Halfway line")
ax12.axhline(80, color="#e74c3c", linestyle=":", linewidth=1, label="Attacking third")
ax12.set_xlabel("Positional step in chain", fontsize=11)
ax12.set_ylabel("Avg normalised x (0=own goal, 120=opp goal)", fontsize=10)
ax12.set_title("How Quickly Does the Ball Progress Through Build-Up?\n(shaded band = ±1 std dev)",
               fontsize=12, fontweight="bold")
ax12.legend(fontsize=9, frameon=True, framealpha=0.9)
ax12.spines[["top","right"]].set_visible(False)
ax12.grid(color="#e0e0e0", linewidth=0.7)
ax12.set_xticks(range(10))
ax12.set_xticklabels([f"Step {i+1}" for i in range(10)], rotation=20, fontsize=8.5)

plt.tight_layout()
out12 = OUT/"vis12_ball_progression.png"
plt.savefig(out12, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print(f"Saved {out12.name}")

print(f"\n✅  All visuals saved to {OUT}")
