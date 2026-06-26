"""
Defensive Reorganization Model (DRM)
=====================================
Models how defensive structure changes between shots within the same possession,
and quantifies the opportunity window created by defensive disorganization.

Theoretical Framework:
----------------------
After shot k-1 in a possession, the defense transitions from a disorganized
state back to a consolidated state. Two competing forces act on shot quality:

  1. GK Displacement Effect: After a save, the GK is displaced from their
     optimal position. This creates extra visible goal area for shot k.
     Recovery: G(t) = G_0 * exp(-lambda_gk * t)

  2. Defender Consolidation: Outfield defenders rush back to block lanes.
     Accumulation: D(t) = D_max * (1 - exp(-lambda_d * t))

The tension between these two forces defines a narrow "opportunity window"
in time where xG for the second shot peaks.

xG Decomposition for shot k > 0:
  xG_k = xG_base(loc_k) + beta_G * G_k - beta_D * n_def_k + epsilon_k

Execution Degradation under pressure:
  Uplift_k = alpha_0 + alpha_1 * (1/delta_t_k) + alpha_2 * min_def_dist_k
"""

import json
import math
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. DATA EXTRACTION
# ─────────────────────────────────────────────

FILES = [
    '/home/user/Football-Manager/2025-09-06_PSV_Eindhoven_W_vs_Excelsior_W_4014321.json',
    '/home/user/Football-Manager/2025-10-12_Heerenveen_W_vs_HERA_United_W_4014348.json',
    '/home/user/Football-Manager/2025-11-16_Heerenveen_W_vs_PEC_Zwolle_W_4014359.json',
]

GOAL_POSTS = np.array([[120, 36], [120, 44]])
GOAL_CENTER = np.array([120, 40])
PITCH_LENGTH = 120
DEFENDER_RADIUS = 0.4  # metres (body width proxy)


def ts_to_sec(ts):
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def dist2d(a, b):
    a, b = np.array(a[:2]), np.array(b[:2])
    return float(np.linalg.norm(a - b))


def unoccluded_goal_angle(sloc, freeze_frame):
    """
    Compute visible goal mouth angle (radians) after subtracting angular
    occlusion from defenders in the shooting lane.

    Method:
      - Total goal angle: atan2 subtended by goal posts from sloc
      - Each defender in [sloc_x, 120] range occludes 2*arcsin(r/d) radians
      - Occlusion is clipped at total goal angle (cannot be negative)
    """
    sx, sy = sloc[0], sloc[1]
    a1 = math.atan2(GOAL_POSTS[0, 1] - sy, GOAL_POSTS[0, 0] - sx)
    a2 = math.atan2(GOAL_POSTS[1, 1] - sy, GOAL_POSTS[1, 0] - sx)
    total_angle = abs(a2 - a1)

    defenders = [
        p for p in freeze_frame
        if not p['teammate'] and p['position']['name'] != 'Goalkeeper'
        and p['location'][0] > sx  # only defenders between shooter and goal
    ]
    occ_total = 0.0
    for d in defenders:
        d_dist = dist2d(sloc, d['location'])
        if d_dist < 0.01:
            continue
        half_occ = math.asin(min(DEFENDER_RADIUS / d_dist, 1.0))
        occ_total += 2 * half_occ

    return max(0.0, total_angle - occ_total)


def gk_lateral_displacement(sloc, freeze_frame):
    """
    GK lateral displacement from the optimal positioning line.

    Optimal GK position: along the bisector of the angle from shooter to
    goal posts. We compute the bisector direction and find how far the GK
    sits off that line laterally.

    Returns displacement in metres (always >= 0), or None if no GK in frame.
    """
    gk_list = [p for p in freeze_frame if p['position']['name'] == 'Goalkeeper' and not p['teammate']]
    if not gk_list:
        return None

    sloc = np.array(sloc[:2])
    gk_loc = np.array(gk_list[0]['location'][:2])

    v1 = GOAL_POSTS[0] - sloc
    v2 = GOAL_POSTS[1] - sloc
    bisector = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
    bisector_norm = np.linalg.norm(bisector)
    if bisector_norm < 1e-9:
        return float(np.linalg.norm(gk_loc - GOAL_CENTER))

    bisector_unit = bisector / bisector_norm
    gk_vec = gk_loc - sloc

    # Component of gk_vec perpendicular to bisector = lateral displacement
    proj = np.dot(gk_vec, bisector_unit) * bisector_unit
    lateral = gk_vec - proj
    return float(np.linalg.norm(lateral))


def shot_to_goal_dist(sloc):
    return dist2d(sloc, GOAL_CENTER)


def extract_sequence_features(events):
    """
    For each possession with >= 1 shot, extract per-shot feature rows.
    Returns list of dicts, one per shot.
    """
    shots = [e for e in events if e.get('type', {}).get('name') == 'Shot']
    poss_shots = defaultdict(list)
    for s in shots:
        poss_shots[s['possession']].append(s)

    rows = []
    for poss_id, slist in sorted(poss_shots.items()):
        slist.sort(key=lambda x: x['index'])
        n = len(slist)

        for k, s in enumerate(slist):
            sh = s['shot']
            ff = sh.get('freeze_frame', [])
            loc = s['location']

            defenders = [
                p for p in ff
                if not p['teammate'] and p['position']['name'] != 'Goalkeeper'
            ]
            lane_defenders = [
                p for p in defenders if p['location'][0] > loc[0]
            ]

            t_sec = ts_to_sec(s['timestamp'])
            dt = None if k == 0 else t_sec - ts_to_sec(slist[k - 1]['timestamp'])
            prev_outcome = slist[k - 1]['shot']['outcome']['name'] if k > 0 else None

            rows.append({
                'possession':       poss_id,
                'shot_order':       k,           # 0 = first shot
                'n_shots_in_poss':  n,
                'timestamp_sec':    t_sec,
                'dt_from_prev':     dt,          # seconds since previous shot
                'outcome':          sh['outcome']['name'],
                'prev_outcome':     prev_outcome,
                # xG / execution
                'xg':               sh['statsbomb_xg'],
                'exec_xg':          sh.get('shot_execution_xg'),
                'uplift':           sh.get('shot_execution_xg_uplift'),
                'gk_pos_supp':      sh.get('gk_positioning_xg_suppression'),
                'gk_stop_supp':     sh.get('gk_shot_stopping_xg_suppression'),
                # geometric / spatial
                'n_defenders_total':    len(defenders),
                'n_lane_defenders':     len(lane_defenders),
                'min_def_dist':         min(dist2d(loc, d['location']) for d in defenders) if defenders else None,
                'unoccluded_angle':     unoccluded_goal_angle(loc, ff),
                'gk_displacement':      gk_lateral_displacement(loc, ff),
                'dist_to_goal':         shot_to_goal_dist(loc),
                'shot_x':               loc[0],
                'shot_y':               loc[1],
                'shot_z':               loc[2] if len(loc) > 2 else 0.0,
                'body_part':            sh.get('body_part', {}).get('name', ''),
                'technique':            sh.get('technique', {}).get('name', ''),
                'one_on_one':           sh.get('one_on_one', False),
                'first_time':           sh.get('first_time', False),
            })
    return rows


# ─────────────────────────────────────────────
# 2. LOAD ALL DATA
# ─────────────────────────────────────────────

all_rows = []
for fpath in FILES:
    with open(fpath) as f:
        events = json.load(f)
    rows = extract_sequence_features(events)
    for r in rows:
        r['file'] = fpath.split('/')[-1]
    all_rows.extend(rows)

# Separate first shots and subsequent shots
first_shots  = [r for r in all_rows if r['shot_order'] == 0]
second_shots = [r for r in all_rows if r['shot_order'] == 1]
third_shots  = [r for r in all_rows if r['shot_order'] == 2]
multi_poss   = [r for r in all_rows if r['n_shots_in_poss'] > 1]

print("=" * 60)
print("DEFENSIVE REORGANIZATION MODEL — SUMMARY STATISTICS")
print("=" * 60)
print(f"Total shots:              {len(all_rows)}")
print(f"Possessions with shots:   {len(set(r['possession'] for r in all_rows))}")
print(f"Multi-shot possessions:   {len(set(r['possession'] for r in multi_poss))}")
print(f"  First shots (k=0):      {len(first_shots)}")
print(f"  Second shots (k=1):     {len(second_shots)}")
print(f"  Third shots  (k=2):     {len(third_shots)}")


# ─────────────────────────────────────────────
# 3. HYPOTHESIS 1: xG CHANGE ACROSS SEQUENCE
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("H1: Does xG change from shot k=0 → k=1 within same possession?")
print("─" * 60)

# Pair first and second shots within same possession
paired = []
poss_to_first = {r['possession']: r for r in first_shots}
for s2 in second_shots:
    s1 = poss_to_first.get(s2['possession'])
    if s1:
        paired.append((s1, s2))

print(f"\n{'Poss':>6} {'xG(k=0)':>10} {'xG(k=1)':>10} {'Δ_xg':>10} "
      f"{'Δt(s)':>8} {'prev_outcome':>14} {'Δ_ang_w':>10} {'Δ_n_def':>9}")
print("-" * 90)
delta_xg = []
for s1, s2 in paired:
    d = s2['xg'] - s1['xg']
    da = (s2['unoccluded_angle'] or 0) - (s1['unoccluded_angle'] or 0)
    dd = s2['n_defenders_total'] - s1['n_defenders_total']
    delta_xg.append(d)
    print(f"{s2['possession']:>6} {s1['xg']:>10.3f} {s2['xg']:>10.3f} {d:>+10.3f} "
          f"{s2['dt_from_prev']:>8.1f} {s2['prev_outcome']:>14} {da:>+10.3f} {dd:>+9}")

if delta_xg:
    print(f"\n  Mean Δ_xg  = {np.mean(delta_xg):+.4f}")
    print(f"  Std  Δ_xg  = {np.std(delta_xg):.4f}")
    print(f"  Positive Δ (xG rose):  {sum(1 for d in delta_xg if d > 0)} / {len(delta_xg)}")
    print(f"  Negative Δ (xG fell):  {sum(1 for d in delta_xg if d < 0)} / {len(delta_xg)}")


# ─────────────────────────────────────────────
# 4. HYPOTHESIS 2: DEFENSIVE CONSOLIDATION OVER TIME
#    Model: n_defenders(k) = D_max * (1 - exp(-lambda * dt))
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("H2: Defensive Consolidation — defender count vs time between shots")
print("─" * 60)
print("\nExponential recovery model: ΔD(Δt) = D_max * (1 - exp(-λ * Δt))")
print(f"\n{'Poss':>6} {'Δt(s)':>8} {'D(k=0)':>8} {'D(k=1)':>8} {'ΔD':>6} {'GK_pos(k=0)':>13} {'GK_pos(k=1)':>13} {'ΔGK_pos':>10}")
print("-" * 85)
dt_vals, delta_d_vals = [], []
gk_supp_delta = []
for s1, s2 in paired:
    dt = s2['dt_from_prev']
    dd = s2['n_defenders_total'] - s1['n_defenders_total']
    gk_delta = None
    if s1['gk_pos_supp'] is not None and s2['gk_pos_supp'] is not None:
        gk_delta = s2['gk_pos_supp'] - s1['gk_pos_supp']
        gk_supp_delta.append(gk_delta)
    delta_d_vals.append(dd)
    dt_vals.append(dt)
    print(f"{s2['possession']:>6} {dt:>8.1f} {s1['n_defenders_total']:>8} {s2['n_defenders_total']:>8} {dd:>+6} "
          f"{s1['gk_pos_supp']:>13.4f} {s2['gk_pos_supp']:>13.4f} {gk_delta:>+10.4f}")

# Fit simple exponential model via log-linearization if enough data
# ΔD ≈ D_max * (1 - exp(-λΔt))  →  log(D_max - ΔD) ≈ log(D_max) - λΔt
print(f"\n  Mean ΔD (defenders gained) = {np.mean(delta_d_vals):+.2f}")
print(f"  Correlation(Δt, ΔD)        = {np.corrcoef(dt_vals, delta_d_vals)[0,1]:+.4f}")
if gk_supp_delta:
    print(f"  Mean Δ(gk_pos_supp)       = {np.mean(gk_supp_delta):+.4f}")
    print(f"  (positive = GK recovers toward centre = more suppression)")


# ─────────────────────────────────────────────
# 5. HYPOTHESIS 3: EXECUTION DEGRADATION UNDER PRESSURE
#    Uplift_k = alpha_0 + alpha_1*(1/dt) + alpha_2*min_def_dist
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("H3: Shot Execution Degradation — does rushing hurt uplift?")
print("─" * 60)
print("\nModel: Uplift = α₀ + α₁·(1/Δt) + α₂·min_def_dist")
print(f"\n{'Poss':>6} {'order':>6} {'Δt(s)':>8} {'1/Δt':>8} {'min_def_d':>10} {'uplift':>9}")
print("-" * 65)

uplift_rows = []
for r in second_shots + third_shots:
    if r['uplift'] is not None and r['dt_from_prev'] is not None and r['min_def_dist'] is not None:
        inv_dt = 1.0 / max(r['dt_from_prev'], 0.1)
        uplift_rows.append((r['possession'], r['shot_order'], r['dt_from_prev'],
                           inv_dt, r['min_def_dist'], r['uplift']))
        print(f"{r['possession']:>6} {r['shot_order']:>6} {r['dt_from_prev']:>8.1f} "
              f"{inv_dt:>8.3f} {r['min_def_dist']:>10.2f} {r['uplift']:>+9.4f}")

if len(uplift_rows) >= 3:
    # OLS: Uplift ~ 1/dt + min_def_dist
    X = np.array([[1.0, row[3], row[4]] for row in uplift_rows])
    y = np.array([row[5] for row in uplift_rows])
    # Normal equations: beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        print(f"\n  OLS estimates (n={len(uplift_rows)}):")
        print(f"    α₀ (intercept)  = {beta[0]:+.4f}")
        print(f"    α₁ (1/Δt)       = {beta[1]:+.4f}  (negative → rushing hurts)")
        print(f"    α₂ (min_def_d)  = {beta[2]:+.4f}  (positive → space helps)")
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
        print(f"    R²              = {r2:.4f}")
        print(f"  Note: n={len(uplift_rows)} — interpret directionally only.")
    except Exception as e:
        print(f"  OLS failed: {e}")


# ─────────────────────────────────────────────
# 6. GK DISPLACEMENT OPPORTUNITY MODEL
#    After a save, GK is displaced → more visible goal for shot k+1
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("H4: GK Displacement creates angular opportunity after save")
print("─" * 60)
print("\nPrediction: saved first shot → larger unoccluded_angle at shot k=1")
print(f"\n{'Poss':>6} {'prev_outcome':>14} {'Δt(s)':>8} {'ang(k=0)':>10} {'ang(k=1)':>10} {'Δ_ang':>8} {'gk_disp(k=1)':>14}")
print("-" * 80)
for s1, s2 in paired:
    a1 = s1['unoccluded_angle'] or 0
    a2 = s2['unoccluded_angle'] or 0
    gk_d2 = s2['gk_displacement']
    print(f"{s2['possession']:>6} {s2['prev_outcome']:>14} {s2['dt_from_prev']:>8.1f} "
          f"{a1:>10.4f} {a2:>10.4f} {a2-a1:>+8.4f} "
          f"{gk_d2 if gk_d2 is not None else 'N/A':>14}")

print("\n  Interpretation:")
print("  - +Δ_ang means goal was MORE visible for the second shot")
print("  - gk_disp measures how far GK was off optimal line at shot k=1")
print("  - After saves with small Δt, GK displacement should be high")


# ─────────────────────────────────────────────
# 7. OPPORTUNITY WINDOW SIMULATION
#    Theoretical: when does xG(k=1) peak as function of Δt?
# ─────────────────────────────────────────────

print("\n" + "─" * 60)
print("THEORETICAL: Opportunity Window — xG(k=1) vs Δt")
print("─" * 60)
print("""
The tension between GK recovery and defender consolidation creates
a theoretical optimal Δt for second-shot opportunity:

  GK displacement: G(t) = G_0 * exp(-λ_gk * t)   [decays fast]
  Defender count:  D(t) = D_max * (1-exp(-λ_d*t))  [grows fast]

  xG_opportunity(t) ∝ G(t) / D(t)
                     = [G_0 * exp(-λ_gk*t)] / [D_max*(1-exp(-λ_d*t))]

Peak opportunity at: d/dt [xG_opp(t)] = 0
  → G_0 * (-λ_gk * exp(-λ_gk*t)) * D(t) = G(t) * D_max * λ_d * exp(-λ_d*t)
  → -λ_gk * (1-exp(-λ_d*t)) = λ_d * exp(-(λ_gk-λ_d)*t)

This has no closed form but can be solved numerically given estimates of
λ_gk (GK recovery rate) and λ_d (defender consolidation rate).
""")

# Illustrative numerical solution
lambda_gk = 0.8   # GK recovers in ~1.25s
lambda_d  = 0.3   # defenders take ~3.3s to consolidate
G_0       = 3.0   # metres of initial GK displacement
D_max     = 8.0   # max defenders in zone

t_vals = np.linspace(0.1, 15, 300)
G_t = G_0 * np.exp(-lambda_gk * t_vals)
D_t = D_max * (1 - np.exp(-lambda_d * t_vals))
opp_t = G_t / (D_t + 1e-3)

t_peak = t_vals[np.argmax(opp_t)]
print(f"  Illustrative parameters: λ_gk={lambda_gk}, λ_d={lambda_d}")
print(f"  G_0={G_0}m, D_max={D_max} defenders")
print(f"  → Theoretical peak opportunity window: Δt ≈ {t_peak:.2f} seconds\n")

print("  Δt(s)   G(t)    D(t)   Opportunity_Index")
print("  " + "-"*45)
for ti in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]:
    idx = np.argmin(np.abs(t_vals - ti))
    print(f"  {ti:>5.1f}   {G_t[idx]:.3f}   {D_t[idx]:.3f}   {opp_t[idx]:.4f}")

print(f"\n  → Maximum opportunity at Δt = {t_peak:.2f}s")
print("  → After ~5s defenders are largely consolidated; GK advantage persists longer")
print("  → Implication: teams should aim to shoot within 1-3s of a save for peak xG")

print("\n" + "=" * 60)
print("END OF MODEL OUTPUT")
print("=" * 60)
