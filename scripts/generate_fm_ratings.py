"""
Generate Football Manager style player ratings for the Eredivisie Vrouwen
2025-2026 season from raw StatsBomb season stats.

Input:  "Team EredivisieW 2025-2026 (1).xlsx"  (sheet: "Player Stats")
Output: "Eredivisie Women 2025-2026 FM Ratings.xlsx"

Method
------
StatsBomb doesn't publish FM-style attributes (Finishing, Tackling, ...), so
each attribute below is built from a small basket of related StatsBomb
per-90 metrics. For every attribute, players are ranked by percentile
*within their position group* (goalkeepers are ranked only against other
goalkeepers) and that percentile is mapped onto FM's 1-20 attribute scale.
Comparing within position group mirrors how FM/scouts judge a player
relative to their role, rather than against unrelated positions.

Players with fewer than MIN_90S 90-minutes-played are dropped as too small
a sample to rate reliably.
"""

import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

SOURCE_FILE = "Team EredivisieW 2025-2026 (1).xlsx"
OUTPUT_FILE = "Eredivisie Women 2025-2026 FM Ratings.xlsx"
MIN_90S = 3.0

POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Centre Back": "CB",
    "Left Centre Back": "CB",
    "Right Centre Back": "CB",
    "Left Back": "FB",
    "Right Back": "FB",
    "Left Wing Back": "FB",
    "Right Wing Back": "FB",
    "Left Defensive Midfielder": "DM",
    "Right Defensive Midfielder": "DM",
    "Centre Defensive Midfielder": "DM",
    "Left Centre Midfielder": "CM",
    "Right Centre Midfielder": "CM",
    "Left Midfielder": "CM",
    "Right Midfielder": "CM",
    "Centre Attacking Midfielder": "AM",
    "Left Attacking Midfielder": "AM",
    "Right Attacking Midfielder": "AM",
    "Left Wing": "WNG",
    "Right Wing": "WNG",
    "Centre Forward": "FW",
    "Left Centre Forward": "FW",
    "Right Centre Forward": "FW",
}

GROUP_LABELS = {
    "GK": "Goalkeeper",
    "CB": "Centre Back",
    "FB": "Full Back-Wing Back",
    "DM": "Defensive Midfielder",
    "CM": "Central Midfielder",
    "AM": "Attacking Midfielder",
    "WNG": "Winger",
    "FW": "Forward",
}

# attribute -> list of (column, invert). invert=True means lower raw value = better.
OUTFIELD_ATTRIBUTES = {
    "Finishing": [
        ("player_season_conversion_ratio", False),
        ("player_season_shot_on_target_ratio", False),
        ("player_season_over_under_performance_90", False),
    ],
    "Heading": [
        ("player_season_aerial_wins_90", False),
        ("player_season_aerial_ratio", False),
    ],
    "Passing": [
        ("player_season_passing_ratio", False),
        ("player_season_pressured_passing_ratio", False),
    ],
    "Crossing": [
        ("player_season_crosses_90", False),
        ("player_season_crossing_ratio", False),
        ("player_season_box_cross_ratio", False),
    ],
    "Dribbling": [
        ("player_season_dribble_ratio", False),
        ("player_season_total_dribbles_90", False),
        ("player_season_obv_dribble_carry_90", False),
    ],
    "Tackling": [
        ("player_season_padj_tackles_90", False),
        ("player_season_challenge_ratio", False),
    ],
    "Marking": [
        ("player_season_padj_interceptions_90", False),
        ("player_season_defensive_actions_above_expectation_90", False),
    ],
    "Technique": [
        ("player_season_obv_90", False),
        ("player_season_positive_outcome_90", False),
    ],
    "Vision": [
        ("player_season_xa_90", False),
        ("player_season_key_passes_90", False),
        ("player_season_through_balls_90", False),
    ],
    "Off The Ball": [
        ("player_season_npga_90", False),
        ("player_season_touches_inside_box_90", False),
    ],
    "Work Rate": [
        ("player_season_padj_pressures_90", False),
        ("player_season_fhalf_pressures_90", False),
    ],
    "Composure": [
        ("player_season_turnovers_90", True),
        ("player_season_dispossessions_90", True),
    ],
    "Teamwork": [
        ("player_season_op_xgbuildup_90", False),
        ("player_season_obv_pass_90", False),
    ],
}

GK_ATTRIBUTES = {
    "Shot Stopping": [
        ("player_season_save_ratio", False),
        ("player_season_gsaa_90", False),
        ("player_season_xs_ratio", False),
    ],
    "Command Of Area": [
        ("player_season_clcaa", False),
        ("player_season_average_x_defensive_action", False),
    ],
    "Distribution": [
        ("player_season_passing_ratio", False),
        ("player_season_long_ball_ratio", False),
    ],
    "One On Ones": [
        ("player_season_gsaa_ratio", False),
        ("player_season_np_optimal_gk_dlength", False),
    ],
    "Composure": [
        ("player_season_errors_90", True),
    ],
}

# attributes that most define "overall" quality per position group
OVERALL_WEIGHTS = {
    "GK": ["Shot Stopping", "Shot Stopping", "Command Of Area", "Distribution", "One On Ones", "Composure"],
    "CB": ["Tackling", "Marking", "Heading", "Passing", "Composure"],
    "FB": ["Tackling", "Marking", "Crossing", "Passing", "Work Rate", "Dribbling"],
    "DM": ["Tackling", "Marking", "Passing", "Work Rate", "Composure", "Technique"],
    "CM": ["Passing", "Vision", "Technique", "Work Rate", "Tackling", "Composure"],
    "AM": ["Vision", "Technique", "Dribbling", "Off The Ball", "Finishing", "Passing"],
    "WNG": ["Dribbling", "Crossing", "Vision", "Off The Ball", "Finishing", "Technique"],
    "FW": ["Finishing", "Off The Ball", "Heading", "Technique", "Dribbling", "Composure"],
}


def percentile_to_fm_scale(pct: pd.Series) -> pd.Series:
    """Map a 0-1 percentile to FM's 1-20 attribute scale."""
    return (1 + pct * 19).round().clip(1, 20).astype(int)


def rate_group(df: pd.DataFrame, attributes: dict) -> pd.DataFrame:
    """Compute 1-20 attribute ratings for every player in df (a single position group)."""
    out = pd.DataFrame(index=df.index)
    for attr, components in attributes.items():
        pct_sum = pd.Series(0.0, index=df.index)
        for col, invert in components:
            values = df[col].fillna(df[col].median())
            pct = values.rank(pct=True, method="average")
            if invert:
                pct = 1 - pct
            pct_sum += pct
        avg_pct = pct_sum / len(components)
        out[attr] = percentile_to_fm_scale(avg_pct)
    return out


def build_overall(ratings: pd.DataFrame, group: str) -> pd.Series:
    attrs = OVERALL_WEIGHTS[group]
    return ratings[attrs].mean(axis=1)


def star_rating(overall_pct: pd.Series) -> pd.Series:
    stars = 1 + overall_pct * 4  # 1.0 - 5.0
    return (stars * 2).round() / 2  # nearest half star


def age_from_birthdate(birth_date, as_of="2026-07-02"):
    if pd.isna(birth_date):
        return None
    bd = pd.to_datetime(birth_date)
    ref = pd.to_datetime(as_of)
    return ref.year - bd.year - ((ref.month, ref.day) < (bd.month, bd.day))


def main():
    df = pd.read_excel(SOURCE_FILE, sheet_name="Player Stats")
    df = df[df["player_season_90s_played"] >= MIN_90S].copy()
    df["position_group"] = df["primary_position"].map(POSITION_GROUPS)
    df = df[df["position_group"].notna()].copy()

    all_ratings = []
    for group, group_df in df.groupby("position_group"):
        attrs = GK_ATTRIBUTES if group == "GK" else OUTFIELD_ATTRIBUTES
        ratings = rate_group(group_df, attrs)
        overall = build_overall(ratings, group)
        overall_pct = overall.rank(pct=True, method="average")
        ratings.insert(0, "Overall", overall.round().astype(int))
        ratings["Star Rating"] = star_rating(overall_pct)
        combined = group_df[[
            "player_name", "team_name", "primary_position", "position_group",
            "birth_date", "player_season_90s_played", "player_season_minutes",
        ]].join(ratings)
        all_ratings.append(combined)

    result = pd.concat(all_ratings).sort_values(
        ["position_group", "Overall"], ascending=[True, False]
    )
    result["Age"] = result["birth_date"].apply(age_from_birthdate)
    result["Position Group"] = result["position_group"].map(GROUP_LABELS)

    front_cols = [
        "player_name", "team_name", "primary_position", "Position Group",
        "Age", "player_season_90s_played", "Overall", "Star Rating",
    ]
    attr_cols = [c for c in result.columns if c not in front_cols + [
        "position_group", "birth_date", "player_season_minutes",
    ]]
    result = result[front_cols + attr_cols]
    result = result.rename(columns={
        "player_name": "Player",
        "team_name": "Team",
        "primary_position": "Position",
        "player_season_90s_played": "90s Played",
    })
    result["90s Played"] = result["90s Played"].round(1)

    write_workbook(result)
    print(f"Rated {len(result)} players -> {OUTPUT_FILE}")


def write_workbook(result: pd.DataFrame):
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for group_code, label in GROUP_LABELS.items():
            sub = result[result["Position Group"] == label].drop(columns=["Position Group"])
            if sub.empty:
                continue
            sub = sub.dropna(axis=1, how="all")
            sub.to_excel(writer, sheet_name=label[:31], index=False)

        top20 = result.sort_values(["Overall", "Star Rating"], ascending=False).head(20)
        top20.to_excel(writer, sheet_name="Top 20 Overall", index=False)

        best_xi_rows = []
        for group_code, label in GROUP_LABELS.items():
            if group_code == "GK":
                continue
            sub = result[result["Position Group"] == label]
            if not sub.empty:
                best_xi_rows.append(sub.sort_values("Overall", ascending=False).iloc[0])
        gk = result[result["Position Group"] == "Goalkeeper"].sort_values("Overall", ascending=False)
        if not gk.empty:
            best_xi_rows.insert(0, gk.iloc[0])
        best_xi = pd.DataFrame(best_xi_rows)
        best_xi.to_excel(writer, sheet_name="Best XI (by group)", index=False)

    style_workbook(OUTPUT_FILE)


def style_workbook(path: str):
    import openpyxl
    wb = openpyxl.load_workbook(path)
    header_fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col_cells in ws.columns:
            length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            col_letter = get_column_letter(col_cells[0].column)
            ws.column_dimensions[col_letter].width = min(max(length + 2, 10), 28)
        if ws.max_row > 1:
            overall_col = None
            for cell in ws[1]:
                if cell.value == "Overall":
                    overall_col = cell.column_letter
                    break
            if overall_col:
                rng = f"{overall_col}2:{overall_col}{ws.max_row}"
                rule = ColorScaleRule(
                    start_type="min", start_color="F8696B",
                    mid_type="percentile", mid_value=50, mid_color="FFEB84",
                    end_type="max", end_color="63BE7B",
                )
                ws.conditional_formatting.add(rng, rule)
    wb.save(path)


if __name__ == "__main__":
    main()
