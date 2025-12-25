# cricstars_app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="CricStars - CWC 2023 Analytics",
    layout="wide"
)

sns.set_style("whitegrid")


@st.cache_data
def load_data(path: str):
    """
    Load ODI World Cup 2023 dataset from kaggle CSV.
    """
    df = pd.read_csv(path)
    return df


def safe_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def get_batting_view(df):
    """
    Try to build a batting table from the csv.
    Edit this function if your column names are different.
    """
    bat_cols = ["batting_team", "team", "Batting team"]
    player_cols = ["batter", "batsman", "player", "Player"]
    runs_cols = ["runs", "batsman_runs", "total_runs", "Runs"]
    balls_cols = ["balls", "ball_faced", "balls_faced", "Balls"]
    fours_cols = ["4s", "fours", "Fours"]
    sixes_cols = ["6s", "sixes", "Sixes"]
    sr_cols = ["strike_rate", "sr", "SR", "Strike Rate"]

    batting_team_col = safe_col(df, bat_cols)
    player_col = safe_col(df, player_cols)
    runs_col = safe_col(df, runs_cols)
    balls_col = safe_col(df, balls_cols)
    fours_col = safe_col(df, fours_cols)
    sixes_col = safe_col(df, sixes_cols)
    sr_col = safe_col(df, sr_cols)

    bat = df.copy()
    if batting_team_col:
        bat = bat[bat[batting_team_col].notna()]
    if player_col:
        bat = bat[bat[player_col].notna()]
    if runs_col:
        bat = bat[bat[runs_col].notna()]

    if sr_col is None and runs_col and balls_col:
        bat["calc_strike_rate"] = np.where(
            bat[balls_col] > 0,
            bat[runs_col] * 100 / bat[balls_col],
            0
        )
        sr_col = "calc_strike_rate"

    return bat, {
        "team": batting_team_col,
        "player": player_col,
        "runs": runs_col,
        "balls": balls_col,
        "fours": fours_col,
        "sixes": sixes_col,
        "sr": sr_col
    }


def get_bowling_view(df):
    """
    Try to build a bowling table from the csv.
    Edit this function if your column names are different.
    """
    bowl_team_col = safe_col(df, ["bowling_team", "team_bowling", "Bowling team"])
    bowler_col = safe_col(df, ["bowler", "player", "Player"])
    overs_col = safe_col(df, ["overs", "Overs"])
    runs_conc_col = safe_col(df, ["runs_conceded", "Runs Conceded", "runs", "Runs"])
    wickets_col = safe_col(df, ["wickets", "Wickets"])
    econ_col = safe_col(df, ["economy", "Econ", "ECO"])

    bowl = df.copy()
    if bowl_team_col:
        bowl = bowl[bowl[bowl_team_col].notna()]
    if bowler_col:
        bowl = bowl[bowl[bowler_col].notna()]

    if econ_col is None and runs_conc_col and overs_col:
        bowl["calc_econ"] = np.where(
            bowl[overs_col] > 0,
            bowl[runs_conc_col] / bowl[overs_col],
            np.nan
        )
        econ_col = "calc_econ"

    return bowl, {
        "team": bowl_team_col,
        "player": bowler_col,
        "overs": overs_col,
        "runs_conc": runs_conc_col,
        "wickets": wickets_col,
        "econ": econ_col
    }


def top_n(df, group_col, metric_col, n=10, ascending=False):
    if group_col is None or metric_col is None:
        return pd.DataFrame()
    agg = df.groupby(group_col)[metric_col].sum().reset_index()
    return agg.sort_values(metric_col, ascending=ascending).head(n)


# sidebar
st.sidebar.title("CricStars - Filters")

data_path_default = "odi_wc_2023.csv"
data_path = st.sidebar.text_input(
    "Path to World Cup 2023 dataset CSV",
    value=data_path_default,
    help="Put your ODI World Cup 2023 CSV here (for example from Kaggle)."
)

if not os.path.exists(data_path):
    st.sidebar.warning(
        f"CSV not found at: {data_path}\n"
        f"Please put your dataset csv with name 'odi_wc_2023.csv' in this folder."
    )

    st.title("CricStars - CWC 2023 Analytics")
    st.write("Please add the dataset first and then refresh the app.")
    st.stop()

df_raw = load_data(data_path)

tabs = st.tabs(["Overview", "Batting", "Bowling", "Fielding", "Player Explorer"])


# ========== OVERVIEW TAB ==========
with tabs[0]:
    st.title("CricStars - Cricket World Cup 2023")

    st.write(
        "This is a simple cricket analytics dashboard for ICC Cricket World Cup 2023. "
        "It shows basic batting, bowling and fielding stats using Python and Streamlit."
    )

    total_rows = len(df_raw)
    teams_col = safe_col(df_raw, ["team", "Team", "batting_team", "bowling_team"])
    if teams_col:
        teams = sorted(df_raw[teams_col].dropna().unique())
    else:
        teams = []

    matches_col = safe_col(df_raw, ["match_id", "Match_id", "match", "Match Number"])
    num_matches = df_raw[matches_col].nunique() if matches_col else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total rows in dataset", total_rows)
    with col2:
        st.metric("Unique teams", len(teams) if teams else 0)
    with col3:
        st.metric("Approx matches", int(num_matches) if num_matches is not None else 0)

    runs_col_global = safe_col(df_raw, ["runs", "total_runs", "Score A", "Score B", "Runs"])
    if runs_col_global:
        st.subheader("Runs distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_raw[runs_col_global].dropna(), bins=30, kde=True, ax=ax)
        ax.set_xlabel("Runs")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)


# ========== BATTING TAB ==========
with tabs[1]:
    st.header("Batting analytics")

    bat_df, bat_cols = get_batting_view(df_raw)

    if bat_cols["team"] is None or bat_cols["player"] is None or bat_cols["runs"] is None:
        st.warning(
            "Batting columns not detected correctly. "
            "Please edit get_batting_view() according to your csv column names."
        )
    else:
        teams = sorted(bat_df[bat_cols["team"]].dropna().unique())
        selected_teams = st.multiselect(
            "Select team(s)",
            options=teams,
            default=teams
        )

        filtered = bat_df[bat_df[bat_cols["team"]].isin(selected_teams)]

        st.subheader("Top run scorers")
        top_runs = top_n(filtered, bat_cols["player"], bat_cols["runs"], n=10, ascending=False)
        st.dataframe(
            top_runs.rename(columns={bat_cols["player"]: "Player", bat_cols["runs"]: "Total Runs"})
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=top_runs,
            x="Total Runs",
            y="Player",
            ax=ax,
            palette="viridis"
        )
        ax.set_xlabel("Runs")
        ax.set_ylabel("Player")
        st.pyplot(fig)

        if bat_cols["sr"] is not None:
            st.subheader("Strike rate vs Runs")
            scatter_df = filtered[
                [bat_cols["player"], bat_cols["runs"], bat_cols["sr"]]
            ].dropna()
            scatter_df = scatter_df.groupby(bat_cols["player"], as_index=False).agg(
                {bat_cols["runs"]: "sum", bat_cols["sr"]: "mean"}
            )
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.scatter(
                scatter_df[bat_cols["runs"]],
                scatter_df[bat_cols["sr"]],
                alpha=0.7
            )
            ax2.set_xlabel("Total runs")
            ax2.set_ylabel("Average strike rate")
            st.pyplot(fig2)

        st.subheader("Team-wise average runs per innings")
        team_avg = filtered.groupby(bat_cols["team"])[bat_cols["runs"]].mean().reset_index()
        team_avg.columns = ["Team", "Avg Runs"]
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=team_avg, x="Team", y="Avg Runs", ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig3)


# ========== BOWLING TAB ==========
with tabs[2]:
    st.header("Bowling analytics")

    bowl_df, bowl_cols = get_bowling_view(df_raw)

    if bowl_cols["team"] is None or bowl_cols["player"] is None or bowl_cols["wickets"] is None:
        st.warning(
            "Bowling columns not detected correctly. "
            "Please edit get_bowling_view() according to your csv column names."
        )
    else:
        bowl_teams = sorted(bowl_df[bowl_cols["team"]].dropna().unique())
        selected_bowl_teams = st.multiselect(
            "Select team(s)",
            options=bowl_teams,
            default=bowl_teams
        )

        bowl_filtered = bowl_df[bowl_df[bowl_cols["team"]].isin(selected_bowl_teams)]

        st.subheader("Top wicket takers")
        top_wkts = top_n(bowl_filtered, bowl_cols["player"], bowl_cols["wickets"], n=10, ascending=False)
        st.dataframe(
            top_wkts.rename(columns={bowl_cols["player"]: "Player", bowl_cols["wickets"]: "Total Wickets"})
        )

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=top_wkts,
            x="Total Wickets",
            y="Player",
            palette="magma",
            ax=ax4
        )
        ax4.set_xlabel("Wickets")
        ax4.set_ylabel("Player")
        st.pyplot(fig4)

        if bowl_cols["econ"] is not None:
            st.subheader("Economy vs Wickets")
            econ_df = bowl_filtered[
                [bowl_cols["player"], bowl_cols["wickets"], bowl_cols["econ"]]
            ].dropna()
            econ_df = econ_df.groupby(bowl_cols["player"], as_index=False).agg(
                {bowl_cols["wickets"]: "sum", bowl_cols["econ"]: "mean"}
            )
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.scatter(
                econ_df[bowl_cols["econ"]],
                econ_df[bowl_cols["wickets"]],
                alpha=0.7
            )
            ax5.set_xlabel("Average economy")
            ax5.set_ylabel("Total wickets")
            st.pyplot(fig5)

        st.subheader("Team-wise average wickets per innings")
        team_wkts = bowl_filtered.groupby(bowl_cols["team"])[bowl_cols["wickets"]].mean().reset_index()
        team_wkts.columns = ["Team", "Avg Wickets"]
        fig6, ax6 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=team_wkts, x="Team", y="Avg Wickets", ax=ax6)
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig6)


# ========== FIELDING TAB ==========
with tabs[3]:
    st.header("Fielding analytics")

    fielder_col = safe_col(df_raw, ["fielder", "player", "Player", "Fielder"])
    catches_col = safe_col(df_raw, ["catches", "Catches", "Ct"])
    stumpings_col = safe_col(df_raw, ["stumpings", "St", "Stumpings"])

    if fielder_col is None or (catches_col is None and stumpings_col is None):
        st.info(
            "Fielding data (catches, stumpings) not found clearly. "
            "If your csv has these columns, update the names in this tab."
        )
    else:
        fld = df_raw.copy()
        fld = fld[fld[fielder_col].notna()]

        st.subheader("Top fielders (dismissals)")

        fld["total_dismissals"] = 0
        if catches_col:
            fld["total_dismissals"] += fld[catches_col].fillna(0)
        if stumpings_col:
            fld["total_dismissals"] += fld[stumpings_col].fillna(0)

        top_fld = top_n(fld, fielder_col, "total_dismissals", n=10, ascending=False)
        top_fld = top_fld.rename(columns={fielder_col: "Player", "total_dismissals": "Dismissals"})
        st.dataframe(top_fld)

        fig7, ax7 = plt.subplots(figsize=(10, 4))
        sns.barplot(data=top_fld, x="Dismissals", y="Player", palette="crest", ax=ax7)
        ax7.set_xlabel("Total dismissals")
        ax7.set_ylabel("Player")
        st.pyplot(fig7)


# ========== PLAYER EXPLORER TAB ==========
with tabs[4]:
    st.header("Player performance explorer")

    bat_df, bat_cols = get_batting_view(df_raw)
    bowl_df, bowl_cols = get_bowling_view(df_raw)

    all_players = set()
    if bat_cols["player"]:
        all_players.update(bat_df[bat_cols["player"]].dropna().unique())
    if bowl_cols["player"]:
        all_players.update(bowl_df[bowl_cols["player"]].dropna().unique())
    all_players = sorted(list(all_players))

    selected_player = st.selectbox("Select player", options=all_players)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Batting summary")
        if bat_cols["player"]:
            pbat = bat_df[bat_df[bat_cols["player"]] == selected_player]
            if not pbat.empty:
                total_runs = pbat[bat_cols["runs"]].sum() if bat_cols["runs"] else None
                inns = pbat.shape[0]
                avg_sr = pbat[bat_cols["sr"]].mean() if bat_cols["sr"] else None

                st.metric("Total runs", int(total_runs) if total_runs is not None else "-")
                st.metric("Innings", int(inns))
                st.metric("Average strike rate", round(avg_sr, 2) if avg_sr is not None else "-")

                if bat_cols["runs"]:
                    st.caption("Runs per innings")
                    fig8, ax8 = plt.subplots(figsize=(6, 3))
                    ax8.plot(range(1, len(pbat) + 1), pbat[bat_cols["runs"]].values, marker="o")
                    ax8.set_xlabel("Innings index")
                    ax8.set_ylabel("Runs")
                    st.pyplot(fig8)
            else:
                st.info("No batting data for this player.")
        else:
            st.info("Batting columns not detected.")

    with col_right:
        st.subheader("Bowling summary")
        if bowl_cols["player"]:
            pbowl = bowl_df[bowl_df[bowl_cols["player"]] == selected_player]
            if not pbowl.empty:
                total_wkts = pbowl[bowl_cols["wickets"]].sum() if bowl_cols["wickets"] else None
                inns_bowl = pbowl.shape[0]
                avg_econ = pbowl[bowl_cols["econ"]].mean() if bowl_cols["econ"] else None

                st.metric("Total wickets", int(total_wkts) if total_wkts is not None else "-")
                st.metric("Bowling innings", int(inns_bowl))
                st.metric("Average economy", round(avg_econ, 2) if avg_econ is not None else "-")

                if bowl_cols["wickets"]:
                    st.caption("Wickets per spell")
                    fig9, ax9 = plt.subplots(figsize=(6, 3))
                    ax9.bar(range(1, len(pbowl) + 1), pbowl[bowl_cols["wickets"]].values)
                    ax9.set_xlabel("Spell index")
                    ax9.set_ylabel("Wickets")
                    st.pyplot(fig9)
            else:
                st.info("No bowling data for this player.")
        else:
            st.info("Bowling columns not detected.")

    st.write(
        "If the player stats do not look correct, please check your csv column names "
        "and update the helper functions at the top of the file."
    )
