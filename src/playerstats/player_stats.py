import numpy as np
import pandas as pd

def get_player_overall_rating_(player_id, match_date, df_player_attr):
    filtered = df_player_attr[
        (df_player_attr['player_api_id'] == player_id) &
        (df_player_attr['date'] <= match_date)
    ]

    if filtered.empty:
        return np.nan

    filtered_sorted = filtered.sort_values(by='date', ascending=False)
    latest_rating = filtered_sorted.iloc[0]['overall_rating']

    return latest_rating

def get_player_id_for_team_(
    row,
    player,
    team_type,
    df_matches,
    n_previous=10
):
    player_id = row[player]

    if not np.isnan(player_id):
        return player_id

    team_id = row[f"{team_type}_team"]
    same_team_matches = df_matches[df_matches[f"{team_type}_team"] == team_id]

    current_match_date = row["date"]
    same_team_matches = same_team_matches[same_team_matches["date"] < current_match_date]
    same_team_matches = same_team_matches.sort_values(by="date", ascending=False)
    same_team_matches = same_team_matches.head(n_previous)
    col_values = same_team_matches[player].dropna()

    if col_values.empty:
        return np.nan

    fallback_id = col_values.value_counts().idxmax()
    return fallback_id

def calculate_player_stat_(match_row, df_matches, df_player_attr, players):
    player_stats_dict = {}
    match_date = match_row['date']
    player_stats_dict['match_api_id'] = match_row['match_api_id']

    for player in players:
        team_type = 'home' if 'home' in player else 'away'

        player_id = get_player_id_for_team_(
            row=match_row,
            player=player,
            team_type=team_type,
            df_matches=df_matches,
            n_previous=10
        )

        rating = get_player_overall_rating_(
            player_id=player_id,
            match_date=match_date,
            df_player_attr=df_player_attr
        )

        rating_col_name = f"player_rating_{player}"
        player_stats_dict[rating_col_name] = rating if not np.isnan(rating) else np.nan

    return player_stats_dict
