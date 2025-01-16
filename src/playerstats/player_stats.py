import numpy as np
import pandas as pd

def get_player_overall_rating_(player_id, match_date, df_player_attr,n_previous=10):
    filtered = df_player_attr[
        (df_player_attr['player_api_id'] == player_id) &
        (df_player_attr['date'] <= match_date)
    ]

    if filtered.empty:
        return np.nan, np.nan, np.nan, np.nan

    filtered_sorted = filtered.sort_values(by='date', ascending=False)

    filtered_subset = filtered_sorted.head(n_previous)

    # Compute the means
    latest_rating = filtered_subset['overall_rating'].mean()
    acceleration_rating = filtered_subset['acceleration'].mean()
    strength_rating = filtered_subset['strength'].mean()
    aggression_rating = filtered_subset['aggression'].mean()

    return latest_rating, acceleration_rating, strength_rating, aggression_rating

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


def calculate_player_stat(match_row, df_matches, df_player_attr, players):
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

        overall_rating, acceleration_rating, strength_rating, aggression_rating  = get_player_overall_rating_(
            player_id=player_id,
            match_date=match_date,
            df_player_attr=df_player_attr
        )

        rating_col_name = f"rating_{player}"
        acceleration_rating_col_name = f"acceleration_rating_{player}"
        strength_rating_col_name = f"strength_rating_{player}"
        aggression_rating_col_name = f"aggression_rating_{player}"

        player_stats_dict[rating_col_name] = overall_rating if not np.isnan(overall_rating) else np.nan
        player_stats_dict[acceleration_rating_col_name] = overall_rating if not np.isnan(acceleration_rating) else np.nan
        player_stats_dict[strength_rating_col_name] = overall_rating if not np.isnan(strength_rating) else np.nan
        player_stats_dict[aggression_rating_col_name] = overall_rating if not np.isnan(aggression_rating) else np.nan

    return player_stats_dict
