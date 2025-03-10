import pandas as pd


def calculate_rolling_avg_pandas(df, window=5):
    """
    Calculates the rolling average of goals for each team across the last 'window' matches,
    considering both home and away games.

    Args:
    df (pd.DataFrame): DataFrame with columns ['home_team', 'away_team', 'home_last_team_goal', 'away_last_team_goal']
    window (int): The number of last matches to include in the rolling average.

    Returns:
    pd.DataFrame: Original DataFrame with additional rolling average columns.
    """

    home_stats = df[['season', 'stage', 'date', 'home_team', 'home_last_team_goal', 'goal_conversion_rate_home',
                     'home_last_team_shoton']].rename(
        columns={'home_team': 'team', 'home_last_team_goal': 'goals',
                 'goal_conversion_rate_home': 'goal_conversion_rate', 'home_last_team_shoton': 'last_team_shoton'}
    )
    away_stats = df[['season', 'stage', 'date', 'away_team', 'away_last_team_goal', 'goal_conversion_rate_away',
                     'away_last_team_shoton']].rename(
        columns={'away_team': 'team', 'away_last_team_goal': 'goals',
                 'goal_conversion_rate_away': 'goal_conversion_rate', 'away_last_team_shoton': 'last_team_shoton'}
    )

    team_goals = pd.concat([home_stats, away_stats], ignore_index=True)

    team_goals = team_goals.sort_values(by=['team', 'season', 'stage', 'date']).reset_index(drop=True)

    team_goals['rolling_avg_goals'] = team_goals.groupby('team')['goals'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0))
    team_goals['rolling_stability_goal'] = team_goals.groupby('team')['goals'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))

    team_goals['rolling_avg_goals_conversion_rate'] = team_goals.groupby('team')['goal_conversion_rate'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0))
    team_goals['rolling_stability_goals_conversion_rate'] = team_goals.groupby('team')[
        'goal_conversion_rate'].transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))

    team_goals['rolling_avg_shoton'] = team_goals.groupby('team')['last_team_shoton'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().fillna(0))
    team_goals['rolling_stability_shoton'] = team_goals.groupby('team')['last_team_shoton'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))

    team_goals = team_goals.drop_duplicates(subset=['team', 'season', 'stage', 'date'])

    df = df.merge(
        team_goals[['team', 'season', 'stage', 'date', 'rolling_avg_goals', 'rolling_stability_goal',
                    'rolling_avg_goals_conversion_rate', 'rolling_stability_goals_conversion_rate',
                    'rolling_avg_shoton', 'rolling_stability_shoton']],
        left_on=['home_team', 'season', 'stage', 'date'],
        right_on=['team', 'season', 'stage', 'date'],
        how='left'
    ).rename(columns={
        'rolling_avg_goals': 'rolling_avg_goals_home',
        'rolling_stability_goal': 'rolling_stability_goal_home',
        'rolling_avg_goals_conversion_rate': 'rolling_avg_goals_conversion_rate_home',
        'rolling_stability_goals_conversion_rate': 'rolling_stability_goals_conversion_rate_home',
        'rolling_avg_shoton': 'rolling_avg_shoton_home',
        'rolling_stability_shoton': 'rolling_stability_shoton_home',
    }).drop(columns=['team'])

    df = df.merge(
        team_goals[['team', 'season', 'stage', 'date', 'rolling_avg_goals', 'rolling_stability_goal',
                    'rolling_avg_goals_conversion_rate', 'rolling_stability_goals_conversion_rate',
                    'rolling_avg_shoton', 'rolling_stability_shoton']],
        left_on=['away_team', 'season', 'stage', 'date'],
        right_on=['team', 'season', 'stage', 'date'],
        how='left'
    ).rename(columns={
        'rolling_avg_goals': 'rolling_avg_goals_away',
        'rolling_stability_goal': 'rolling_stability_goal_away',
        'rolling_avg_goals_conversion_rate': 'rolling_avg_goals_conversion_rate_away',
        'rolling_stability_goals_conversion_rate': 'rolling_stability_goals_conversion_rate_away',
        'rolling_avg_shoton': 'rolling_avg_shoton_away',
        'rolling_stability_shoton': 'rolling_stability_shoton_away',
    }).drop(columns=['team'])

    return df
