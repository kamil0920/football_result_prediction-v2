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

    # Reshape the data into long format so each team has a single column for goals
    home_stats = df[['season', 'stage', 'date', 'home_team', 'home_last_team_goal', 'goal_conversion_rate_home']].rename(
        columns={'home_team': 'team', 'home_last_team_goal': 'goals', 'goal_conversion_rate_home': 'goal_conversion_rate', }
    )
    away_stats = df[['season', 'stage', 'date','away_team', 'away_last_team_goal', 'goal_conversion_rate_away']].rename(
        columns={'away_team': 'team', 'away_last_team_goal': 'goals', 'goal_conversion_rate_away': 'goal_conversion_rate', }
    )

    # Combine home and away records into a single dataframe
    team_goals = pd.concat([home_stats, away_stats], ignore_index=True)

    # Sort by index to maintain match chronology
    team_goals = team_goals.sort_values(by=['team', 'season', 'stage', 'date']).reset_index(drop=True)

    # Calculate rolling average of goals for each team
    team_goals['rolling_avg_goals'] = team_goals.groupby('team')['goals'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    team_goals['rolling_goal_stability'] = team_goals.groupby('team')['goal_conversion_rate'].transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))


    # Remove duplicate rows before merging (fixes memory explosion issue)
    team_goals = team_goals.drop_duplicates(subset=['team', 'season', 'stage', 'date'])

    # Merge rolling average back to original DataFrame, keeping only necessary columns
    df = df.merge(
        team_goals[['team', 'season', 'stage', 'date', 'rolling_avg_goals', 'rolling_goal_stability']],
        left_on=['home_team', 'season', 'stage', 'date'],
        right_on=['team', 'season', 'stage', 'date'],
        how='left'
    ).rename(columns={'rolling_avg_goals': 'rolling_avg_goals_home', 'rolling_goal_stability': 'rolling_goal_stability_home'}).drop(columns=['team'])

    df = df.merge(
        team_goals[['team', 'season', 'stage', 'date', 'rolling_avg_goals', 'rolling_goal_stability']],
        left_on=['away_team', 'season', 'stage', 'date'],
        right_on=['team', 'season', 'stage', 'date'],
        how='left'
    ).rename(columns={'rolling_avg_goals': 'rolling_avg_goals_away', 'rolling_goal_stability': 'rolling_goal_stability_away'}).drop(columns=['team'])

    return df