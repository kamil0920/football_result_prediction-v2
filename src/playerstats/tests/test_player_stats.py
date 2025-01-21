import pytest
import numpy as np
import pandas as pd
from src.playerstats.player_stats import (
    get_player_overall_rating_,
    get_player_id_for_team_,
    get_player_stat
)


@pytest.fixture
def df_player_attr():
    """
    A small fixture DataFrame to simulate player attributes over time.
    """
    data = {
        'player_api_id': [1001, 1001, 1002, 1003, 1001],
        'date': [
            '2010-01-01',
            '2010-06-01',
            '2010-03-01',
            '2010-01-15',
            '2010-07-01'
        ],
        'overall_rating': [60, 65, 70, 75, 68],
        'acceleration': [60, 65, 70, 75, 68],
        'strength': [60, 65, 70, 75, 68],
        'aggression': [60, 65, 70, 75, 68],
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


@pytest.fixture
def df_matches():
    """
    A small fixture DataFrame to simulate match details.
    We'll have columns for home_team, away_team, date, home_player_1, etc.
    """
    data = {
        'match_api_id': [2001, 2002, 2003],
        'date': [
            '2010-06-05',
            '2010-07-10',
            '2010-07-11'
        ],
        'home_team': [3001, 3001, 3002],
        'away_team': [4001, 4002, 4002],
        'home_player_1': [1001, np.nan, 1003],
        'away_player_1': [1002, 1002, np.nan],
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


def test_get_player_overall_rating_normal_case(df_player_attr):
    player_id = 1001
    match_date = pd.to_datetime("2010-06-15")
    rating, acceleration, strength, aggression = get_player_overall_rating_(
        player_id, match_date, df_player_attr, n_previous=10
    )

    assert rating == 62.5
    assert acceleration == 62.5
    assert strength == 62.5
    assert aggression == 62.5


def test_get_player_overall_rating_no_data(df_player_attr):
    """
    If no matching player_api_id or no date <= match_date, expect np.nan.
    """
    player_id = 9999
    match_date = pd.to_datetime("2010-06-15")
    rating, acceleration, strength, aggression = get_player_overall_rating_(player_id, match_date, df_player_attr)
    assert np.isnan(rating), "Rating should be NaN for unknown player."


def test_get_player_id_for_team_with_existing_id(df_matches):
    """
    If the row already has a valid (non-NaN) player_id, we just return it.
    """
    row = df_matches.iloc[0]
    found_id = get_player_id_for_team_(
        row=row,
        player='home_player_1',
        team_type='home',
        df_matches=df_matches,
        n_previous=10
    )
    assert found_id == 1001, "Should return the existing player ID if it's not NaN."


def test_get_player_id_for_team_with_fallback(df_matches):
    """
    If the row's player is NaN, it should look up the most recent
    N matches for the same team and pick the mode.
    """
    row = df_matches.iloc[1]
    found_id = get_player_id_for_team_(
        row=row,
        player='home_player_1',
        team_type='home',
        df_matches=df_matches,
        n_previous=10
    )
    assert found_id == 1001, "Should fallback to the most frequent ID among recent matches."


def test_get_player_id_for_team_no_history(df_matches):
    """
    If there's no previous match for that team, we expect NaN.
    """
    new_row = {
        'match_api_id': 9999,
        'date': pd.to_datetime('2010-01-01'),
        'home_team': 9998,
        'home_player_1': np.nan
    }
    row_series = pd.Series(new_row)

    found_id = get_player_id_for_team_(
        row=row_series,
        player='home_player_1',
        team_type='home',
        df_matches=df_matches,
        n_previous=10
    )
    assert np.isnan(found_id), "Should return NaN if no history is found for that team."


def test_calculate_player_stat_integration(df_matches, df_player_attr):
    row = df_matches.iloc[0]
    players = ['home_player_1', 'away_player_1']

    result_dict = get_player_stat(
        match_row=row,
        df_matches=df_matches,
        df_player_attr=df_player_attr,
        players=players
    )

    assert 'match_api_id' in result_dict
    assert result_dict['match_api_id'] == 2001

    assert result_dict['player_rating_home_player_1'] == 62.5

    assert result_dict['player_rating_away_player_1'] == 70
