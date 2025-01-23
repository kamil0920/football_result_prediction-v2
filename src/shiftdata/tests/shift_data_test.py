import pytest
import pandas as pd
import numpy as np
from src.shiftdata.shift_data import ShiftDataPreprocessor

@pytest.fixture
def sample_data():
    """Provides a small sample DataFrame for testing."""
    data = {
        'match_api_id': [1, 2, 3, 4, 5, 6],
        'season': ['2015/2016'] * 6,
        'stage': [1, 1, 2, 2, 3, 3],
        'date': pd.date_range('2015-08-01', periods=6, freq='D'),
        'home_team': ['TeamA', 'TeamB', 'TeamA', 'TeamB', 'TeamA', 'TeamB'],
        'away_team': ['TeamB', 'TeamA', 'TeamB', 'TeamA', 'TeamB', 'TeamA'],
        'home_team_goal': [2, 3, 1, 0, 2, 2],
        'away_team_goal': [1, 2, 2, 3, 0, 2],
        'result_match': ['H', 'H', 'A', 'A', 'H', 'H'],
        'home_shoton': [10, 12, 8, 5, 15, 14],
        'away_shoton': [9, 11, 10, 7, 5, 12],
        'home_possession': [55, 60, 45, 40, 65, 62],
        'away_possession': [45, 40, 55, 60, 35, 38]
    }
    return pd.DataFrame(data)

def test_select_and_rename_columns_home(sample_data):
    """Test that columns for the home team are selected and renamed correctly."""
    preprocessor = ShiftDataPreprocessor(sample_data)
    home_df = preprocessor.select_and_rename_columns('home_')

    expected_cols = {
        'match_api_id', 'season', 'stage', 'date',
        'team', 'opponent_goal', 'team_goal',
        'result_match', 'team_shoton', 'team_possession', 'is_home', 'away_team'
    }
    assert set(home_df.columns).issuperset(expected_cols)
    assert len(home_df) == len(sample_data)
    assert home_df['is_home'].unique().tolist() == [1]

def test_select_and_rename_columns_away(sample_data):
    """Test that columns for the away team are selected and renamed correctly."""
    preprocessor = ShiftDataPreprocessor(sample_data)
    away_df = preprocessor.select_and_rename_columns('away_')
    expected_cols = {
        'match_api_id', 'season', 'stage', 'date',
        'team', 'opponent_goal', 'team_goal',
        'result_match', 'team_shoton', 'team_possession', 'is_home', 'home_team'
    }
    assert set(away_df.columns).issuperset(expected_cols)
    assert len(away_df) == len(sample_data)
    assert away_df['is_home'].unique().tolist() == [0]

def test_concatenate_teams(sample_data):
    """Test that home and away DataFrames are concatenated properly."""
    preprocessor = ShiftDataPreprocessor(sample_data)
    home_df = preprocessor.select_and_rename_columns('home_')
    away_df = preprocessor.select_and_rename_columns('away_')
    team_df = preprocessor.concatenate_teams(home_df, away_df)

    assert len(team_df) == len(sample_data) * 2

    teams_in_order = team_df['team'].unique()
    assert list(teams_in_order) == sorted(teams_in_order, key=lambda x: x.lower())

def test_fill_na_and_zero_with_rolling_mean():
    """Test rolling mean imputation for NaN or zero values."""
    preprocessor = ShiftDataPreprocessor(pd.DataFrame())
    series = pd.Series([1, 0, 3, np.nan, 5, 0, 7], dtype=float)
    filled_series = preprocessor.fill_na_and_zero_with_rolling_mean_(series, window_size=2)

    assert len(filled_series) == len(series)

    assert not any(filled_series.isna())
    assert not any(filled_series == 0)

def test_shift_features(sample_data):
    """Test the shifting of features and the dropping of rows with missing data."""
    preprocessor = ShiftDataPreprocessor(sample_data)
    home_df = preprocessor.select_and_rename_columns('home_')
    away_df = preprocessor.select_and_rename_columns('away_')
    _ = preprocessor.concatenate_teams(home_df, away_df)

    features_to_shift = ['team_shoton', 'team_possession']
    shifted_df = preprocessor.shift_features(features_to_shift)

    for feature in features_to_shift:
        shifted_col = f"{feature}_shifted"
        assert shifted_col in shifted_df.columns

    for feature in features_to_shift:
        shifted_col = f"{feature}_shifted"
        assert not shifted_df[shifted_col].isna().any()
        assert not (shifted_df[shifted_col] == 0).any()

def test_merge_shifted_features(sample_data):
    """Test that the shifted features are properly merged back into the original dataframe."""
    preprocessor = ShiftDataPreprocessor(sample_data)

    home_df = preprocessor.select_and_rename_columns('home_')
    away_df = preprocessor.select_and_rename_columns('away_')
    team_df = preprocessor.concatenate_teams(home_df, away_df)

    features_to_shift = ['team_shoton', 'team_possession']
    shifted_df = preprocessor.shift_features(features_to_shift)

    merged_df = preprocessor.merge_shifted_features(shifted_df)

    assert any('home_prev_team_shoton_shifted' in c for c in merged_df.columns)
    assert any('away_prev_team_shoton_shifted' in c for c in merged_df.columns)

    assert 'home_shoton' not in merged_df.columns
    assert 'away_shoton' not in merged_df.columns

    for feature in features_to_shift:
        shifted_col = f"{feature}_shifted"
        assert not shifted_df[shifted_col].isna().any()
        assert not (shifted_df[shifted_col] == 0).any()