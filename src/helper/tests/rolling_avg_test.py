import pandas as pd
import pytest

from src.helper.rolling_avg import calculate_rolling_avg_pandas


@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'season': [2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024],
        'date': pd.to_datetime(['2023-08-01', '2023-08-08', '2023-08-15', '2023-08-22',
                                '2024-01-01', '2024-01-08', '2024-01-15', '2024-01-22']),
        'stage': [1, 2, 3, 4, 1, 2, 3, 4],
        'home_team': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A'],
        'away_team': ['B', 'A', 'C', 'A', 'B', 'C', 'A', 'C'],
        'home_last_team_goal': [2, 1, 3, 2, 1, 2, 3, 2],
        'away_last_team_goal': [1, 2, 1, 3, 2, 3, 2, 1]
    })
    return data


def test_basic_rolling_avg(sample_data):
    """Test rolling average calculation on a sample dataset."""
    df = calculate_rolling_avg_pandas(sample_data, window=3)

    assert 'rolling_avg_goals_home' in df.columns
    assert 'rolling_avg_goals_away' in df.columns

    assert df.iloc[2]['rolling_avg_goals_home'] is not None
    assert df.iloc[2]['rolling_avg_goals_away'] is not None


def test_edge_case_few_matches():
    """Test when a team has fewer matches than the rolling window."""
    df = pd.DataFrame({
        'season': [2023, 2023],
        'date': pd.to_datetime(['2023-08-01', '2023-08-08']),
        'stage': [1, 2],
        'home_team': ['A', 'B'],
        'away_team': ['B', 'A'],
        'home_last_team_goal': [2, 1],
        'away_last_team_goal': [1, 2]
    })

    df = calculate_rolling_avg_pandas(df, window=5)

    assert df['rolling_avg_goals_home'].notnull().all()
    assert df['rolling_avg_goals_away'].notnull().all()


def test_chronology_handling(sample_data):
    """Test if matches are sorted correctly before calculating rolling average."""
    df = calculate_rolling_avg_pandas(sample_data, window=3)

    sorted_df = df.sort_values(by=['season', 'stage', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(df, sorted_df)


def test_no_extra_rows_added(sample_data):
    """Ensure that the number of rows remains the same after merging rolling averages."""
    original_rows = sample_data.shape[0]
    df = calculate_rolling_avg_pandas(sample_data, window=3)
    assert df.shape[0] == original_rows


def test_missing_team_data():
    """Test behavior when a team appears only once in the dataset."""
    df = pd.DataFrame({
        'season': [2023],
        'date': [pd.to_datetime('2023-08-01')],
        'stage': [1],
        'home_team': ['A'],
        'away_team': ['B'],
        'home_last_team_goal': [2],
        'away_last_team_goal': [1]
    })

    df = calculate_rolling_avg_pandas(df, window=3)

    assert df['rolling_avg_goals_home'].notnull().all()
    assert df['rolling_avg_goals_away'].notnull().all()

