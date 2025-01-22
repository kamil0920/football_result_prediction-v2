import pandas as pd
import numpy as np
import logging

class ShiftDataPreprocessor:
    def __init__(self, df):
        """
        Initializes the ShiftDataPreprocessor with the original DataFrame.

        Parameters:
        - df: pandas DataFrame containing the original match data.
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.team_df = None
        self.df_final = None
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.debug("FootballDataPreprocessor initialized.")

    def select_and_rename_columns(self, team_role):
        """
        Selects relevant columns for a team (home or away).

        Parameters:
        - team_role: 'home_' or 'away_' indicating the team role.

        Returns:
        - Renamed DataFrame for the team.
        """
        logging.debug(f"Selecting and renaming columns for {team_role.strip('_')} team.")

        if team_role == 'home_':
            team_columns = ['match_api_id', 'season', 'stage', 'date', 'home_team', 'away_team',
                            'home_team_goal', 'away_team_goal', 'result_match',
                            'home_shoton', 'home_possession']

        elif team_role == 'away_':
            team_columns = ['match_api_id', 'season', 'stage', 'date', 'away_team', 'home_team',
                            'home_team_goal', 'away_team_goal', 'result_match',
                            'away_shoton', 'away_possession']
        else:
            raise ValueError("team_role must be either 'home_' or 'away_'")

        selected_df = self.df_original[team_columns].copy()

        rename_base = {
            f'{team_role}team': 'team',
            f'{team_role}team_goal': 'team_goal',
            f'{"away_" if team_role == "home_" else "home_"}team_goal': 'opponent_goal',
            f'{team_role}shoton': 'team_shoton',
            f'{team_role}possession': 'team_possession'
        }
        selected_df = selected_df.rename(columns=rename_base)

        selected_df['is_home'] = 1 if team_role == 'home_' else 0

        logging.debug(f"Columns after renaming for {team_role.strip('_')} team:")
        logging.debug(selected_df.columns.tolist())

        return selected_df

    def concatenate_teams(self, home_df, away_df):
        """
        Concatenates home and away DataFrames into a single team-centric DataFrame.

        Parameters:
        - home_df: DataFrame containing home team data.
        - away_df: DataFrame containing away team data.

        Returns:
        - Concatenated team-centric DataFrame.
        """
        logging.debug("Concatenating home and away DataFrames.")

        # Concatenate
        team_df = pd.concat([home_df, away_df], ignore_index=True)

        # Sort by team and date
        team_df = team_df.sort_values(by=['team', 'date']).reset_index(drop=True)

        self.team_df = team_df

        logging.debug("Concatenation successful.")
        logging.debug(f"Concatenated DataFrame shape: {team_df.shape}")

        return team_df

    def fill_na_and_zero_with_rolling_mean(self, series: pd.Series, window_size=5) -> pd.Series:
        """
        Fill values that are NaN or 0 using a rolling mean over the last `window_size` valid data points.
        """
        # Copy the original to avoid mutating
        series_filled = series.copy()

        # Identify rows that are NaN or 0
        mask = series_filled.isna() | (series_filled == 0)

        # We'll create a temporary series where "missing" (NaN or 0) is replaced by NaN,
        # so that rolling().mean() ignores these values.
        temp_series = series_filled.where(~mask, np.nan)

        # Rolling mean for the series:
        rolled_values = temp_series.rolling(window=window_size, min_periods=1).mean()

        # Now overwrite only the missing (NaN or 0) spots in the original with the rolled values
        series_filled[mask] = rolled_values[mask]

        return series_filled

    def shift_features(self, features_to_shift):
        if self.team_df is None:
            raise ValueError("team_df is not defined.")

        shifted_df = self.team_df.copy()

        # STEP 1: Shift the features
        for feature in features_to_shift:
            shifted_col = f"{feature}_shifted"
            shifted_df[shifted_col] = shifted_df.groupby('team')[feature].shift(1)

        # STEP 2: Fill NaN or 0 with rolling mean
        for feature in features_to_shift:
            shifted_col = f"{feature}_shifted"
            shifted_df[shifted_col] = (
                shifted_df
                .groupby('team')[shifted_col]
                .apply(lambda grp: self.fill_na_and_zero_with_rolling_mean(grp, window_size=5))
                .reset_index(level=0, drop=True)
            )

        # STEP 4: Drop rows still missing
        #   If you only want to drop rows where the shifted_col is STILL NaN or 0, do something like:
        for feature in features_to_shift:
            shifted_col = f"{feature}_shifted"
            shifted_df = shifted_df[shifted_df[shifted_col].notna()]  # drop rows that remain NaN
            shifted_df = shifted_df[shifted_df[shifted_col] != 0]  # drop rows that remain 0

        return shifted_df

    def merge_shifted_features(self, team_df_shifted):
        """
        Merges the shifted home and away features back into the original DataFrame.

        Parameters:
        - home_prev: DataFrame containing shifted home features.
        - away_prev: DataFrame containing shifted away features.

        Returns:
        - Merged DataFrame.
        """
        logging.info("Merging shifted features back into the original DataFrame.")

        team_df_shifted = team_df_shifted.reset_index(drop=True)

        home_prev = self.team_df[self.team_df['is_home'] == 1][['match_api_id', 'team']].copy()

        shifted_cols = team_df_shifted.filter(like="_shifted").columns  # all cols containing '_shifted'
        columns_needed = ["match_api_id", "team"] + list(shifted_cols)

        home_prev = home_prev.merge(
            team_df_shifted[columns_needed],
            on=['match_api_id', 'team'],
            how='left'
        )

        home_prev = home_prev.rename(columns=lambda x: 'home_prev_' + x if x != 'match_api_id' else x)

        away_prev = self.team_df[self.team_df['is_home'] == 0][['match_api_id', 'team']].copy()

        away_prev = away_prev.merge(
            team_df_shifted[columns_needed],
            on=['match_api_id', 'team'],
            how='left'
        )

        away_prev = away_prev.rename(columns=lambda x: 'away_prev_' + x if x != 'match_api_id' else x)

        self.df_original = self.df_original.merge(home_prev, on='match_api_id', how='left')
        self.df_original = self.df_original.merge(away_prev, on='match_api_id', how='left')

        original_home_features = ['home_shoton', 'home_possession', 'home_prev_team' ]
        original_away_features = ['away_shoton', 'away_possession', 'away_prev_team' ]

        df_final = self.df_original.drop(columns=original_home_features + original_away_features)

        return df_final


