import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        logging.info("FootballDataPreprocessor initialized.")

    def select_and_rename_columns(self, team_role, rating_types):
        """
        Selects relevant columns for a team (home or away) and renames player rating columns.

        Parameters:
        - team_role: 'home_' or 'away_' indicating the team role.
        - rating_types: List of rating types to rename.

        Returns:
        - Renamed DataFrame for the team.
        """
        logging.info(f"Selecting and renaming columns for {team_role.strip('_')} team.")

        # if team_role == 'home_':
        #     team_columns = ['match_api_id', 'season', 'stage', 'date', 'home_team', 'away_team',
        #                     'home_team_goal', 'away_team_goal', 'result_match',
        #                     'home_shoton', 'home_possession']
        # elif team_role == 'away_':
        #     team_columns = ['match_api_id', 'season', 'stage', 'date', 'away_team', 'home_team',
        #                     'home_team_goal', 'away_team_goal', 'result_match',
        #                     'away_shoton', 'away_possession']
        # else:
        #     raise ValueError("team_role must be either 'home_' or 'away_'")

        selected_df = self.df_original.copy()

        rename_base = {
            f'{team_role}team': 'team',
            f'{team_role}team_goal': 'team_goal',
            f'{"away_" if team_role == "home_" else "home_"}team_goal': 'opponent_goal',
            f'{team_role}shoton': 'team_shoton',
            f'{team_role}possession': 'team_possession'
        }
        selected_df = selected_df.rename(columns=rename_base)

        for rating_type in rating_types:
            cols_to_rename = [col for col in self.df_original.columns if
                              col.startswith(f'{rating_type}_{team_role}player_')]
            rename_dict = {col: f'{team_role}{rating_type}_player_{col.split("_player_")[-1]}' for col in
                           cols_to_rename}
            selected_df = selected_df.rename(columns=rename_dict)

        selected_df['is_home'] = 1 if team_role == 'home_' else 0

        selected_df = selected_df.loc[:, ~selected_df.columns.duplicated()]

        logging.info(f"Columns after renaming for {team_role.strip('_')} team:")
        logging.info(selected_df.columns.tolist())

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
        logging.info("Concatenating home and away DataFrames.")

        # Reset index to ensure a default integer index
        home_df = home_df.reset_index(drop=True)
        away_df = away_df.reset_index(drop=True)

        cols_home = away_df.filter(like='home')
        cols_away = home_df.filter(like='away')
        cols_home['is_home'] = 1

        set_away_cols = set(cols_away)
        set_home_cols = set(cols_home)

        away_only_cols = set_away_cols.difference(set_home_cols)
        print("Columns only in cols_away:", away_only_cols)

        # Concatenate
        team_df = pd.concat([home_df, away_df], ignore_index=True)

        # Sort by team and date
        team_df = team_df.sort_values(by=['team', 'date']).reset_index(drop=True)

        self.team_df = team_df

        logging.info("Concatenation successful.")
        logging.info(f"Concatenated DataFrame shape: {team_df.shape}")

        return team_df

    def shift_features(self, features_to_shift):
        """
        Shifts specified features by one match for each team to create lagged features.

        Parameters:
        - features_to_shift: List of column names to shift.

        Returns:
        - DataFrame containing shifted features.
        """
        if self.team_df is None:
            raise ValueError("team_df is not defined. Please run concatenate_teams() first.")

        logging.info("Shifting features by one match for each team.")

        team_df_shifted = self.team_df.groupby('team')[features_to_shift].shift(1)

        logging.info("Shifting complete.")

        return team_df_shifted

    def rename_shifted_columns(self, shifted_df):
        """
        Renames shifted columns with appropriate prefixes based on team role.

        Parameters:
        - shifted_df: DataFrame containing shifted features.

        Returns:
        - Two DataFrames: home_prev_shifted and away_prev_shifted with renamed columns.
        """
        if self.team_df is None:
            raise ValueError("team_df is not defined. Please run concatenate_teams() first.")

        logging.info("Renaming shifted columns with 'home_prev_' and 'away_prev_' prefixes.")

        # Separate home and away shifted data
        home_prev = self.team_df[self.team_df['is_home'] == 1][['match_api_id']].reset_index(drop=True)
        away_prev = self.team_df[self.team_df['is_home'] == 0][['match_api_id']].reset_index(drop=True)

        # Shifted home features
        home_prev_shifted = shifted_df[self.team_df['is_home'] == 1].reset_index(drop=True)
        home_prev_shifted = home_prev_shifted.rename(columns=lambda x: f'home_prev_{x}')
        home_prev = pd.concat([home_prev, home_prev_shifted], axis=1)

        # Shifted away features
        away_prev_shifted = shifted_df[self.team_df['is_home'] == 0].reset_index(drop=True)
        away_prev_shifted = away_prev_shifted.rename(columns=lambda x: f'away_prev_{x}')
        away_prev = pd.concat([away_prev, away_prev_shifted], axis=1)

        logging.info("Renaming of shifted columns complete.")

        return home_prev, away_prev

    def merge_shifted_features(self, home_prev, away_prev):
        """
        Merges the shifted home and away features back into the original DataFrame.

        Parameters:
        - home_prev: DataFrame containing shifted home features.
        - away_prev: DataFrame containing shifted away features.

        Returns:
        - Merged DataFrame.
        """
        logging.info("Merging shifted features back into the original DataFrame.")

        # Merge shifted home features
        df_merged = self.df_original.merge(home_prev, on='match_api_id', how='left')

        # Merge shifted away features
        df_merged = df_merged.merge(away_prev, on='match_api_id', how='left')

        self.df = df_merged

        logging.info("Merging complete.")
        logging.info(f"Merged DataFrame shape: {df_merged.shape}")

        return df_merged

    def drop_original_columns(self):
        """
        Drops the original feature columns to prevent data leakage.

        Returns:
        - DataFrame with original features dropped.
        """
        logging.info("Dropping original feature columns to prevent data leakage.")

        # Define original home and away features precisely (excluding shifted columns)
        original_home_features = ['home_shoton', 'home_possession'] + [
            col for col in self.df.columns if col.startswith('player_rating_home_player_') or
                                              col.startswith('acceleration_rating_home_player_') or
                                              col.startswith('strength_rating_home_player_') or
                                              col.startswith('aggression_rating_home_player_')
        ]

        original_away_features = ['away_shoton', 'away_possession'] + [
            col for col in self.df.columns if col.startswith('player_rating_away_player_') or
                                              col.startswith('acceleration_rating_away_player_') or
                                              col.startswith('strength_rating_away_player_') or
                                              col.startswith('aggression_rating_away_player_')
        ]

        # Define columns_to_check as a flat list
        columns_to_check = original_home_features + original_away_features

        # Verify that all columns_to_check exist in df_
        missing_cols = [col for col in columns_to_check if col not in self.df.columns]
        if missing_cols:
            logging.warning(f"These columns are missing in df_: {missing_cols}")

        # Drop original feature columns
        df_final = self.df.drop(columns=original_home_features + original_away_features)

        self.df_final = df_final

        logging.info("Original feature columns dropped.")
        logging.info(f"Final DataFrame shape: {df_final.shape}")

        return df_final

    def impute_missing_values_row_based(self):
        """
        Imputes missing values in shifted features based on other values within the same match row.
        For numerical columns, fills NaN with the mean of available shifted features in the row.
        """
        if self.df_final is None:
            raise ValueError("df_final is not defined. Please run drop_original_columns() first.")

        logging.info("Imputing missing values based on match row.")

        # Identify shifted feature columns
        shifted_feature_cols = [col for col in self.df_final.columns if
                                col.startswith('home_prev_') or col.startswith('away_prev_')]

        # Function to impute a row
        def impute_row(row):
            # Calculate mean of available values in shifted features
            mean_val = row[shifted_feature_cols].mean()
            # Fill NaNs with the mean value
            row[shifted_feature_cols] = row[shifted_feature_cols].fillna(mean_val)
            return row

        # Apply the imputation row-wise
        self.df_final = self.df_final.apply(impute_row, axis=1)

        logging.info("Row-based imputation complete.")

        return self.df_final

    def handle_missing_data(self):
        """
        Handles missing data by imputing missing values based on match row values.
        Drops any remaining rows with NaNs in shifted features.

        Returns:
        - Cleaned DataFrame ready for analysis or modeling.
        """
        logging.info("Handling missing data.")

        # First, perform row-based imputation
        self.impute_missing_values_row_based()

        # Identify shifted feature columns
        shifted_feature_cols = [col for col in self.df_final.columns if
                                col.startswith('home_prev_') or col.startswith('away_prev_')]

        # Check for any remaining NaNs
        remaining_nans = self.df_final[shifted_feature_cols].isna().sum().sum()
        if remaining_nans > 0:
            logging.info(
                f"{remaining_nans} remaining NaN values found in shifted features. Dropping corresponding rows.")
            # Drop rows with any NaNs in shifted features
            self.df_final = self.df_final.dropna(subset=shifted_feature_cols).reset_index(drop=True)
        else:
            logging.info("No remaining NaN values in shifted features.")

        logging.info(f"DataFrame shape after handling missing data: {self.df_final.shape}")

        return self.df_final

    def plot_nan_summary(self, nan_summary, title='NaN Percentage per Column'):
        """
        Plots a bar chart of NaN percentages per column.

        Parameters:
        - nan_summary: DataFrame containing 'NaN Percentage (%)' indexed by column names.
        - title: Title of the plot.
        """
        plt.figure(figsize=(12, 6))
        nan_summary['NaN Percentage (%)'].plot(kind='bar', color='skyblue')
        plt.ylabel('NaN Percentage (%)')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def get_nan_summary(self, columns_to_check):
        """
        Calculates and returns the NaN count and percentage for specified columns.

        Parameters:
        - columns_to_check: List of column names to check for NaN values.

        Returns:
        - DataFrame summarizing NaN counts and percentages.
        """
        nan_counts = self.df_original[columns_to_check].isna().sum()
        nan_percent = (nan_counts / len(self.df_original)) * 100
        nan_summary = pd.DataFrame({
            'NaN Count': nan_counts,
            'NaN Percentage (%)': nan_percent
        }).sort_values(by='NaN Percentage (%)', ascending=False)
        return nan_summary
