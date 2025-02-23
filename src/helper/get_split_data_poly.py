import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def split_data_for_training(N_older_seasons=7):
    import os
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', 'data', 'engineered', 'raw_engineered_features.csv')
    df_matches = pd.read_csv(csv_path)

    df_matches = df_matches.sort_values(by=["season", "stage", "date"])

    season_dummies = pd.get_dummies(df_matches['season'], prefix='season', drop_first=True)
    df_matches = df_matches.join(season_dummies)

    sorted_seasons = sorted(df_matches["season"].unique())
    newest_season = sorted_seasons[-1]
    older_seasons = sorted_seasons[:-1]

    # max_stage = df_matches.loc[df_matches["season"] == newest_season, "stage"].max()
    max_stage = 17
    penultimate_stage = max_stage - 1

    train_seasons = sorted(older_seasons[-N_older_seasons:], reverse=True)

    X_train_old = df_matches[df_matches["season"].isin(train_seasons)]
    X_train_new = df_matches[(df_matches["season"] == newest_season) & (df_matches["stage"] < penultimate_stage)]
    df_train = pd.concat([X_train_old, X_train_new], ignore_index=True)

    df_val = df_matches[
        (df_matches["season"] == newest_season) & (df_matches["stage"] == penultimate_stage)].reset_index(
        drop=True)
    df_tst = df_matches[(df_matches["season"] == newest_season) & (df_matches["stage"] == max_stage)].reset_index(
        drop=True)

    feature_cols_to_drop = ["match_api_id", "result_match", "season", "stage", "date", "home_team", "away_team"]
    X_trn = df_train.drop(columns=feature_cols_to_drop)
    y_trn = df_train["result_match"]

    X_val = df_val.drop(columns=feature_cols_to_drop)
    y_val = df_val["result_match"]

    X_tst = df_tst.drop(columns=feature_cols_to_drop)
    y_tst = df_tst["result_match"]

    # Configure polynomial features for top interactions
    poly_features = [
        'points_difference',
        'team_acceleration_home',
        'team_strength_away',
        'away_last_team_possession'
    ]

    poly_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2,
                                    interaction_only=True,
                                    include_bias=False))
    ])

    # Apply to selected columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('poly', poly_transformer, poly_features)
        ],
        remainder='passthrough',
        force_int_remainder_cols=False
    )

    # Create preprocessing-only pipeline
    preprocessing_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    # Fit and transform data
    X_train_processed = preprocessing_pipeline.fit_transform(X_trn)
    X_val_processed = preprocessing_pipeline.transform(X_val)
    X_test_processed = preprocessing_pipeline.transform(X_tst)

    # Get readable feature names
    feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Create processed DataFrames
    trn_df = pd.DataFrame(X_train_processed, columns=feature_names)
    val_df = pd.DataFrame(X_val_processed, columns=feature_names)
    tst_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # Convert all columns to numeric, coercing errors to NaN if needed
    trn_df = trn_df.apply(pd.to_numeric, errors='coerce')
    val_df = val_df.apply(pd.to_numeric, errors='coerce')
    tst_df = tst_df.apply(pd.to_numeric, errors='coerce')

    return trn_df, y_trn, val_df, y_val, tst_df, y_tst