import pandas as pd

def split_data_for_training(N_older_seasons=7):
    import os
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, '..', '..', 'data', 'preprocessed', 'preprocessed_1.csv')
    df_matches = pd.read_csv(csv_path)

    df_matches = df_matches.sort_values(by=["season", "stage", "date"])

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

    return X_trn, y_trn, X_val, y_val, X_tst, y_tst