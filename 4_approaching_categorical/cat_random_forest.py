import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # Load the full training data with folds
    df = pd.read_csv("./cat_train_folds.csv")

    # All columns are features except id, target and kfold columns
    features = [
        col for col in df.columns
        if col not in ("id", "target", "kfold")
    ]

    # Fill all NaN values with NONE
    # Note that I am converting all columns to "strings"
    # It doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Now it's time to label encode the features
    for col in features:
        # Initialize LabelEncoder for each feature column
        lbl = preprocessing.LabelEncoder()

        # Fit label encoder on all data
        lbl.fit(df[col])

        # Transform all the data
        df.loc[:, col] = lbl.transform(df[col])

    # Get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Get training data
    x_train = df_train[features].values

    # Get validation data
    x_valid = df_valid[features].values

    # Initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # Fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    # Predict on validation data
    # We need the probability values as we are calculating AUC
    # We will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # Get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # Print auc
    print(f"Fold = {fold}, AUC = {auc}")


if __name__ == "__main__":
    for fold_ in range(5):
        # Run function for fold_ = 0, 1, 2, ..., 4
        run(fold_)

