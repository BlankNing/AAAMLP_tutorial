import pandas as pd
from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    # Load the full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")

    # All columns are features except id, target and kfold columns
    features = [col for col in df.columns if col not in ("id", "target", "kfold")]

    # Fill all NaN values with NONE
    # Note that I am converting all columns to "strings"
    # It doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Initialize OneHotEncoder from scikit-learn
    ohe = preprocessing.OneHotEncoder()

    # Fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # Transform training data
    x_train = ohe.transform(df_train[features])

    # Transform validation data
    x_valid = ohe.transform(df_valid[features])

    # Initialize Truncated SVD
    # We are reducing the data to 120 components
    svd = decomposition.TruncatedSVD(n_components=120)

    # Fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)

    # Transform sparse training data
    x_train = svd.transform(x_train)

    # Transform sparse validation data
    x_valid = svd.transform(x_valid)

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
        run(fold_)
