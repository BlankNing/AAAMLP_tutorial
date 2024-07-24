import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == "__main__":
    # Read training data
    df = pd.read_csv("../input/imdb.csv")

    # Map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    # Create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # Randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Fetch labels
    y = df.sentiment.values

    # Initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # Go over the folds created
    for fold_ in range(5):
        # Temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)

        # Initialize CountVectorizer with NLTK's word_tokenize function as tokenizer
        count_vec = CountVectorizer(
            tokenizer=word_tokenize,
            token_pattern=None
        )

        # Fit count_vec on training data reviews
        count_vec.fit(train_df.review)

        # Transform training and validation data reviews
        xtrain = count_vec.transform(train_df.review)
        xtest = count_vec.transform(test_df.review)

        # Initialize logistic regression model
        model = linear_model.LogisticRegression()

        # Fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)

        # Make predictions on test data
        # Threshold for predictions is 0.5
        preds = model.predict(xtest)

        # Calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)

        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")
