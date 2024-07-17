# rf_gp_minimize.py
import numpy as np
import pandas as pd
from functools import partial
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from skopt import gp_minimize
from skopt import space
from skopt.plots import plot_convergence


def optimize(params, param_names, x, y):
    """
    The main optimization function.
    This function takes all the arguments from the search space
    and training features and targets. It then initializes
    the models by setting the chosen parameters and runs
    cross-validation and returns a negative accuracy score
    :param params: list of params from gp_minimize
    :param param_names: list of param names. order is important!
    :param x: training data
    :param y: labels/targets
    :return: negative accuracy after 5 folds
    """
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_accuracy = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_accuracy)
    return -1 * np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv("./mobile_train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]
    optimization_function = partial(optimize, param_names=param_names, x=X, y=y)
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )
    best_params = dict(zip(param_names, result.x))
    print(best_params)
    plot_convergence(result)

'''
Iteration No: 1 started. Evaluating function at random point.
Iteration No: 1 ended. Evaluation done at random point.
Time taken: 86.3882
Function value obtained: -0.8295
Current minimum: -0.8295
Iteration No: 2 started. Evaluating function at random point.
Iteration No: 2 ended. Evaluation done at random point.
Time taken: 34.2156
Function value obtained: -0.8515
Current minimum: -0.8515
Iteration No: 3 started. Evaluating function at random point.
Iteration No: 3 ended. Evaluation done at random point.
Time taken: 54.8496
Function value obtained: -0.8815
Current minimum: -0.8815
Iteration No: 4 started. Evaluating function at random point.
Iteration No: 4 ended. Evaluation done at random point.
Time taken: 93.2673
Function value obtained: -0.8885
Current minimum: -0.8885
Iteration No: 5 started. Evaluating function at random point.
Iteration No: 5 ended. Evaluation done at random point.
Time taken: 100.4486
Function value obtained: -0.8905
Current minimum: -0.8905
Iteration No: 6 started. Evaluating function at random point.
Iteration No: 6 ended. Evaluation done at random point.
Time taken: 79.2433
Function value obtained: -0.8855
Current minimum: -0.8905
Iteration No: 7 started. Evaluating function at random point.
Iteration No: 7 ended. Evaluation done at random point.
Time taken: 74.8305
Function value obtained: -0.8785
Current minimum: -0.8905
Iteration No: 8 started. Evaluating function at random point.
Iteration No: 8 ended. Evaluation done at random point.
Time taken: 138.5076
Function value obtained: -0.8875
Current minimum: -0.8905
Iteration No: 9 started. Evaluating function at random point.
Iteration No: 9 ended. Evaluation done at random point.
Time taken: 28.6222
Function value obtained: -0.8965
Current minimum: -0.8965
Iteration No: 10 started. Evaluating function at random point.
Iteration No: 10 ended. Evaluation done at random point.
Time taken: 42.7720
Function value obtained: -0.8975
Current minimum: -0.8975
Iteration No: 11 started. Searching for the next optimal point.
Iteration No: 11 ended. Search finished for the next optimal point.
Time taken: 221.4024
Function value obtained: -0.9055
Current minimum: -0.9055
Iteration No: 12 started. Searching for the next optimal point.
Iteration No: 12 ended. Search finished for the next optimal point.
Time taken: 21.2692
Function value obtained: -0.8920
Current minimum: -0.9055
Iteration No: 13 started. Searching for the next optimal point.
Iteration No: 13 ended. Search finished for the next optimal point.
Time taken: 111.0204
Function value obtained: -0.7645
Current minimum: -0.9055
Iteration No: 14 started. Searching for the next optimal point.
Iteration No: 14 ended. Search finished for the next optimal point.
Time taken: 7.5631
Function value obtained: -0.6950
Current minimum: -0.9055
Iteration No: 15 started. Searching for the next optimal point.
Iteration No: 15 ended. Search finished for the next optimal point.
Time taken: 212.2756
Function value obtained: -0.9020
Current minimum: -0.9055
{'max_depth': 15, 'n_estimators': 1408, 'criterion': 'entropy', 'max_features': 0.9784855466232195}
'''