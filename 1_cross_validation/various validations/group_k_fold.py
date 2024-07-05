from sklearn.model_selection import GroupKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Sample data
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)
groups = np.repeat(np.arange(100), 10)  # 100 groups, each group has 10 samples

gkf = GroupKFold(n_splits=5)

for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
