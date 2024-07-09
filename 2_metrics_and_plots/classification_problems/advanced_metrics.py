from sklearn import metrics

y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

cohen_quad = metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
accuracy = metrics.accuracy_score(y_true, y_pred)

print(cohen_quad, accuracy)
