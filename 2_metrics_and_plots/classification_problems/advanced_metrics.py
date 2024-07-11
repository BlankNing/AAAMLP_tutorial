from sklearn import metrics
from pred_true_threshold_roc import true_positive, false_positive, true_negative, false_negative

'''
quadratic weighted kappa QWK Cohen's kappa

QWK measures the “agreement” between two “ratings”. The ratings can be any real numbers in 0 to N. And 
predictions are also in the same range.

So, it’s suitable for a classification problem with N different categories/classes. 
If the agreement is high, the score is closer towards 1.0. 
In the case of low agreement, the score is close to 0. 

You can see that even though accuracy is high, QWK is less. A QWK greater than 
0.85 is considered to be very good!
'''

y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]

cohen_quad = metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")
accuracy = metrics.accuracy_score(y_true, y_pred)

print(cohen_quad, accuracy)


'''
An important metric is Matthew’s Correlation Coefficient (MCC). MCC ranges 
from -1 to 1. 1 is perfect prediction, -1 is imperfect prediction, and 0 is random 
prediction. The formula for MCC is quite simple.
'''

def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient
    for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    numerator = (tp * tn) - (fp * fn)
    denominator = (
    (tp + fp) *
    (fn + tn) *
    (fp + tn) *
    (tp + fn)
    )
    denominator = denominator ** 0.5
    return numerator/denominator

'''
One thing to keep in mind is that to evaluate un-supervised methods, for example, 
some kind of clustering, it’s better to create or manually label the test set and keep 
it separate from everything that is going on in your modelling part. When you are 
done with clustering, you can evaluate the performance on the test set simply by 
using any of the supervised learning metrics
'''