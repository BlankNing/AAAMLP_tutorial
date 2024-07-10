# example: ask to predict different objects in a given image
# chair,vessel,window,micro-oven
# you need to determine if the object exists in that environment

'''
If you have a list of original classes for a given 
sample and list of predicted classes for the same, precision is defined as the number 
of hits in the predicted list considering only top-k predictions, divided by k.
'''

def pk(y_true, y_pred, k):
    """
    This function calculates precision at k 
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :param k: the value for k
    :return: precision at a given value k
    """
    # if k is 0, return 0. we should never have this
    # as k is always >= 1
    if k == 0:
        return 0
    # we are interested only in top-k predictions
    y_pred = y_pred[:k]
    # convert predictions to set
    pred_set = set(y_pred)
    # convert actual values to set
    true_set = set(y_true)
    # find common values
    common_values = pred_set.intersection(true_set)
    # return length of common values over k
    return len(common_values) / len(y_pred[:k])


def apk(y_true, y_pred, k):
    """
    This function calculates average precision at k 
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: average precision at a given value k
    """
    # initialize p@k list of values
    pk_values = []
    # loop over all k. from 1 to k + 1
    for i in range(1, k + 1):
        # calculate p@i and append to list
        pk_values.append(pk(y_true, y_pred, i))
    # if we have no values in the list, return 0
    if len(pk_values) == 0:
        return 0
    # else, we return the sum of list over length of list
    return sum(pk_values) / len(pk_values)

def mapk(y_true, y_pred, k):
    """
    This function calculates mean avg precision at k 
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean avg precision at a given value k
    """
    # initialize empty list for apk values
    apk_values = []
    # loop over all samples
    for i in range(len(y_true)):
        # store apk values for every sample
        apk_values.append(
        apk(y_true[i], y_pred[i], k=k)
        )
    # return mean of apk values list
    return sum(apk_values) / len(apk_values)

if __name__ =='__main__':
    y_true = [[1, 2, 3], [0, 2], [1], [2, 3], [1, 0], []]
    y_pred = [[0, 1, 2], [1], [0, 2, 3], [2, 3, 4, 0], [0, 1, 2], [0]]
    
    for i in range(len(y_true)):
        for j in range(1, 4):
            print(
                f"""
                y_true={y_true[i]},
                y_pred={y_pred[i]},
                AP@{j}={apk(y_true[i], y_pred[i], k=j)}
                """
            )
    
    print(mapk(y_true, y_pred, k=1))
    
    print(mapk(y_true, y_pred, k=2))

    print(mapk(y_true, y_pred, k=3))

    print(mapk(y_true, y_pred, k=4))

# How do you implement log loss for multi-label classification? try it yourself

