import pandas as pd
df = pd.read_csv('/home/ningzhou/PycharmProjects/AAAMLP/1_cross_validation/winequality-red.csv')
quality_mapping = {
 3: 0,
 4: 1,
 5: 2,
 6: 3,
 7: 4,
 8: 5
}
df.loc[:, "quality"] = df.quality.map(quality_mapping)
df = df.sample(frac=1).reset_index(drop=True)
df_train = df.head(1000)
df_test = df.tail(599)


# import scikit-learn tree and metrics
from sklearn import tree
from sklearn import metrics
# import matplotlib and seaborn
# for plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# this is our global size of label text
# on the plots
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
# # This line ensures that the plot is displayed
# # inside the notebook
# %matplotlib inline
# initialize lists to store accuracies
# for training and test data
# we start with 50% accuracy
train_accuracies = [0.5]
test_accuracies = [0.5]
# iterate over a few depth values
for depth in range(1, 25):
 # init the model
 clf = tree.DecisionTreeClassifier(max_depth=depth)
 # columns/features for training
 # note that, this can be done outside 
 # the loop
 cols = [
 'fixed acidity', 
 'volatile acidity',
 'citric acid', 
 'residual sugar',
 'chlorides',
 'free sulfur dioxide', 
 'total sulfur dioxide',
 'density',
 'pH', 
 'sulphates',
  'alcohol'
 ]
 # fit the model on given features
 clf.fit(df_train[cols], df_train.quality)
 # create training & test predictions
 train_predictions = clf.predict(df_train[cols])
 test_predictions = clf.predict(df_test[cols])
 # calculate training & test accuracies
 train_accuracy = metrics.accuracy_score(
 df_train.quality, train_predictions
 )
 test_accuracy = metrics.accuracy_score(
 df_test.quality, test_predictions
 )
 
 # append accuracies
 train_accuracies.append(train_accuracy)
 test_accuracies.append(test_accuracy)
# create two plots using matplotlib
# and seaborn
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
plt.legend(loc="upper left", prop={'size': 15})
plt.xticks(range(0, 26, 5))
plt.xlabel("max_depth", size=20)
plt.ylabel("accuracy", size=20)
plt.savefig('dtree_acc_depth_plot.png',dpi=300)
plt.show()