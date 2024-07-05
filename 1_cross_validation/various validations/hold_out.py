from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data
X = np.random.rand(1000, 20)  # 1000 samples, 20 features
y = np.random.randint(0, 2, 1000)  # Binary target

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on the training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate the model on the validation set
y_pred = model.predict(X_val)
print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
