import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the MNIST dataset
digits = datasets.load_digits()
data = digits.data
y = digits.target

# 可视化前25个图像
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(data[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Label: {y[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Perform preprocessing standard scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform t-SNE decomposition
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(data_scaled)

# Plot the t-SNE embedding
plt.figure(figsize=(10, 8))
colors = ['navy', 'darkorange', 'cornflowerblue', 'teal', 'crimson', 'gray', 'olive', 'gold', 'hotpink', 'deeppink']
for digit, color in zip(np.unique(y), colors):
    plt.scatter(X_tsne[y == digit, 0], X_tsne[y == digit, 1], c=color, label=digit, marker='o', s=10)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('t-SNE Embedding of MNIST Dataset')
plt.savefig('tsne_mnist.png',dpi=300)
plt.show()