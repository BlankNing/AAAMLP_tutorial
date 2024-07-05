from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载MNIST数据集
digits = load_digits()
X = digits.data
y = digits.target

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化降维后的数据
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title('PCA Visualization of MNIST Dataset')
plt.savefig('pca_mnist.png')
plt.show()

# differences between tsne and pca:
# https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/