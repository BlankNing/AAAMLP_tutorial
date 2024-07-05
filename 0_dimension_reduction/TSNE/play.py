import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import imageio

digits = load_digits()
X = digits.data
y = digits.target

tsne = TSNE(n_components=2, random_state=42)
images = []
for i in range(250,350):
    tsne.n_iter = i
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.get_cmap('nipy_spectral', 10), edgecolor='none')
    plt.colorbar()
    plt.title(f'TSNE Iteration: {i}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'./tmp/tsne_iter_{i}.png')
    image = imageio.imread(f'./tmp/tsne_iter_{i}.png')
    images.append(image)

imageio.mimsave('tsne_animation.gif', images, duration=0.05)