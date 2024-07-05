import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
data = np.random.randn(1000)  # 1000个样本

# Sturge's Rule
num_bins_sturge = int(np.ceil(1 + np.log2(len(data))))
print(f'Sturge\'s Rule: {num_bins_sturge} bins')

# Scott's Rule
bin_width_scott = 3.5 * np.std(data) / (len(data) ** (1/3))
num_bins_scott = int(np.ceil((data.max() - data.min()) / bin_width_scott))
print(f'Scott\'s Rule: {num_bins_scott} bins')

# Freedman-Diaconis Rule
iqr = np.percentile(data, 75) - np.percentile(data, 25)
bin_width_fd = 2 * iqr / (len(data) ** (1/3))
num_bins_fd = int(np.ceil((data.max() - data.min()) / bin_width_fd))
print(f'Freedman-Diaconis Rule: {num_bins_fd} bins')

# 绘制直方图对比
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(data, bins=num_bins_sturge, edgecolor='black')
plt.title(f'Sturge\'s Rule ({num_bins_sturge} bins)')

plt.subplot(1, 3, 2)
plt.hist(data, bins=num_bins_scott, edgecolor='black')
plt.title(f'Scott\'s Rule ({num_bins_scott} bins)')

plt.subplot(1, 3, 3)
plt.hist(data, bins=num_bins_fd, edgecolor='black')
plt.title(f'Freedman-Diaconis Rule ({num_bins_fd} bins)')

plt.tight_layout()
plt.show()
