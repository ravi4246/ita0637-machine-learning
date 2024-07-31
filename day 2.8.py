import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(42)
n_samples = 500
mean1 = [0, 0]
cov1 = [[1, 0.8], [0.8, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 2)
mean2 = [5, 5]
cov2 = [[1, -0.8], [-0.8, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 2)
data = np.vstack([data1, data2])
def em_algorithm(data, n_components, n_iter=100):
    n_samples, n_features = data.shape
    means = np.random.rand(n_components, n_features)
    covariances = np.array([np.eye(n_features)] * n_components)
    weights = np.ones(n_components) / n_components
    for _ in range(n_iter):
        responsibilities = np.zeros((n_samples, n_components))
        for i in range(n_components):
            responsibilities[:, i] = weights[i] * multivariate_normal.pdf(data, means[i], covariances[i])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        for i in range(n_components):
            weight_sum = responsibilities[:, i].sum()
            means[i] = (responsibilities[:, i] @ data) / weight_sum
            diff = data - means[i]
            covariances[i] = (responsibilities[:, i] * diff.T @ diff) / weight_sum
            weights[i] = weight_sum / n_samples
    return means, covariances, weights
n_components = 2
means, covariances, weights = em_algorithm(data, n_components)
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], c='gray', s=10, label='Data')
x = np.linspace(-3, 8, 100)
y = np.linspace(-3, 8, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack([X, Y])
for i in range(n_components):
    rv = multivariate_normal(means[i], covariances[i])
    plt.contour(X, Y, rv.pdf(pos), levels=[0.1, 0.2, 0.3], label=f'Component {i+1}')
plt.legend()
plt.title('Gaussian Mixture Model - EM Algorithm')
dbplt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
