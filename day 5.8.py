import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
np.random.seed(42)
n_samples = 500
C = np.array([[0., -0.7], [3.5, .7]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)
probs = gmm.predict_proba(X)
labels = gmm.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title('EM Algorithm with Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Component')
plt.show()
print(f"Means:\n{gmm.means_}")
print(f"Covariances:\n{gmm.covariances_}")
