import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
np.random.seed(42)
n_samples = 300
mu1, sigma1 = [2, 2], [[1, 0.5], [0.5, 1]]
mu2, sigma2 = [7, 7], [[1, -0.5], [-0.5, 1]]
data1 = np.random.multivariate_normal(mu1, sigma1, n_samples)
data2 = np.random.multivariate_normal(mu2, sigma2, n_samples)
data = np.vstack((data1, data2))
k = 2
np.random.seed(42)
mu = data[np.random.choice(data.shape[0], k, False), :]
sigma = [np.cov(data.T)] * k
pi = [1/k] * k
def e_step(data, mu, sigma, pi):
    r = np.zeros((data.shape[0], k))
    for i in range(k):
        r[:, i] = pi[i] * multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i])
    r = r / r.sum(axis=1, keepdims=True)
    return r
def m_step(data, r):
    N_k = r.sum(axis=0)
    mu = np.dot(r.T, data) / N_k[:, np.newaxis]
    sigma = []
    for i in range(k):
        diff = data - mu[i]
        cov = np.dot(r[:, i] * diff.T, diff) / N_k[i]
        sigma.append(cov)
    pi = N_k / data.shape[0]
    return mu, sigma, pi
def log_likelihood(data, mu, sigma, pi):
    likelihood = 0
    for i in range(k):
        likelihood += pi[i] * multivariate_normal.pdf(data, mean=mu[i], cov=sigma[i])
    return np.sum(np.log(likelihood))
tol = 1e-4
log_likelihoods = []
for _ in range(100):
    r = e_step(data, mu, sigma, pi)
    mu, sigma, pi = m_step(data, r)
    log_likelihoods.append(log_likelihood(data, mu, sigma, pi))
    if len(log_likelihoods) < 2:
        continue
    if np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
        break
print("Means:", mu)
print("Covariances:", sigma)
print("Mixing coefficients:", pi)
plt.plot(log_likelihoods)
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood Progression')
plt.show()
plt.scatter(data[:, 0], data[:, 1], s=5)
for i in range(k):
    plt.scatter(mu[i][0], mu[i][1], c='red', marker='x')
    eigenvalues, eigenvectors = np.linalg.eigh(sigma[i])
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    theta = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
    for j in [1, 2]:
        ell = plt.matplotlib.patches.Ellipse((mu[i][0], mu[i][1]), 2 * np.sqrt(eigenvalues[0]), 2 * np.sqrt(eigenvalues[1]), angle=theta, edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(ell)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('EM Algorithm for GMM')
plt.show()
