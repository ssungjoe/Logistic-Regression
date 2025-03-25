import numpy as np
import matplotlib.pyplot as plt

# 초기 parameter w=[1, -1, -1]
# 학습률 alpha와 iterations 선정
w_init = np.array([1, -1, -1]) 
alpha = 0.1
iterations = 800

# Hypothesis Function h(x)
def h_x(z):
    return 1 / (1 + np.exp(-z))

# Cost function J(w)
def J_w(X, y, w):
    m = len(y)
    h = h_x(np.dot(X, w)) 
    cost = - (1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Find Best Parameter using Gradient Descent
def gradient_descent(X, y, w, alpha, iterations):
    m = len(y)

    for i in range(iterations):
        h = h_x(np.dot(X, w))
        gradient = (1/m) * np.dot(X.T, (h - y))
        w = w - alpha * gradient
    
    return w

# 초기 학습 데이터 (X=[x_1, x_2], y={0, 1})
X = np.array([[-1, 1], [0, 2], [1, 1], [2, 0], [0, -1], [-1, -2], [1, -1], [-2, 0]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# vector <1, x_1, x_2>
X = np.c_[np.ones(X.shape[0]), X]

w_best = gradient_descent(X, y, w_init, alpha, iterations)

# Best Parameter w*
print(f"w* = {w_best.round(5)}")

cost_value = J_w(X, y, w_best)
print(f"J(w) = {cost_value.round(5)}")

# Decision Boundary
# w^T dot X = 0
x1_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
x2_boundary = -(w_best[1] * x1_range + w_best[0]) / w_best[2]

# Plot
plt.scatter(X[:, 1], X[:, 2], c=y, edgecolors='none', cmap='coolwarm', s=100)
plt.plot(x1_range, x2_boundary, color='limegreen', linestyle='--', label="Decision Boundary")
plt.title("Logistic Regression")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()