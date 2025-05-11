import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Загрузка выборки
df = pd.read_csv("./pr1/generated_dataset.csv")
X = df.iloc[:, :5].values.T  # (features, samples)

# Кластеризация методом C-средних
n_clusters = 3
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

print("Центры кластеров:")
print(cntr)

# Подготовка данных по x1
x1 = df["x1"].values
centers_x1 = [c[0] for c in cntr]

# Определение принадлежности каждой точки к кластеру (по максимуму в строке матрицы u)
cluster_assignments = np.argmax(u, axis=0)

# Функции принадлежности
def gaussian_membership(x, c, sigma=0.1):
    return np.exp(-np.square(x - c) / (2 * sigma ** 2))

def generalized_gaussian_membership(x, c, sigma=0.1, beta=3):
    return np.exp(-np.power(np.abs(x - c), beta) / (2 * sigma ** 2))

def triangular_membership(x, a, b, c):
    return np.maximum(np.minimum((x - a)/(b - a), (c - x)/(c - b)), 0)

def trapezoidal_membership(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x - a)/(b - a), 1), (d - x)/(d - c)), 0)

# Визуализация всех функций по x1
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Цвета для кластеров
colors = ['r', 'g', 'b']

# 1. Гауссова
for i in range(n_clusters):
    mu = gaussian_membership(x1, centers_x1[i])
    axs[0].scatter(x1, mu, color=colors[i], alpha=0.6, label=f'Кластер {i+1}', s=20)
    axs[0].axvline(centers_x1[i], color=colors[i], linestyle='--', linewidth=1.5)
axs[0].set_title("Гауссовы функции принадлежности (точки)")
axs[0].legend()
axs[0].grid(True)

# 2. Обобщённая гауссова
for i in range(n_clusters):
    mu = generalized_gaussian_membership(x1, centers_x1[i])
    axs[1].scatter(x1, mu, color=colors[i], alpha=0.6, label=f'Кластер {i+1}', s=20)
    axs[1].axvline(centers_x1[i], color=colors[i], linestyle='--', linewidth=1.5)
axs[1].set_title("Обобщённые гауссовы функции принадлежности (β=3) (точки)")
axs[1].legend()
axs[1].grid(True)

# 3. Треугольная
delta = 0.1
for i in range(n_clusters):
    a, b, c_ = centers_x1[i] - delta, centers_x1[i], centers_x1[i] + delta
    mu = triangular_membership(x1, a, b, c_)
    axs[2].scatter(x1, mu, color=colors[i], alpha=0.6, label=f'Кластер {i+1}', s=20)
    axs[2].axvline(centers_x1[i], color=colors[i], linestyle='--', linewidth=1.5)
axs[2].set_title("Треугольные функции принадлежности (точки)")
axs[2].legend()
axs[2].grid(True)

# 4. Трапециевидная
for i in range(n_clusters):
    a, b, c_, d = centers_x1[i] - 0.15, centers_x1[i] - 0.05, centers_x1[i] + 0.05, centers_x1[i] + 0.15
    mu = trapezoidal_membership(x1, a, b, c_, d)
    axs[3].scatter(x1, mu, color=colors[i], alpha=0.6, label=f'Кластер {i+1}', s=20)
    axs[3].axvline(centers_x1[i], color=colors[i], linestyle='--', linewidth=1.5)
axs[3].set_title("Трапециевидные функции принадлежности (точки)")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()