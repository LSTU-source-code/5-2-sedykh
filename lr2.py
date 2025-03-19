import numpy as np
import matplotlib.pyplot as plt


# Параметры
np.random.seed(0)
NUM_POINTS = 500  # Количество точек
NUM_CLUSTERS = 3  # Количество кластеров
EXP_WEIGHT = 2    # Экспоненциальный вес
TOLERANCE = 1e-4  # Точность кластеризации
THRESHOLD = 0.3   # Порог принадлежности


def generate_data():
    """
    Генерация данных для кластеризации.
    Возвращает массив точек X и целевые значения Y.
    """
    true_centers = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8, 0.8, 0.8],
        [0.5, 0.5, 0.5, 0.5, 0.5]
    ])
    
    cluster_sizes = [NUM_POINTS // NUM_CLUSTERS] * (NUM_CLUSTERS - 1)
    cluster_sizes.append(NUM_POINTS - sum(cluster_sizes))
    
    X = np.vstack([
        np.random.normal(loc=true_centers[i], scale=0.1, size=(cluster_sizes[i], 5))
        for i in range(NUM_CLUSTERS)
    ])
    Y = np.sin(np.sum(X, axis=1))
    return X, Y


def initialize_membership_matrix(num_points, num_clusters):
    """
    Инициализация матрицы принадлежности U.
    """
    membership_matrix = np.random.rand(num_points, num_clusters)
    membership_matrix /= membership_matrix.sum(axis=1, keepdims=True)
    return membership_matrix


def compute_cluster_centers(membership_matrix, data, exp_weight):
    """
    Вычисление центров кластеров.
    """
    numerator = np.dot((membership_matrix ** exp_weight).T, data)
    denominator = membership_matrix ** exp_weight
    denominator = denominator.sum(axis=0, keepdims=True).T
    return numerator / denominator


def compute_distances(data, centers):
    """
    Вычисление расстояний от точек до центров кластеров.
    """
    distances = np.zeros((data.shape[0], centers.shape[0]))
    for k in range(centers.shape[0]):
        diff = data - centers[k]
        distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))
    return distances


def update_membership_matrix(distances, exp_weight):
    """
    Обновление матрицы принадлежности U.
    """
    exponent = 2 / (exp_weight - 1)
    temp = 1 / (distances ** exponent)
    temp[~np.isfinite(temp)] = 0  # Защита от деления на ноль
    denominator = temp.sum(axis=1, keepdims=True)
    return temp / denominator


def fuzzy_c_means(data, num_clusters, exp_weight, tolerance):
    """
    Алгоритм Fuzzy C-Means.
    Возвращает центры кластеров и финальную матрицу принадлежности.
    """
    membership_matrix = initialize_membership_matrix(data.shape[0], num_clusters)
    
    while True:
        centers = compute_cluster_centers(membership_matrix, data, exp_weight)
        distances = compute_distances(data, centers)
        distances[distances == 0] = np.finfo(float).eps
        
        new_membership_matrix = update_membership_matrix(distances, exp_weight)
        
        if np.max(np.abs(new_membership_matrix - membership_matrix)) < tolerance:
            break
        
        membership_matrix = new_membership_matrix
    
    return centers, membership_matrix


def compute_regression_coefficients(X, Y):
    """
    Вычисление коэффициентов регрессии.
    """
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    B = np.linalg.inv(A.T @ A) @ A.T @ Y
    return B


def compute_mse(Y_true, Y_pred):
    """
    Вычисление среднеквадратичной ошибки (MSE).
    """
    return np.mean((Y_true - Y_pred) ** 2)


def defuzzify_center_of_gravity(U_row, Y_preds):
    """
    Дефаззификация методом центра тяжести.
    """
    return np.sum(U_row * Y_preds) / np.sum(U_row)


def defuzzify_mean_max(U_row, Y_preds):
    """
    Дефаззификация методом среднего максимума.
    """
    max_value = np.max(U_row)
    indices = np.where(U_row == max_value)[0]
    return np.mean(Y_preds[indices])


def defuzzify_max_membership(U_row, Y_preds):
    """
    Дефаззификация методом максимума принадлежности.
    """
    return Y_preds[np.argmax(U_row)]


def plot_results(Y_true, Y_pred, title, color='red'):
    """
    Визуализация результатов.
    """
    plt.figure(figsize=(15, 5))
    plt.scatter(range(NUM_POINTS), Y_true, color='green', label='Фактические Y', alpha=0.6)
    plt.scatter(range(NUM_POINTS), Y_pred, color=color, label='Предсказанные Y', alpha=0.6)
    plt.title(title)
    plt.xlabel("Номер точки")
    plt.ylabel("Значение Y")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Генерация данных
    X, Y = generate_data()
    
    # Запуск алгоритма Fuzzy C-Means
    centers, membership_matrix = fuzzy_c_means(X, NUM_CLUSTERS, EXP_WEIGHT, TOLERANCE)
    
    # Разделение данных на кластеры
    max_membership_indices = np.argmax(membership_matrix, axis=1)
    clusters = [
        (X[max_membership_indices == k], Y[max_membership_indices == k])
        for k in range(NUM_CLUSTERS)
    ]
    
    # Вычисление коэффициентов регрессии для каждого кластера
    regression_coeffs = [
        compute_regression_coefficients(clusters[k][0], clusters[k][1])
        for k in range(NUM_CLUSTERS)
    ]
    
    # Предсказание Y для каждого кластера
    Y_reg = np.zeros_like(Y)
    for i in range(NUM_POINTS):
        cluster_idx = max_membership_indices[i]
        Y_reg[i] = np.dot(np.append(X[i], 1), regression_coeffs[cluster_idx])
    
    # Вычисление MSE для кластеризованных данных
    mse_clustered = compute_mse(Y, Y_reg)
    print("\nMSE (Кластеризованные данные):", mse_clustered)
    plot_results(Y, Y_reg, "Метод с тремя уравнениями регрессии", color='brown')
    
    # Предсказание Y для некластеризованных данных
    overall_coeffs = compute_regression_coefficients(X, Y)
    Y_reg_overall = np.array([np.dot(np.append(X[i], 1), overall_coeffs) for i in range(NUM_POINTS)])
    mse_non_clustered = compute_mse(Y, Y_reg_overall)
    print("\nMSE (Некластеризованные данные):", mse_non_clustered)
    plot_results(Y, Y_reg_overall, "Метод с одним уравнением регрессии", color='orange')
    
    # Дефаззификация
    Y_defuzzified_cog = np.zeros(NUM_POINTS)
    Y_defuzzified_mean_max = np.zeros(NUM_POINTS)
    Y_defuzzified_max = np.zeros(NUM_POINTS)
    
    for i in range(NUM_POINTS):
        valid_clusters = np.where(membership_matrix[i] > THRESHOLD)[0]
        if len(valid_clusters) == 0:
            continue
        
        Y_preds = []
        U_valid = []
        for k in valid_clusters:
            Y_preds.append(np.dot(np.append(X[i], 1), regression_coeffs[k]))
            U_valid.append(membership_matrix[i, k])
        
        Y_preds = np.array(Y_preds)
        U_valid = np.array(U_valid)
        
        Y_defuzzified_cog[i] = defuzzify_center_of_gravity(U_valid, Y_preds)
        Y_defuzzified_mean_max[i] = defuzzify_mean_max(U_valid, Y_preds)
        Y_defuzzified_max[i] = defuzzify_max_membership(U_valid, Y_preds)
    
    # Вычисление MSE для дефаззификации
    mse_cog = compute_mse(Y, Y_defuzzified_cog)
    mse_mean_max = compute_mse(Y, Y_defuzzified_mean_max)
    mse_max = compute_mse(Y, Y_defuzzified_max)
    
    print("\nMSE (Метод центра тяжести):", mse_cog)
    print("MSE (Метод среднего максимума):", mse_mean_max)
    print("MSE (Метод максимума принадлежности):", mse_max)
    
    # Визуализация дефаззификации
    plot_results(Y, Y_defuzzified_cog, "Метод центра тяжести", color='red')
    plot_results(Y, Y_defuzzified_mean_max, "Метод среднего максимума", color='blue')
    plot_results(Y, Y_defuzzified_max, "Метод максимума принадлежности", color='purple')


if __name__ == "__main__":
    main()