import numpy as np

# Константы и параметры
np.random.seed(0)
NUM_POINTS = 500  # Количество точек
NUM_CLUSTERS = 3  # Количество кластеров
EXP_WEIGHT = 2    # Экспоненциальный вес
TOLERANCE = 1e-4  # Точность кластеризации


def generate_data():
    """
    Генерация данных для кластеризации.
    Возвращает массив точек X.
    """
    true_centers = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8, 0.8, 0.8],
        [0.5, 0.5, 0.5, 0.5, 0.5]
    ])
    
    # Распределение точек между кластерами
    cluster_sizes = [NUM_POINTS // NUM_CLUSTERS] * (NUM_CLUSTERS - 1)
    cluster_sizes.append(NUM_POINTS - sum(cluster_sizes))
    
    # Генерация точек
    data = np.vstack([
        np.random.normal(loc=true_centers[i], scale=0.1, size=(cluster_sizes[i], 5))
        for i in range(NUM_CLUSTERS)
    ])
    return data


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


def compute_cluster_variances(data, membership_matrix, centers):
    """
    Вычисление сигмы (σ) для каждого кластера.
    """
    variances = np.zeros(len(centers))
    for k in range(len(centers)):
        cluster_points = data * (1 - membership_matrix[:, k, np.newaxis])
        variance = np.mean(np.sum((cluster_points - centers[k]) ** 2, axis=1))
        variances[k] = np.sqrt(variance)
    return variances


def apply_fuzzifiers(distances, variances):
    """
    Применение различных фаззификаторов.
    """
    fuzz_gaussian = np.exp(- (distances ** 2) / (2 * variances ** 2))
    fuzz_gaussian /= fuzz_gaussian.sum(axis=1, keepdims=True)
    
    b_param = 2
    fuzz_generalized_gaussian = np.exp(-((distances / variances) ** b_param))
    fuzz_generalized_gaussian /= fuzz_generalized_gaussian.sum(axis=1, keepdims=True)
    
    fuzz_rational = 1 / (1 + (distances / variances) ** (2 * b_param))
    fuzz_rational /= fuzz_rational.sum(axis=1, keepdims=True)
    
    d_param = np.max(distances) / 2
    fuzz_triangular = np.maximum(1 - (distances / d_param), 0)
    fuzz_triangular /= (fuzz_triangular.sum(axis=1, keepdims=True) + np.finfo(float).eps)
    
    return fuzz_gaussian, fuzz_generalized_gaussian, fuzz_rational, fuzz_triangular


# Основная программа
if __name__ == "__main__":
    # Генерация данных
    X = generate_data()
    
    # Запуск алгоритма Fuzzy C-Means
    centers, membership_matrix = fuzzy_c_means(X, NUM_CLUSTERS, EXP_WEIGHT, TOLERANCE)
    
    # Вычисление сигмы
    variances = compute_cluster_variances(X, membership_matrix, centers)
    
    # Применение фаззификаторов
    distances = compute_distances(X, centers)
    fuzz_gaussian, fuzz_generalized, fuzz_rational, fuzz_triangular = apply_fuzzifiers(distances, variances)
    
    # Вывод результатов
    print("Исходные данные:")
    print(X)
    
    print("\nЦентры кластеров:")
    print(centers)
    
    print("\nИтоговая матрица принадлежности (Fuzzy C-Means):")
    print(membership_matrix)
    
    print("\nОтклонения от центров кластеров (σ):")
    print(variances)
    
    print("\nИтоговая матрица принадлежности (Гауссовская функция):")
    print(fuzz_gaussian)
    
    print("\nИтоговая матрица принадлежности (Обобщённая гауссовская функция):")
    print(fuzz_generalized)
    
    print("\nИтоговая матрица принадлежности (Рациональная функция):")
    print(fuzz_rational)
    
    print("\nИтоговая матрица принадлежности (Треугольная функция):")
    print(fuzz_triangular)