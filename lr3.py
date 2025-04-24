import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# --- Функции принадлежности ---
def trapezoidal_membership(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)

def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

# --- Ввод типа функции ---
mf_type = input("Выберите тип функции принадлежности (Трапециевидная/Треугольная): ").strip()

# --- Ввод параметров ---
def get_params(n):
    result = []
    for i in range(1, n + 1):
        while True:
            try:
                raw = input(f"Параметры для A{i} (через пробел): ")
                values = list(map(float, raw.split()))
                expected = 4 if mf_type == "Трапециевидная" else 3
                if len(values) == expected:
                    result.append(values)
                    break
                else:
                    print(f"Введите {expected} значения.")
            except ValueError:
                print("Ошибка: попробуйте снова.")
    return result

params = get_params(5)

# --- Построение функций принадлежности ---
membership_fn = trapezoidal_membership if mf_type == "Трапециевидная" else triangular_membership
mu = [lambda x, p=p: membership_fn(x, *p) for p in params]

# --- Ввод линейных функций ---
x_sym = sp.Symbol('x')

def is_linear(expr):
    return expr.is_polynomial(x_sym) and expr.as_poly(x_sym).degree() <= 1

def get_linear_function(prompt):
    while True:
        try:
            expr = sp.sympify(input(prompt))
            if is_linear(expr):
                return sp.lambdify(x_sym, expr, 'numpy')
            print("Введите линейную функцию вида ax + b.")
        except sp.SympifyError:
            print("Ошибка синтаксиса. Попробуйте снова.")

fs = [get_linear_function(f"f{i+1}(x) = ") for i in range(5)]

# --- Расчет выходной функции ---
def takagi_sugeno(x):
    return sum(mu[i](x) * fs[i](x) for i in range(5))

# --- Ввод x для вычисления ---
def get_input_values():
    while True:
        try:
            raw = input("Введите значения x через пробел: ")
            return list(map(float, raw.split()))
        except ValueError:
            print("Ошибка: попробуйте снова.")

x_input = get_input_values()
y_output = [takagi_sugeno(x) for x in x_input]

print("\nЗначения функции y(x):")
for xi, yi in zip(x_input, y_output):
    print(f"y({xi}) = {yi}")

# --- Построение графиков ---
x_range = np.linspace(min(min(p) for p in params), max(max(p) for p in params), 400)

membership_values = [[mu[i](x) for x in x_range] for i in range(5)]
ts_values = [takagi_sugeno(x) for x in x_range]

# --- Визуализация ---
plt.style.use('seaborn-v0_8-darkgrid')

fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

# Графики пользовательских функций
for i, f in enumerate(fs):
    x_local = np.linspace(min(params[i]), max(params[i]), 200)
    axs[0].plot(x_local, f(x_local), label=f"f{i+1}(x)", linewidth=2)

axs[0].set_title("Линейные функции пользователя", fontsize=14)
axs[0].set_ylabel("Значения", fontsize=12)
axs[0].legend()
axs[0].grid(True)

# Функции принадлежности
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for i in range(5):
    axs[1].plot(x_range, membership_values[i], label=f"μA{i+1}", color=colors[i], linestyle='--')

axs[1].set_title(f"Функции принадлежности ({mf_type})", fontsize=14)
axs[1].set_xlabel("x", fontsize=12)
axs[1].set_ylabel("μ(x)", fontsize=12)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# --- График TS модели ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

# Функция Такаги-Сугено
ax1.plot(x_range, ts_values, label="y(x) (TS)", color="crimson", linewidth=2.5)
ax1.set_title("Модель Такаги-Сугено", fontsize=14)
ax1.set_ylabel("y(x)", fontsize=12)
ax1.grid(True)
ax1.legend()

# Повтор функций принадлежности
for i in range(5):
    ax2.plot(x_range, membership_values[i], label=f"μA{i+1}", color=colors[i], linestyle='--')

ax2.set_title(f"Функции принадлежности ({mf_type})", fontsize=14)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("μ(x)", fontsize=12)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
