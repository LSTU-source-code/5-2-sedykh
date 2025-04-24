import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# --- Функции принадлежности --- 
def trapezoid_mf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)

def triangle_mf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def is_linear_function(expr, var):
    return expr.is_polynomial(var) and expr.as_poly(var).degree() <= 1

class FuzzyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Модель Такаги-Сугено")
        self.geometry("800x700")
        self.inputs_frame = tk.Frame(self)
        self.inputs_frame.pack(pady=10)

        self.fp_type = tk.StringVar(value="Треугольная")
        tk.Label(self.inputs_frame, text="Тип функции принадлежности:").grid(row=0, column=0, sticky="w")
        fp_type_menu = tk.OptionMenu(self.inputs_frame, self.fp_type, "Треугольная", "Трапециевидная", command=self.update_input_fields)
        fp_type_menu.grid(row=0, column=1, sticky="w")

        self.param_entries = []
        for i in range(5):
            tk.Label(self.inputs_frame, text=f"A{i+1} параметры:").grid(row=i+1, column=0, sticky="w")
            entry = tk.Entry(self.inputs_frame, width=30)
            entry.grid(row=i+1, column=1, sticky="w")
            self.param_entries.append(entry)

        self.func_entries = []
        for i in range(5):
            tk.Label(self.inputs_frame, text=f"f{i+1}(x):").grid(row=i+1, column=2, sticky="w")
            entry = tk.Entry(self.inputs_frame, width=30)
            entry.grid(row=i+1, column=3, sticky="w")
            self.func_entries.append(entry)

        tk.Label(self.inputs_frame, text="Значения x (через пробел):").grid(row=6, column=0, sticky="w")
        self.x_entry = tk.Entry(self.inputs_frame, width=60)
        self.x_entry.grid(row=6, column=1, columnspan=3, sticky="w")

        self.calc_btn = tk.Button(self, text="Рассчитать и построить графики", command=self.run)
        self.calc_btn.pack(pady=10)

        self.set_default_values()

    def set_default_values(self):
        triangle_defaults = [
            [0, 1, 2],  
            [1, 2, 3],  
            [2, 3, 4],  
            [3, 4, 5],  
            [4, 5, 6]
        ]
        
        trapezoid_defaults = [
            [0, 1, 2, 3],  
            [1, 2, 3, 4],  
            [2, 3, 4, 5],  
            [3, 4, 5, 6],  
            [4, 5, 6, 7]   
        ]

        function_defaults = [
            "x",           
            "2*x + 1",     
            "0.5 * x - 2", 
            "-x + 3",      
            "3 * x"        
        ]

        x_defaults = "0 1 2 3 4 5"

        if self.fp_type.get() == "Треугольная":
            for i, entry in enumerate(self.param_entries):
                entry.delete(0, tk.END)
                entry.insert(0, ' '.join(map(str, triangle_defaults[i])))
        else:
            for i, entry in enumerate(self.param_entries):
                entry.delete(0, tk.END)
                entry.insert(0, ' '.join(map(str, trapezoid_defaults[i])))

        for i, entry in enumerate(self.func_entries):
            entry.delete(0, tk.END)
            entry.insert(0, function_defaults[i])

        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, x_defaults)

    def update_input_fields(self, *args):
        """
        Обновляет количество полей для ввода параметров в зависимости от выбранного типа функции принадлежности.
        """
        # Очищаем старые поля ввода параметров
        for entry in self.param_entries:
            entry.grid_forget()

        # Очищаем старые значения параметров
        self.param_entries = []

        # Обновляем количество полей для ввода в зависимости от типа функции
        if self.fp_type.get() == "Треугольная":
            for i in range(5):
                tk.Label(self.inputs_frame, text=f"A{i+1} параметры:").grid(row=i+1, column=0, sticky="w")
                entry = tk.Entry(self.inputs_frame, width=30)
                entry.grid(row=i+1, column=1, sticky="w")
                self.param_entries.append(entry)
        elif self.fp_type.get() == "Трапециевидная":
            for i in range(5):
                tk.Label(self.inputs_frame, text=f"A{i+1} параметры:").grid(row=i+1, column=0, sticky="w")
                entry = tk.Entry(self.inputs_frame, width=30)
                entry.grid(row=i+1, column=1, sticky="w")
                self.param_entries.append(entry)

        # Устанавливаем значения по умолчанию для новых полей
        self.set_default_values()

    def run(self):
        try:
            x = sp.symbols('x')

            params = []
            for entry in self.param_entries:
                values = list(map(float, entry.get().split()))
                expected_len = 4 if self.fp_type.get() == "Трапециевидная" else 3
                if len(values) != expected_len:
                    raise ValueError(f"Неверное количество параметров: {values}")
                params.append(values)

            functions = []
            for entry in self.func_entries:
                expr = sp.sympify(entry.get())
                if not is_linear_function(expr, x):
                    raise ValueError(f"Функция не линейна: {expr}")
                functions.append(sp.lambdify(x, expr, 'numpy'))

            x_vals = list(map(float, self.x_entry.get().split()))

            if self.fp_type.get() == "Треугольная":
                mus = [lambda x, p=p: triangle_mf(x, *p) for p in params]
            else:
                mus = [lambda x, p=p: trapezoid_mf(x, *p) for p in params]

            def y_ts(x_val):
                return sum(m(x_val) * f(x_val) for m, f in zip(mus, functions))

            print("Значения y(x):")
            for xv in x_vals:
                print(f"y({xv}) = {y_ts(xv)}")

            intervals = [(min(p), max(p)) for p in params]
            x_min, x_max = min(i[0] for i in intervals), max(i[1] for i in intervals)
            xs = np.linspace(x_min, x_max, 300)

            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            for i, (f, intv) in enumerate(zip(functions, intervals)):
                x_plot = np.linspace(intv[0], intv[1], 100)
                axs[0].plot(x_plot, f(x_plot), label=f"f{i+1}(x)")
            axs[0].set_title("Линейные функции")
            axs[0].legend()
            axs[0].grid(True)

            for i, m in enumerate(mus):
                axs[1].plot(xs, [m(x) for x in xs], label=f"mu A{i+1}")
            axs[1].set_title("Функции принадлежности")
            axs[1].set_ylim(-0.05, 1.05)
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.show()

            y_vals = [y_ts(xi) for xi in xs]
            plt.figure(figsize=(10, 5))
            plt.plot(xs, y_vals, 'r-', label="y(x) — TS модель")
            plt.title("Результирующая функция Такаги-Сугено")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":
    app = FuzzyApp()
    app.mainloop()
