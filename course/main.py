import numpy as np
import skfuzzy as fuzz
import customtkinter as ctk
import tkinter.messagebox as messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Настройки шрифтов
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

LARGE_FONT = ('Segoe UI', 14)
TITLE_FONT = ('Segoe UI', 16, 'bold')
BUTTON_FONT = ('Segoe UI', 14, 'bold')

GRAPH_FONT = {'font.family': 'Segoe UI', 'font.size': 14}
plt.rcParams.update(GRAPH_FONT)


class FuzzyLogic:
    def __init__(self):
        self.hours = np.arange(0, 12.1, 0.1)

        # Треугольные функции
        self.tri_low = fuzz.trimf(self.hours, [0, 0, 4])
        self.tri_medium = fuzz.trimf(self.hours, [2, 5, 8])
        self.tri_high = fuzz.trimf(self.hours, [6, 10, 12])

        # Трапецеидальные
        self.trap_low = fuzz.trapmf(self.hours, [0, 0, 2, 4])
        self.trap_medium = fuzz.trapmf(self.hours, [3, 5, 6, 8])
        self.trap_high = fuzz.trapmf(self.hours, [7, 9, 10, 12])

        # Гауссовские
        self.gauss_low = fuzz.gaussmf(self.hours, 2, 1)
        self.gauss_medium = fuzz.gaussmf(self.hours, 5, 1.2)
        self.gauss_high = fuzz.gaussmf(self.hours, 9, 1.5)

        # Обобщенные гауссовские
        self.gbell_low = fuzz.gbellmf(self.hours, 1.5, 2, 2)
        self.gbell_medium = fuzz.gbellmf(self.hours, 2, 2, 5)
        self.gbell_high = fuzz.gbellmf(self.hours, 2, 2, 9)

    def fuzzify_hours(self, value, method='triangular'):
        if method == 'triangular':
            low = fuzz.interp_membership(self.hours, self.tri_low, value)
            medium = fuzz.interp_membership(self.hours, self.tri_medium, value)
            high = fuzz.interp_membership(self.hours, self.tri_high, value)
            funcs = [self.tri_low, self.tri_medium, self.tri_high]
        elif method == 'trapezoidal':
            low = fuzz.interp_membership(self.hours, self.trap_low, value)
            medium = fuzz.interp_membership(self.hours, self.trap_medium, value)
            high = fuzz.interp_membership(self.hours, self.trap_high, value)
            funcs = [self.trap_low, self.trap_medium, self.trap_high]
        elif method == 'gaussian':
            low = fuzz.interp_membership(self.hours, self.gauss_low, value)
            medium = fuzz.interp_membership(self.hours, self.gauss_medium, value)
            high = fuzz.interp_membership(self.hours, self.gauss_high, value)
            funcs = [self.gauss_low, self.gauss_medium, self.gauss_high]
        elif method == 'gbell':
            low = fuzz.interp_membership(self.hours, self.gbell_low, value)
            medium = fuzz.interp_membership(self.hours, self.gbell_medium, value)
            high = fuzz.interp_membership(self.hours, self.gbell_high, value)
            funcs = [self.gbell_low, self.gbell_medium, self.gbell_high]
        elif method == 'all':
            return self.fuzzify_hours(value, 'triangular')
        else:
            raise ValueError("Неизвестный метод фаззификации.")

        return {
            'low': low,
            'medium': medium,
            'high': high,
            'funcs': funcs
        }

    def defuzzify(self, aggregated, method='centroid'):
        if method not in ['centroid', 'bisector', 'mom']:
            raise ValueError("Метод дефаззификации не распознан.")
        return fuzz.defuzz(self.hours, aggregated, method)


class KnowledgeBase:
    def __init__(self):
        self.rules = [
            # Исходные правила
            {'conditions': {'уровень знаний': 'низкий', 'оставшееся время': '< 1 дня'}, 'recommendation': 'Паниковое повторение основ'},
            {'conditions': {'уровень знаний': 'средний', 'оставшееся время': '4-7 дней', 'мотивация': 'высокая'}, 'recommendation': 'Создайте подробный план подготовки'},
            {'conditions': {'усталость': 'да', 'концентрация': 'низкая'}, 'recommendation': 'Сделайте перерыв на 1 день, отдохните'},
            {'conditions': {'тип экзамена': 'тестирование', 'уровень знаний': 'высокий'}, 'recommendation': 'Тренируйтесь на онлайн-тестах'},
            {'conditions': {'оставшееся время': '> недели', 'мотивация': 'низкая'}, 'recommendation': 'Установите ежедневные цели'},
            {'conditions': {'оставшееся время': '1-3 дня', 'качество сна': 'плохое'}, 'recommendation': 'Нормализуйте режим сна'},
            {'conditions': {'концентрация': 'низкая', 'мотивация': 'низкая'}, 'recommendation': 'Измените среду обучения или место'},
            {'conditions': {'уровень знаний': 'высокий', 'оставшееся время': '> недели'}, 'recommendation': 'Продолжайте систематически повторять'},
            {'conditions': {'оставшееся время': '1-3 дня', 'усталость': 'нет'}, 'recommendation': 'Финальный интенсивный повтор'},
            {'conditions': {'оставшееся время': '4-7 дней', 'усталость': 'нет', 'мотивация': 'высокая'}, 'recommendation': 'Работайте по 2-3 часа с перерывами'},
            {'conditions': {'уровень знаний': 'низкий', 'мотивация': 'низкая'}, 'recommendation': 'Начните с базовых тем и простых задач'},
            {'conditions': {'уровень знаний': 'средний', 'тип экзамена': 'билеты'}, 'recommendation': 'Подготовьте конспекты по билетам'},
            {'conditions': {'усталость': 'да', 'качество сна': 'плохое'}, 'recommendation': 'Пересмотрите свой график сна'},
            {'conditions': {'концентрация': 'высокая', 'оставшееся время': '> недели'}, 'recommendation': 'Создайте расписание с приоритетами'},
            {'conditions': {'мотивация': 'высокая', 'концентрация': 'средняя'}, 'recommendation': 'Используйте таймер Помодоро'},
            {'conditions': {'оставшееся время': '1-3 дня', 'тип экзамена': 'задачи'}, 'recommendation': 'Решайте примеры под таймер'},
            {'conditions': {'уровень знаний': 'высокий', 'усталость': 'да'}, 'recommendation': 'Сделайте короткий отдых перед финалом'},
            {'conditions': {'качество сна': 'плохое', 'мотивация': 'низкая'}, 'recommendation': 'Улучшите условия сна и употребляйте меньше кофе'},
            {'conditions': {'концентрация': 'низкая', 'типо экзамена': 'тестирование'}, 'recommendation': 'Попробуйте тесты с ограничением времени'},
            {'conditions': {'оставшееся время': '> недели', 'усталость': 'нет'}, 'recommendation': 'Постоянно тренируйтесь по одному разделу в день'},
            {'conditions': {'уровень знаний': 'средний', 'качество сна': 'хорошее'}, 'recommendation': 'Уделяйте больше внимания слабым местам'},
            {'conditions': {'мотивация': 'высокая', 'типо экзамена': 'билеты'}, 'recommendation': 'Запишите ответы на видео для лучшего запоминания'},
            {'conditions': {'усталость': 'нет', 'концентрация': 'высокая'}, 'recommendation': 'Максимально используйте этот период продуктивности'},
            {'conditions': {'оставшееся время': '4-7 дней', 'качество сна': 'плохое'}, 'recommendation': 'Восстановите силы перед решающей фазой'},
            {'conditions': {'уровень знаний': 'низкий', 'типо экзамена': 'задачи'}, 'recommendation': 'Повторяйте формулы и типовые решения'},
            {'conditions': {'мотивация': 'средняя', 'типо экзамена': 'тестирование'}, 'recommendation': 'Проходите тесты раз в два дня'},
            {'conditions': {'концентрация': 'средняя', 'оставшееся время': '1-3 дня'}, 'recommendation': 'Фокусируйтесь на самых важных темах'},
            {'conditions': {'усталость': 'да', 'типо экзамена': 'билеты'}, 'recommendation': 'Отдыхайте, но держите билеты под рукой'},
            {'conditions': {'качество сна': 'хорошее', 'уровень знаний': 'средний'}, 'recommendation': 'Добавьте практику к теории'},
            {'conditions': {'оставшееся время': '> недели', 'мотивация': 'средняя'}, 'recommendation': 'Постепенно увеличивайте нагрузку'},
            {'conditions': {'уровень знаний': 'высокий', 'мотивация': 'средняя'}, 'recommendation': 'Оставайтесь в ритме без выгорания'}
        ]

    def get_rules(self):
        return self.rules


class InferenceEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def forward_chaining(self, symptoms):
        matched_recs = []
        for rule in self.knowledge_base.get_rules():
            match_count = sum(1 for k, v in rule['conditions'].items() if symptoms.get(k) == v)
            if match_count >= 2:
                matched_recs.append(rule['recommendation'])
        return matched_recs

    def backward_chaining(self, hypothesis, symptoms):
        for rule in self.knowledge_base.get_rules():
            if rule['recommendation'].lower() == hypothesis.lower():
                matched = {}
                unmatched = {}
                for k, v in rule['conditions'].items():
                    if symptoms.get(k) == v:
                        matched[k] = v
                    else:
                        unmatched[k] = (v, symptoms.get(k))
                return matched, unmatched
        return None, None


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🎓 Планирование времени для экзаменов")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        self.fuzzy = FuzzyLogic()
        self.kb = KnowledgeBase()
        self.engine = InferenceEngine(self.kb)

        self.create_widgets()

    def create_widgets(self):
        tab_control = ctk.CTkTabview(self, width=1200, height=800)
        tab_control.pack(padx=10, pady=10, expand=True, fill="both")

        tab_crisp = tab_control.add("Чёткая логика")
        tab_fuzzy = tab_control.add("Нечёткая логика")

        self.build_crisp_tab(tab_crisp)
        self.build_fuzzy_tab(tab_fuzzy)

    def build_crisp_tab(self, frame):
        frame.grid_columnconfigure(0, minsize=150)
        frame.grid_columnconfigure(1, weight=1)

        self.symptoms_vars = {
            'уровень знаний': ctk.StringVar(value='средний'),
            'оставшееся время': ctk.StringVar(value='4-7 дней'),
            'тип экзамена': ctk.StringVar(value='билеты'),
            'мотивация': ctk.StringVar(value='средняя'),
            'усталость': ctk.StringVar(value='нет'),
            'концентрация': ctk.StringVar(value='средняя'),
            'качество сна': ctk.StringVar(value='хорошее')
        }

        options_for = {
            'уровень знаний': ['низкий', 'средний', 'высокий'],
            'оставшееся время': ['< 1 дня', '1-3 дня', '4-7 дней', '> недели'],
            'тип экзамена': ['тестирование', 'билеты', 'задачи'],
            'мотивация': ['низкая', 'средняя', 'высокая'],
            'усталость': ['нет', 'да'],
            'концентрация': ['низкая', 'средняя', 'высокая'],
            'качество сна': ['плохое', 'хорошее']
        }

        row = 0
        for symptom, var in self.symptoms_vars.items():
            ctk.CTkLabel(frame, text=symptom, font=LARGE_FONT).grid(row=row, column=0, sticky='w', padx=15, pady=10)
            combo = ctk.CTkComboBox(frame, values=options_for[symptom], variable=var, width=200)
            combo.grid(row=row, column=1, padx=15, pady=10, sticky='ew')
            row += 1

        self.chain_var = ctk.StringVar(value='1')
        ctk.CTkLabel(frame, text="Метод вывода:", font=LARGE_FONT).grid(row=row, column=0, sticky='w', padx=15, pady=10)
        ctk.CTkLabel(frame, text="1 - Прямой анализ\n2 - Обратный поиск", font=LARGE_FONT).grid(
            row=row, column=1, sticky='w', padx=15, pady=5)
        row += 1
        ctk.CTkComboBox(frame, values=['1', '2'], variable=self.chain_var, width=200).grid(
            row=row, column=1, padx=15, pady=5, sticky='ew')

        row += 1
        self.hypothesis_var = ctk.StringVar()
        ctk.CTkLabel(frame, text="Гипотеза (для метода 2):", font=LARGE_FONT).grid(
            row=row, column=0, sticky='w', padx=15, pady=10)
        ctk.CTkEntry(frame, textvariable=self.hypothesis_var, width=200).grid(
            row=row, column=1, padx=15, pady=10, sticky='ew')

        row += 1
        ctk.CTkButton(frame, text="Получить рекомендации", command=self.run_crisp_mode, font=BUTTON_FONT).grid(
            row=row, column=0, columnspan=2, pady=20, padx=15, ipady=10, sticky='nsew')

    def build_fuzzy_tab(self, frame):
        frame.grid_columnconfigure(0, weight=1)

        self.hours_input = ctk.DoubleVar(value=6.0)
        self.defuzz_method = ctk.StringVar(value='centroid')
        self.fuzz_method = ctk.StringVar(value='all')

        ctk.CTkLabel(frame, text="Часы в день на подготовку:", font=LARGE_FONT).pack(pady=(20, 5))
        ctk.CTkEntry(frame, textvariable=self.hours_input, width=200).pack(pady=5)

        ctk.CTkLabel(frame, text="Метод фаззификации:", font=LARGE_FONT).pack(pady=(20, 5))
        ctk.CTkComboBox(frame, values=['all', 'triangular', 'trapezoidal', 'gaussian', 'gbell'],
                        variable=self.fuzz_method, width=200).pack(pady=5)

        ctk.CTkLabel(frame, text="Метод дефаззификации:", font=LARGE_FONT).pack(pady=(20, 5))
        ctk.CTkComboBox(frame, values=['centroid', 'bisector', 'mom'],
                        variable=self.defuzz_method, width=200).pack(pady=5)

        ctk.CTkButton(frame, text="Рассчитать", command=self.run_fuzzy_mode, font=BUTTON_FONT).pack(
            pady=30, ipady=10, ipadx=20)

        self.fig, self.ax = plt.subplots(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)

    def run_fuzzy_mode(self):
        try:
            value = float(self.hours_input.get())
            if value < 0 or value > 12:
                messagebox.showerror("❌ Ошибка", "Часы должны быть в диапазоне 0–12")
                return
        except Exception:
            messagebox.showerror("❌ Ошибка", "Введите корректное число для часов.")
            return

        method = self.defuzz_method.get()
        fuzz_method = self.fuzz_method.get()
        results = self.fuzzy.fuzzify_hours(value, fuzz_method)
        memberships = {k: results[k] for k in ['low', 'medium', 'high']}
        funcs = results['funcs']

        aggregated = np.fmax(np.fmax(
            memberships['low'] * funcs[0],
            memberships['medium'] * funcs[1]),
            memberships['high'] * funcs[2])

        crisp_value = self.fuzzy.defuzzify(aggregated, method)

        stress_based_rec = {
            'low': ['Хорошо! Добавьте немного практики.', 'Поддерживайте текущий ритм.'],
            'medium': ['Увеличьте отдых.', 'Делайте перерывы каждые 45 минут.'],
            'high': ['Уменьшите нагрузку.', 'Обратитесь к психологу.', 'Старайтесь спать минимум 7 часов.']
        }

        temp_diagnoses = []
        sorted_levels = sorted(memberships.items(), key=lambda x: x[1], reverse=True)
        for level, strength in sorted_levels:
            if strength > 0.2:
                for r in stress_based_rec.get(level, []):
                    temp_diagnoses.append(f"{r} ({strength:.2f})")

        max_term = max(memberships, key=memberships.get)
        max_value = memberships[max_term]

        self.ax.clear()
        self.ax.plot(self.fuzzy.hours, funcs[0], 'b', label='Низкий стресс')
        self.ax.plot(self.fuzzy.hours, funcs[1], 'g', label='Средний стресс')
        self.ax.plot(self.fuzzy.hours, funcs[2], 'r', label='Высокий стресс')
        self.ax.fill_between(self.fuzzy.hours, np.zeros_like(aggregated), aggregated, facecolor='y', alpha=0.1)
        self.ax.axvline(value, color='k', linestyle=':', label=f'Часы в день: {value:.2f}')
        self.ax.axvline(crisp_value, color='k', linestyle='--', label=f'Оценка стресса: {crisp_value:.2f}')

        self.ax.set_title(f"Степень стресса\nМакс. уровень: {max_term} ({max_value:.2f})", fontsize=14)
        self.ax.set_xlabel("Часы в день")
        self.ax.set_ylabel("Степень принадлежности")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

        membership_msg = "\n".join([f"{k}: {v:.2f}" for k, v in memberships.items()])
        diagnoses_msg = "Рекомендации по стрессу:\n" + "\n".join(temp_diagnoses) if temp_diagnoses else ""

        messagebox.showinfo("📊 Результат",
                            f"Степени стресса:\n{membership_msg}\n"
                            f"Максимальный уровень: {max_term} ({max_value:.2f})\n"
                            f"Дефаззифицированное значение: {crisp_value:.2f} ч.\n"
                            f"{diagnoses_msg}\n"
                            f"Метод фаззификации: {fuzz_method}, Метод дефаззификации: {method}")

    def run_crisp_mode(self):
        symptoms = {k: v.get() for k, v in self.symptoms_vars.items()}
        chain = self.chain_var.get()

        if chain == '1':
            recommendations = self.engine.forward_chaining(symptoms)
            if recommendations:
                messagebox.showinfo("🎓 Рекомендации", "Найденные рекомендации:\n" + "\n".join(recommendations))
            else:
                messagebox.showinfo("⚠️ Рекомендации", "Не найдено подходящих рекомендаций.")

        elif chain == '2':
            hypothesis = self.hypothesis_var.get().strip()
            if not hypothesis:
                messagebox.showerror("❌ Ошибка", "Введите гипотезу для метода 2.")
                return
            matched, unmatched = self.engine.backward_chaining(hypothesis, symptoms)
            if matched is None:
                messagebox.showinfo("🔍 Результат", f"Гипотеза '{hypothesis}' не найдена.")
                return

            msg = f"Гипотеза: {hypothesis}\n"
            msg += "Подтверждённые параметры:\n"
            for k, v in matched.items():
                msg += f"- {k}: {v}\n"
            if unmatched:
                msg += "\nНесовпадающие параметры:\n"
                for k, (expected, actual) in unmatched.items():
                    msg += f"- {k}: ожидалось {expected}, указано {actual}\n"
            else:
                msg += "\nВсе параметры соответствуют гипотезе."

            messagebox.showinfo("📊 Результат", msg)
        else:
            messagebox.showerror("❌ Ошибка", "Неверный метод вывода.")


if __name__ == "__main__":
    app = App()
    app.mainloop()