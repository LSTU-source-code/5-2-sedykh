import numpy as np
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import skfuzzy as fuzz

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FuzzyPlanner:
    def __init__(self):
        self.days = np.arange(0, 31, 1)
        self.volume = np.arange(0, 101, 1)
        self.anxiety = np.arange(0, 11, 0.1)

        self.days_low = fuzz.trimf(self.days, [0, 0, 10])
        self.days_medium = fuzz.trimf(self.days, [5, 15, 25])
        self.days_high = fuzz.trimf(self.days, [20, 30, 30])

        self.volume_small = fuzz.trimf(self.volume, [0, 0, 30])
        self.volume_medium = fuzz.trimf(self.volume, [20, 50, 80])
        self.volume_large = fuzz.trimf(self.volume, [60, 100, 100])

        self.anxiety_low = fuzz.trimf(self.anxiety, [0, 0, 3])
        self.anxiety_medium = fuzz.trimf(self.anxiety, [2, 5, 8])
        self.anxiety_high = fuzz.trimf(self.anxiety, [6, 10, 10])

    def fuzzify(self, days, volume, anxiety):
        d_low = fuzz.interp_membership(self.days, self.days_low, days)
        d_med = fuzz.interp_membership(self.days, self.days_medium, days)
        d_high = fuzz.interp_membership(self.days, self.days_high, days)

        v_small = fuzz.interp_membership(self.volume, self.volume_small, volume)
        v_med = fuzz.interp_membership(self.volume, self.volume_medium, volume)
        v_large = fuzz.interp_membership(self.volume, self.volume_large, volume)

        a_low = fuzz.interp_membership(self.anxiety, self.anxiety_low, anxiety)
        a_med = fuzz.interp_membership(self.anxiety, self.anxiety_medium, anxiety)
        a_high = fuzz.interp_membership(self.anxiety, self.anxiety_high, anxiety)

        return {
            'days': {'low': d_low, 'medium': d_med, 'high': d_high},
            'volume': {'small': v_small, 'medium': v_med, 'large': v_large},
            'anxiety': {'low': a_low, 'medium': a_med, 'high': a_high}
        }


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Экспертная система подготовки к экзаменам")
        self.geometry("1000x700")
        self.fuzzy_logic = FuzzyPlanner()

        self.mode = ctk.StringVar(value="fuzzy")

        self.setup_ui()

    def setup_ui(self):
        ctk.CTkLabel(self, text="Система планирования подготовки к экзаменам", font=("Segoe UI", 22)).pack(pady=20)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.mode_switch = ctk.CTkSegmentedButton(self.main_frame, values=["crisp", "fuzzy"], variable=self.mode)
        self.mode_switch.pack(pady=15)

        self.days_input = ctk.CTkSlider(self.main_frame, from_=0, to=30, number_of_steps=30)
        self.volume_input = ctk.CTkSlider(self.main_frame, from_=0, to=100, number_of_steps=100)
        self.anxiety_input = ctk.CTkSlider(self.main_frame, from_=0, to=10, number_of_steps=100)

        self.exam_count = ctk.CTkOptionMenu(self.main_frame, values=["1", "2", "3", "4", "5"])
        self.prep_level = ctk.CTkOptionMenu(self.main_frame, values=["низкий", "средний", "высокий"])

        self._make_labeled_slider("Дней до экзамена", self.days_input)
        self._make_labeled_slider("Объём материала (часов)", self.volume_input)
        self._make_labeled_slider("Уровень тревожности", self.anxiety_input)

        self._make_labeled_dropdown("Количество экзаменов", self.exam_count)
        self._make_labeled_dropdown("Уровень подготовки", self.prep_level)

        ctk.CTkButton(self.main_frame, text="Получить рекомендации", command=self.get_recommendations).pack(pady=20)

        self.result_text = ctk.CTkTextbox(self.main_frame, height=150, font=("Segoe UI", 14))
        self.result_text.pack(fill="x", padx=20, pady=10)

        self.plot_frame = ctk.CTkFrame(self.main_frame)
        self.plot_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def _make_labeled_slider(self, label, slider):
        ctk.CTkLabel(self.main_frame, text=label, anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        slider.pack(padx=20, pady=5, fill="x")

    def _make_labeled_dropdown(self, label, dropdown):
        ctk.CTkLabel(self.main_frame, text=label, anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        dropdown.pack(padx=20, pady=5, fill="x")

    def get_recommendations(self):
        mode = self.mode.get()
        days = self.days_input.get()
        volume = self.volume_input.get()
        anxiety = self.anxiety_input.get()
        exams = int(self.exam_count.get())
        prep = self.prep_level.get()

        if mode == "crisp":
            self.show_crisp_result(days, volume, exams, prep)
        else:
            self.show_fuzzy_result(days, volume, anxiety)

    def show_crisp_result(self, days, volume, exams, prep):
        recommendation = ""

        if days < 5 or volume > 70:
            recommendation += "🧠 Срочно начни подготовку! Учить по 6+ часов в день.\n"
        elif prep == "низкий":
            recommendation += "📚 Составь расписание и начни с самых сложных тем.\n"
        elif prep == "высокий" and days > 10:
            recommendation += "✅ Можешь распределить материал равномерно по 2–3 часа в день.\n"
        else:
            recommendation += "⚖️ Учи по 4 часа в день, делай перерывы каждые 60 минут.\n"

        if exams >= 3:
            recommendation += "📝 Удели внимание приоритизации тем для каждого экзамена.\n"

        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", recommendation)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    def show_fuzzy_result(self, days, volume, anxiety):
        fuzzy_values = self.fuzzy_logic.fuzzify(days, volume, anxiety)
        days_lvl = max(fuzzy_values['days'], key=fuzzy_values['days'].get)
        volume_lvl = max(fuzzy_values['volume'], key=fuzzy_values['volume'].get)
        anxiety_lvl = max(fuzzy_values['anxiety'], key=fuzzy_values['anxiety'].get)

        message = f"📊 Дней до экзамена: {days_lvl} ({fuzzy_values['days'][days_lvl]:.2f})\n"
        message += f"📊 Объём материала: {volume_lvl} ({fuzzy_values['volume'][volume_lvl]:.2f})\n"
        message += f"📊 Тревожность: {anxiety_lvl} ({fuzzy_values['anxiety'][anxiety_lvl]:.2f})\n\n"

        # Генерация рекомендаций
        if days_lvl == "low" and volume_lvl == "large":
            message += "🚨 Учти: мало времени и большой объём. Учить 6+ часов/день.\n"
        elif anxiety_lvl == "high":
            message += "🧘 Уровень тревожности высок — включи отдых и снизь нагрузку на вечер.\n"
        else:
            message += "📅 Составь умеренное расписание с фокусом на ключевые темы.\n"

        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", message)

        # Визуализация
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(7, 2))
        ax.plot(self.fuzzy_logic.days, self.fuzzy_logic.days_low, label="Мало")
        ax.plot(self.fuzzy_logic.days, self.fuzzy_logic.days_medium, label="Средне")
        ax.plot(self.fuzzy_logic.days, self.fuzzy_logic.days_high, label="Много")
        ax.axvline(days, color='black', linestyle='--', label=f"Выбор: {days:.1f} дн.")
        ax.set_title("Фаззификация дней до экзамена")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()
