def ask_question(factor):
    """Задаёт вопрос пользователю в зависимости от фактора"""
    questions = {
        1: "Сколько дней осталось до экзамена? (целое число)",
        2: "Оцените свой уровень знаний по шкале от 1 до 10:",
        3: "Сколько часов в день вы тратите на подготовку? (число)",
        4: "Есть ли у вас сложные темы? (да/нет)",
        5: "Какова ваша самодисциплина? (высокая/средняя/низкая)",
        6: "Сколько часов в день вы тратите на отвлекающие факторы? (число)",
        7: "Насколько эффективны ваши текущие методы подготовки? (высокая/средняя/низкая)",
        8: "Оцените уровень стресса от 1 до 10:",
        9: "Доступны ли вам учебные ресурсы? (да/нет)",
        10: "Какой у вас график занятий? (загруженный/свободный)",
        11: "Есть ли поддержка от окружения? (да/нет)"
    }
    return input(questions[factor] + "\n> ").strip().lower()

def validate_input(factor, raw_input):
    """Проверяет и преобразует ввод пользователя"""
    try:
        if factor in [1, 2, 3, 6, 8]:  # Числовые значения
            value = int(raw_input)
            if factor == 2 and not (1 <= value <= 10):
                raise ValueError("Значение должно быть от 1 до 10")
            return value
        elif factor in [4, 5, 7, 9, 10, 11]:  # Бинарные/категориальные значения
            if raw_input not in ['да', 'нет', 'высокая', 'средняя', 'низкая', 'загруженный', 'свободный']:
                raise ValueError("Недопустимое значение")
            return raw_input
        else:
            raise ValueError("Неизвестный фактор")
    except ValueError as e:
        print(f"Ошибка ввода: {e}. Попробуйте снова.")
        return validate_input(factor, ask_question(factor))

def collect_data():
    """Собирает данные от пользователя"""
    data = {}
    for factor in range(1, 12):
        data[factor] = validate_input(factor, ask_question(factor))
    return data

# Структура правил прямой цепочки (fact -> [new_facts])
rules = [
    # Уровень 2: Промежуточные причины
    {
        "type": "single",
        "factor": "factor_3",
        "condition": lambda x: x < 2,
        "result": "low_study_time"
    },
    {
        "type": "single",
        "factor": "factor_5",
        "condition": lambda x: x == "низкая",
        "result": "low_discipline"
    },
    {
        "type": "single",
        "factor": "factor_6",
        "condition": lambda x: x > 3,
        "result": "high_distractions"
    },
    {
        "type": "single",
        "factor": "factor_2",
        "condition": lambda x: x < 5,
        "result": "low_knowledge"
    },
    {
        "type": "single",
        "factor": "factor_4",
        "condition": lambda x: x == "да",
        "result": "difficult_topics"
    },
    {
        "type": "single",
        "factor": "factor_7",
        "condition": lambda x: x == "низкая",
        "result": "inefficient_methods"
    },
    {
        "type": "single",
        "factor": "factor_1",
        "condition": lambda x: x < 7,
        "result": "limited_time"
    },
    {
        "type": "single",
        "factor": "factor_9",
        "condition": lambda x: x == "нет",
        "result": "no_resources"
    },
    {
        "type": "single",
        "factor": "factor_11",
        "condition": lambda x: x == "нет",
        "result": "no_support"
    },
    
    # Уровень 3: Комбинации причин
    {
        "type": "combination",
        "factors": ["low_study_time", "low_discipline"],
        "result": "plan_needed"
    },
    {
        "type": "combination",
        "factors": ["difficult_topics", "inefficient_methods"],
        "result": "active_review_needed"
    },
    {
        "type": "combination",
        "factors": ["limited_time", "low_knowledge"],
        "result": "increase_time_needed"
    },
    {
        "type": "combination",
        "factors": ["difficult_topics", "no_resources", "no_support"],
        "result": "help_needed"
    },
    
    # Уровень 4: Рекомендации
    {
        "type": "recommendation",
        "source": "plan_needed",
        "text": "Создать детальный план подготовки"
    },
    {
        "type": "recommendation",
        "source": "active_review_needed",
        "text": "Использовать методы активного повторения"
    },
    {
        "type": "recommendation",
        "source": "increase_time_needed",
        "text": "Увеличить время подготовки"
    },
    {
        "type": "recommendation",
        "source": "help_needed",
        "text": "Обратиться за помощью"
    }
]

def apply_rules(facts):
    """Применяет правила прямой цепочки для вывода новых фактов"""
    changed = True
    while changed:
        changed = False
        # Обрабатываем одиночные условия
        for rule in rules:
            if rule["type"] == "single":
                factor_value = facts.get(rule["factor"], None)
                if factor_value is not None and rule["condition"](factor_value):
                    if rule["result"] not in facts:
                        facts[rule["result"]] = True
                        changed = True
            elif rule["type"] == "combination":
                if all(fact in facts for fact in rule["factors"]) and rule["result"] not in facts:
                    facts[rule["result"]] = True
                    changed = True
    return facts

def get_recommendations(facts):
    """Извлекает финальные рекомендации из фактов"""
    recommendations = []
    for rule in rules:
        if rule["type"] == "recommendation" and rule["source"] in facts:
            recommendations.append(rule["text"])
    return recommendations

def main():
    """Основная функция программы"""
    print("Добро пожаловать в экспертную систему планирования подготовки к экзаменам!")
    print("Пожалуйста, ответьте на следующие вопросы.\n")
    
    # Сбор данных
    raw_data = collect_data()
    
    # Преобразование данных в формат фактов
    facts = {f"factor_{k}": v for k, v in raw_data.items()}
    
    # Применение правил прямой цепочки
    updated_facts = apply_rules(facts)
    
    # Получение рекомендаций
    recommendations = get_recommendations(updated_facts)

    # Вывод результата
    print("\nРекомендации:")
    if recommendations:
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("На основе ваших ответов нет подходящих рекомендаций.")

if __name__ == "__main__":
    main()