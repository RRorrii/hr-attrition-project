import torch
import torch.nn as nn


class AttritionNN(nn.Module):
    """
    Нейронная сеть для прогнозирования увольнения (бинарная классификация).
    Архитектура: 44 -> 64 -> 32 -> 1
    """

    def __init__(self, input_size):
        super(AttritionNN, self).__init__()

        # Первый скрытый слой
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)  # 30% нейронов отключаются для борьбы с переобучением

        # Второй скрытый слой
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)  # 20% нейронов отключаются

        # Выходной слой (1 нейрон для бинарной классификации)
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()  # Превращает число в вероятность от 0 до 1

    def forward(self, x):
        # Прямой проход (как данные идут через сеть)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x


# Блок для локальной проверки архитектуры
if __name__ == "__main__":
    print("Инициализация модели...")

    # В нашем датасете после подготовки осталось 44 признака
    input_features = 44
    model = AttritionNN(input_size=input_features)

    # Создаем "фейковые" данные (размер батча 5, 44 признака)
    # Это имитация того, как мы подадим 5 сотрудников на вход
    dummy_input = torch.randn(5, input_features)

    # Прогоняем через модель
    output = model(dummy_input)

    print("\n--- АРХИТЕКТУРА МОДЕЛИ ---")
    print(model)
    print("\n--- ТЕСТОВЫЙ ЗАПУСК ---")
    print("Размер входных данных (batch_size, features):", dummy_input.shape)
    print("Размер выходных данных (batch_size, 1):", output.shape)
    print("\nПример предсказанных вероятностей увольнения для 5 сотрудников:")
    # Округляем для красоты вывода
    print(torch.round(output.detach() * 100) / 100)