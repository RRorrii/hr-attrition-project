import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import sys
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import AttritionNN

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
models_dir = os.path.join(project_root, 'models')


def main():
    print("1. Загрузка подготовленных данных...")
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv')).values.astype(np.float32)
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.astype(np.float32)
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv')).values.astype(np.float32)
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.astype(np.float32)

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).view(-1, 1)

    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).view(-1, 1)

    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"2. Инициализация модели (входных признаков: {X_train.shape[1]})...")
    model = AttritionNN(input_size=X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100  # Увеличили максимум эпох, но Early Stopping остановит раньше
    print(f"3. Начинаем обучение (макс. {epochs} эпох, с Early Stopping)...")
    print("-" * 65)

    # --- ПАРАМЕТРЫ EARLY STOPPING ---
    best_test_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 10  # Сколько эпох ждать улучшения
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Проверяем на тесте каждую эпоху
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            predicted = (test_outputs >= 0.5).float()
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / y_test_tensor.size(0)

        train_loss_avg = epoch_loss / len(train_loader)

        # Логика Early Stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = "★ (Best)"
        else:
            patience_counter += 1
            marker = ""

        # Печатаем каждую 5-ю эпоху, чтобы не засорять консоль, или если это Best
        if (epoch + 1) % 5 == 0 or marker == "★ (Best)":
            print(
                f"Эпоха [{epoch + 1:03d}] | Train Loss: {train_loss_avg:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {accuracy * 100:.2f}% {marker}")

        if patience_counter >= patience:
            print(f"\n! Ранняя остановка на эпохе {epoch + 1}. Модель перестала улучшаться на тесте.")
            break

    print("-" * 65)
    print("4. Сохранение ЛУЧШЕЙ обученной модели...")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'attrition_model.pth')

    # Загружаем лучшие веса перед сохранением
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), model_save_path)
    print(f"Модель сохранена в: {model_save_path}")
    print(f"Лучший Test Loss: {best_test_loss:.4f}")


if __name__ == "__main__":
    main()