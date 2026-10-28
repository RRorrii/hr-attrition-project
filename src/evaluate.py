import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import AttritionNN

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
models_dir = os.path.join(project_root, 'models')


def main():
    print("1. Загрузка тестовых данных и модели...")
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv')).values.astype(np.float32)
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.astype(np.float32)

    X_test_tensor = torch.tensor(X_test)

    model = AttritionNN(input_size=X_test.shape[1])
    model_path = os.path.join(models_dir, 'attrition_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Режим оценки

    print("2. Получение предсказаний...")
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor).numpy().flatten()

    # Превращаем вероятности в классы (0 или 1)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    y_true = y_test.astype(int).flatten()

    print("\n3. Расчет метрик качества:")
    print("-" * 30)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred) * 100:.2f}%")
    print(f"Recall:    {recall_score(y_true, y_pred) * 100:.2f}%")
    print(f"F1-Score:  {f1_score(y_true, y_pred) * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_pred_probs) * 100:.2f}%")
    print("-" * 30)

    # --- ПОСТРОЕНИЕ ГРАФИКОВ ---
    print("\n4. Построение графиков (Confusion Matrix и ROC-кривая)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Матрица ошибок (Confusion Matrix)')
    axes[0].set_xlabel('Предсказано (0: Остался, 1: Уволился)')
    axes[0].set_ylabel('Реально (0: Остался, 1: Уволился)')
    axes[0].set_xticklabels(['Остался', 'Уволился'])
    axes[0].set_yticklabels(['Остался', 'Уволился'])

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred_probs):.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC-кривая')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    # Сохраняем график в папку reports (или просто показываем)
    plt.show()


if __name__ == "__main__":
    main()