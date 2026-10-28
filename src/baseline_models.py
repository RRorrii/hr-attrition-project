import pandas as pd
import numpy as np
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Настройка путей
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    # Замер времени обучения
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Замер времени инференса (предсказания)
    start_pred = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start_pred

    # Расчет метрик
    metrics = {
        "Модель": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
        "Время обуч. (сек)": train_time,
        "Время инференса (сек)": pred_time
    }
    return metrics


def main():
    print("1. Загрузка данных...")
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()

    results = []

    # 2. Логистическая регрессия (Baseline)
    print("2. Обучение Логистической регрессии (Baseline)...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    results.append(evaluate_model("Logistic Regression", lr_model, X_train, y_train, X_test, y_test))

    # 3. Случайный лес (Классическая модель для табличных данных)
    print("3. Обучение Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    results.append(evaluate_model("Random Forest", rf_model, X_train, y_train, X_test, y_test))

    # 4. Вывод сравнения
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ (включая нашу Нейросеть)")
    print("=" * 80)

    # Добавляем данные нашей нейросети из предыдущего этапа
    nn_results = {
        "Модель": "Neural Network (PyTorch)",
        "Accuracy": 0.8673,
        "Precision": 0.6538,
        "Recall": 0.3617,
        "F1-Score": 0.4658,
        "ROC-AUC": 0.7730,
        "Время обуч. (сек)": "~2.5 (с Early Stop)",
        "Время инференса (сек)": "<0.01"
    }
    results.append(nn_results)

    df_results = pd.DataFrame(results)

    # Красивый вывод таблицы
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.to_string(index=False))
    print("=" * 80)

    # 5. Анализ влияния признаков (Интерпретация)
    print("\nТОП-10 САМЫХ ВАЖНЫХ ПРИЗНАКОВ (по версии Random Forest):")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(10):
        print(f"{i + 1}. {X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")


if __name__ == "__main__":
    main()