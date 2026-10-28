import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)


def main():
    print("1. Загрузка исходных данных...")
    # Формируем абсолютный путь до датасета
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df = pd.read_csv(raw_data_path)

    print("2. Удаление неинформативных признаков...")
    # Удаляем константы и ID
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=cols_to_drop)

    print("3. Кодирование целевой переменной...")
    # Yes -> 1, No -> 0
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

    print("4. Кодирование категориальных признаков (One-Hot Encoding)...")
    # Превращаем текст в 0 и 1
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("5. Разделение на признаки (X) и целевую переменную (y)...")
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    print("6. Разделение на Train и Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("7. Масштабирование (StandardScaler)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("8. Сохранение подготовленных данных в data/processed/...")
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    X_train_scaled.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)

    print("\n=========================================")
    print("ГОТОВО! Данные подготовлены.")
    print(f"Размер обучающей выборки: {X_train_scaled.shape} (строк, признаков)")
    print(f"Размер тестовой выборки: {X_test_scaled.shape} (строк, признаков)")
    print(f"Всего признаков после кодирования: {X_train_scaled.shape[1]}")
    print("=========================================")


if __name__ == "__main__":
    main()