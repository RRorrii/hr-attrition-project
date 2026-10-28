import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import sys
import joblib
import io
from datetime import datetime

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from model import AttritionNN

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
models_dir = os.path.join(project_root, 'models')


@st.cache_resource
def load_model_and_scaler():
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    features = X_train.columns.tolist()
    model = AttritionNN(input_size=len(features))
    model.load_state_dict(torch.load(os.path.join(models_dir, 'attrition_model.pth')))
    model.eval()
    scaler = joblib.load(os.path.join(processed_dir, 'scaler.pkl'))
    return model, scaler, features


model, scaler, features = load_model_and_scaler()

st.title("👥 Система прогнозирования увольнений")
st.markdown("Аналитическая платформа для HR-специалистов (Прототип)")

tab1, tab2 = st.tabs(["🔮 Индивидуальный прогноз", "📂 Массовая загрузка (CSV)"])

with tab1:
    st.markdown(
        "Введите данные сотрудника, чтобы оценить риск его увольнения. Остальные параметры система берет как 'средние по компании'.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Основная информация")
        age = st.slider("Возраст (Age)", 18, 60, 30)
        overtime = st.selectbox("Переработки (OverTime)", ["No", "Yes"])
        job_level = st.slider("Уровень должности (JobLevel)", 1, 5, 2)
        monthly_income = st.number_input("Зарплата в месяц (MonthlyIncome)", 1000, 20000, 5000, step=500)
        years_at_company = st.slider("Стаж в компании (YearsAtCompany)", 0, 40, 3)

    with col2:
        st.subheader("Удовлетворенность и условия")
        env_satisfaction = st.slider("Удовлетворенность средой", 1, 4, 3)
        job_satisfaction = st.slider("Удовлетворенность работой", 1, 4, 3)
        work_life = st.slider("Баланс работы и жизни", 1, 4, 3)
        distance = st.slider("Расстояние до дома (км)", 1, 30, 5)

    if st.button("🔮 Рассчитать риск увольнения", type="primary"):
        # Берем средние значения из скалера, чтобы не обнулять стаж и опыт!
        input_df = pd.DataFrame([scaler.mean_], columns=features)

        # Подставляем введенные пользователем данные
        input_df['Age'] = age
        input_df['JobLevel'] = job_level
        input_df['MonthlyIncome'] = monthly_income
        input_df['YearsAtCompany'] = years_at_company
        input_df['EnvironmentSatisfaction'] = env_satisfaction
        input_df['JobSatisfaction'] = job_satisfaction
        input_df['WorkLifeBalance'] = work_life
        input_df['DistanceFromHome'] = distance

        if 'OverTime_Yes' in input_df.columns:
            input_df['OverTime_Yes'] = 1.0 if overtime == "Yes" else 0.0

        input_scaled = scaler.transform(input_df)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prob = model(input_tensor).item()

        st.divider()
        st.subheader("Результат прогноза:")
        if prob > 0.2:
            st.error(f"⚠️ **Высокий риск увольнения!** Вероятность: {prob * 100:.1f}%")
            st.progress(float(prob))
            st.write("Рекомендация: провести встречу 1-на-1, пересмотреть условия.")
        else:
            st.success(f"✅ **Низкий риск увольнения.** Вероятность: {prob * 100:.1f}%")
            st.progress(float(prob))
            st.write("Рекомендация: сотрудник лоялен. Поддерживайте условия.")

with tab2:
    st.subheader("Загрузка кадровых данных за период")
    st.markdown("Загрузите файл в формате `.csv` с исходными данными (структура IBM HR Dataset).")

    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success("Файл успешно загружен!")

        df_for_pred = raw_df.copy()
        cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Attrition']
        cols_to_drop = [col for col in cols_to_drop if col in df_for_pred.columns]
        df_for_pred = df_for_pred.drop(columns=cols_to_drop)

        cat_cols = df_for_pred.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df_for_pred, columns=cat_cols, drop_first=True)

        for col in features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[features]

        X_scaled = scaler.transform(df_encoded)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_tensor).numpy().flatten()

        raw_df['Predicted_Attrition_Risk'] = probs
        raw_df['Predicted_Attrition'] = (probs >= 0.2).astype(int)

        st.write("### Результаты прогноза:")
        st.dataframe(raw_df[['Age', 'Department', 'MonthlyIncome', 'Predicted_Attrition_Risk']].head(10))

        st.write("### Общая статистика:")
        high_risk_count = (probs >= 0.5).sum()
        st.write(f"Всего сотрудников в файле: **{len(raw_df)}**")
        st.write(
            f"В группе риска (вероятность > 50%): **{high_risk_count}** ({high_risk_count / len(raw_df) * 100:.1f}%)")

        csv_buffer = io.BytesIO()
        raw_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Скачать отчет (CSV с прогнозами)",
            data=csv_data,
            file_name=f"hr_attrition_report_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            type="primary"
        )

st.sidebar.info("Проект выполнен командой ИИ-26 и ИИ-25. Модель: PyTorch (Neural Network).")