import streamlit as st
import torch
import pandas as pd
import numpy as np
import os
import sys
import joblib
import io
from datetime import datetime
import matplotlib.pyplot as plt

# 1. Настройка страницы (обязательно первым делом!)
st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")

# 2. Инициализация хранилища для периодов (Пункт 10 ТЗ)
if 'history' not in st.session_state:
    st.session_state.history = []

# 3. Подключение нашей модели
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from model import AttritionNN

# --- НАСТРОЙКА ПУТЕЙ ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
processed_dir = os.path.join(project_root, 'data', 'processed')
models_dir = os.path.join(project_root, 'models')


# Загрузка модели и скалера (кэшируем)
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


# Универсальная функция предобработки данных (для вкладок 2 и 3)
def preprocess_uploaded_data(raw_df):
    df_for_pred = raw_df.copy()
    # Удаляем неинформативные колонки
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber', 'Attrition']
    cols_to_drop = [col for col in cols_to_drop if col in df_for_pred.columns]
    df_for_pred = df_for_pred.drop(columns=cols_to_drop)

    # One-Hot Encoding
    cat_cols = df_for_pred.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_for_pred, columns=cat_cols, drop_first=True)

    # Выравниваем колонки под те, на которых обучалась модель
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[features]

    # Масштабирование
    X_scaled = scaler.transform(df_encoded)
    return X_scaled


# --- ИНТЕРФЕЙС ---
st.title("👥 Система прогнозирования увольнений")
st.markdown("Аналитическая платформа для HR-специалистов (Прототип)")

# Создаем три вкладки
tab1, tab2, tab3 = st.tabs([
    "🔮 Индивидуальный прогноз",
    "📂 Массовая загрузка (CSV/XLSX)",
    "📈 HR-Аналитика"
])

# ================= ВКЛАДКА 1: Индивидуальный прогноз =================
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

    if st.button("🔮 Рассчитать риск увольнения", type="primary", key="btn_indiv"):
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

        # Используем бизнес-порог 20% (0.2)
        if prob >= 0.2:
            st.error(f"⚠️ **Высокий риск увольнения!** Вероятность: {prob * 100:.1f}%")
            st.progress(float(prob))
            st.write("Рекомендация: провести встречу 1-на-1, пересмотреть условия труда или зарплату.")
        else:
            st.success(f"✅ **Низкий риск увольнения.** Вероятность: {prob * 100:.1f}%")
            st.progress(float(prob))
            st.write("Рекомендация: сотрудник лоялен. Поддерживайте текущие условия.")

# ================= ВКЛАДКА 2: Массовая загрузка =================
with tab2:
    st.subheader("Загрузка кадровых данных за период")
    st.markdown("Загрузите файл в формате `.csv` или `.xlsx` (структура IBM HR Dataset).")

    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx"], key="mass_upload")

    if uploaded_file is not None:
        # Поддержка CSV и XLSX
        if uploaded_file.name.endswith('.csv'):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)

        st.success("Файл успешно загружен!")

        # Сохраняем период в историю (Пункт 10 ТЗ)
        period_name = datetime.now().strftime("Срез от %d.%m.%Y %H:%M")
        st.session_state.history.append({'name': period_name, 'data': raw_df})

        # Предобработка и прогноз
        X_scaled = preprocess_uploaded_data(raw_df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_tensor).numpy().flatten()

        raw_df['Predicted_Attrition_Risk'] = probs
        raw_df['Predicted_Attrition'] = (probs >= 0.2).astype(int)  # Порог 0.2

        st.write("### Результаты прогноза (первые 10 строк):")
        st.dataframe(raw_df[['Age', 'Department', 'MonthlyIncome', 'Predicted_Attrition_Risk']].head(10))

        # Анализ ошибок (Пункт 14 ТЗ)
        if 'Attrition' in raw_df.columns:
            st.write("### 🔍 Анализ ошибок модели (FP / FN)")
            actual = raw_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
            predicted = raw_df['Predicted_Attrition']

            fp = raw_df[(predicted == 1) & (actual == 0)]
            fn = raw_df[(predicted == 0) & (actual == 1)]

            st.write(f"🔴 **Ложноположительные (FP):** {len(fp)} чел. (Модель сказала 'уйдет', а он остался)")
            st.write(
                f"🟡 **Ложноотрицательные (FN):** {len(fn)} чел. (Модель сказала 'останется', а он уволился) — *Самые опасные!*")

            with st.expander("Посмотреть список упущенных сотрудников (FN)"):
                st.dataframe(fn[['Age', 'Department', 'MonthlyIncome']].head(5))

        st.write("### Общая статистика:")
        high_risk_count = (probs >= 0.2).sum()
        st.write(f"Всего сотрудников в файле: **{len(raw_df)}**")
        st.write(
            f"В группе риска (вероятность > 20%): **{high_risk_count}** ({high_risk_count / len(raw_df) * 100:.1f}%)")

        # Экспорт отчета (Пункт 16 ТЗ)
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

# ================= ВКЛАДКА 3: HR-Аналитика =================
with tab3:
    st.subheader("Аналитика рисков по подразделениям")

    # Выбор источника данных (загруженный сейчас или из истории)
    df_hr = None
    if st.session_state.history:
        periods = [h['name'] for h in st.session_state.history]
        selected_period = st.selectbox("Выберите сохраненный период", periods)
        df_hr = next(h['data'] for h in st.session_state.history if h['name'] == selected_period)
        st.info(f"Загружен сохраненный период: {selected_period}")
    else:
        st.warning("Нет загруженных данных. Загрузите файл во вкладке 'Массовая загрузка' или выберите файл ниже.")
        uploaded_analytics = st.file_uploader("Загрузите CSV/XLSX для аналитики", type=["csv", "xlsx"], key="analytics")
        if uploaded_analytics is not None:
            if uploaded_analytics.name.endswith('.csv'):
                df_hr = pd.read_csv(uploaded_analytics)
            else:
                df_hr = pd.read_excel(uploaded_analytics)
            # Добавляем в историю
            st.session_state.history.append({'name': "Текущий срез", 'data': df_hr})

    # Если данные есть (из истории или загружены), строим аналитику
    if df_hr is not None:
        # Предобработка и прогноз
        X_scaled = preprocess_uploaded_data(df_hr)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            probs = model(X_tensor).numpy().flatten()

        df_hr['Risk_Probability'] = probs
        df_hr['In_Risk_Zone'] = (probs >= 0.2).astype(int)

        # 1. Общая статистика
        total_emp = len(df_hr)
        risk_emp = df_hr['In_Risk_Zone'].sum()
        attrition_rate = (risk_emp / total_emp) * 100

        st.write("### Сводные показатели (Прогноз на следующий период)")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.metric("Всего сотрудников", total_emp)
        col_stat2.metric("В группе риска (>20%)", risk_emp)
        col_stat3.metric("Прогноз текучести", f"{attrition_rate:.1f}%")

        # 2. Визуализация по отделам
        st.write("### Динамика рисков в разрезе Департаментов")

        if 'Department' in df_hr.columns:
            dept_stats = df_hr.groupby('Department')['In_Risk_Zone'].agg(['count', 'sum']).reset_index()
            dept_stats.columns = ['Department', 'Всего', 'В зоне риска']

            # Строим график
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(dept_stats['Department']))
            width = 0.35

            ax.bar(x - width / 2, dept_stats['Всего'], width, label='Всего', color='#4C72B0')
            ax.bar(x + width / 2, dept_stats['В зоне риска'], width, label='В зоне риска', color='#C44E52')

            ax.set_ylabel('Количество сотрудников')
            ax.set_title('Сотрудники в зоне риска по департаментам')
            ax.set_xticks(x)
            ax.set_xticklabels(dept_stats['Department'])
            ax.legend()

            st.pyplot(fig)
        else:
            st.warning("В загруженном файле нет колонки 'Department' для построения графика.")

# Боковая панель
st.sidebar.info("Проект выполнен командой ИИ-26 и ИИ-25. Модель: PyTorch (Neural Network).")