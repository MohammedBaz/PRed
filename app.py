import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- Data Generation Function ---
def generate_dummy_data(num_hospitals=5, num_months=36):
    np.random.seed(42)
    data = []
    cluster_id = 1 # Assuming all hospitals belong to one cluster for this demo

    for hospital_id in range(1, num_hospitals + 1):
        for month in range(1, num_months + 1):
            avg_temp = 20 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 2)
            is_ramadan = 1 if month == 9 else 0
            is_hajj = 1 if month == 12 else 0
            is_summer = 1 if 6 <= month <= 8 else 0
            patient_volume = 500 + 200 * is_ramadan + 300 * is_hajj + 100 * is_summer + np.random.randint(-50, 50)
            er_visits = int(patient_volume * (0.2 + 0.05 * is_ramadan + 0.1 * is_hajj + 0.02 * is_summer + np.random.normal(0, 0.02)))
            bed_occupancy_rate = 0.7 + 0.1 * is_ramadan + 0.15 * is_hajj + 0.05 * is_summer + np.random.normal(0, 0.05)
            bed_occupancy_rate = min(bed_occupancy_rate, 1.0)  # Cap at 100%
            surgery_cases = int(patient_volume * (0.05 - 0.02 * is_ramadan + 0.01 * is_hajj + np.random.normal(0, 0.01)))
            staffing_level_nurses = 1.2 - 0.2 * is_ramadan + 0.1 * is_hajj + np.random.normal(0, 0.1)
            staffing_level_doctors = 0.8 - 0.1 * is_ramadan + 0.05 * is_hajj + np.random.normal(0, 0.05)
            antibiotic_use_rate = 400 + 50 * is_ramadan + 80 * is_hajj + 20 * is_summer + np.random.normal(0, 30)
            hand_hygiene_compliance = 0.8 + 0.05 * is_ramadan - 0.1 * is_hajj + np.random.normal(0, 0.05)
            cleaning_score = 8 + np.random.normal(0, 0.5)
            num_clabsi_cases = int(max(0, 2 + 1 * is_ramadan + 2 * is_hajj + 0.5 * is_summer + np.random.normal(0, 1)))
            ventilator_days = int(max(0, 50 + 20 * is_ramadan + 30 * is_hajj + 10 * is_summer + np.random.normal(0, 10)))
            central_line_days = int(max(0, 200 + 50 * is_ramadan + 80 * is_hajj + 20 * is_summer + np.random.normal(0, 20)))
            clabsi_rate = 0 if central_line_days == 0 else (num_clabsi_cases / central_line_days) * 1000
            icu_patients = int(max(0, 10 + 5 * is_ramadan + 8 * is_hajj + 2 * is_summer + np.random.normal(0, 2)))
            medication_stockout = int(max(0, 1 + 0.5 * is_ramadan + 1 * is_hajj + np.random.normal(0, 0.5)))
            equipment_failures = int(max(0, 0 + 0.2 * is_ramadan + 0.5 * is_hajj + np.random.normal(0, 0.2)))

            data.append([month, hospital_id, cluster_id, avg_temp, is_ramadan, is_hajj, is_summer, patient_volume, er_visits, bed_occupancy_rate,
                         surgery_cases, staffing_level_nurses, staffing_level_doctors, antibiotic_use_rate, hand_hygiene_compliance,
                         cleaning_score, num_clabsi_cases, clabsi_rate, ventilator_days, central_line_days, icu_patients, medication_stockout, equipment_failures])

    columns = ["month", "hospital_id", "cluster_id", "avg_temp", "is_ramadan", "is_hajj", "is_summer", "patient_volume", "er_visits",
               "bed_occupancy_rate", "surgery_cases", "staffing_level_nurses", "staffing_level_doctors", "antibiotic_use_rate",
               "hand_hygiene_compliance", "cleaning_score", "num_clabsi_cases", "clabsi_rate", "ventilator_days", "central_line_days", "icu_patients", "medication_stockout", "equipment_failures"]
    df = pd.DataFrame(data, columns=columns)
    return df

# --- Streamlit App ---
st.set_page_config(page_title="Healthcare Cluster Prediction", layout="wide")

st.title("Healthcare Cluster Prediction")

# Generate Dummy Data
df = generate_dummy_data()

# Sidebar Filters
st.sidebar.header("Filters")
selected_hospitals = st.sidebar.multiselect("Select Hospitals", df["hospital_id"].unique(), default=df["hospital_id"].unique())
selected_months = st.sidebar.multiselect("Select Months", df["month"].unique(), default=df["month"].unique())

# Filter Data
filtered_df = df[df["hospital_id"].isin(selected_hospitals) & df["month"].isin(selected_months)]

# --- Scenario 1: Infection Prediction ---
st.header("Scenario 1: Infection Prediction (CLABSI)")

# Model
X1 = filtered_df[["avg_temp", "patient_volume", "er_visits", "antibiotic_use_rate", "hand_hygiene_compliance",
                  "cleaning_score", "staffing_level_nurses", "staffing_level_doctors", "bed_occupancy_rate", "icu_patients", "ventilator_days", "central_line_days", "medication_stockout", "equipment_failures"]]
y1 = filtered_df["clabsi_rate"]
X1 = sm.add_constant(X1)  # Add constant
model1 = sm.OLS(y1, X1).fit()

# Prediction
st.markdown("### CLABSI Rate Prediction")
# Create a dataframe for predictions
future_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Example: Predict for the next 12 months
future_hospitals = selected_hospitals # Predict for selected hospitals
prediction_df = pd.DataFrame([(month, hospital) for month in future_months for hospital in future_hospitals],
                             columns=['month', 'hospital_id'])

# Add other necessary columns with average or default values
prediction_df['avg_temp'] = prediction_df['month'].apply(lambda m: 20 + 10 * np.sin(2 * np.pi * m / 12))  # Simulate temperature
prediction_df['is_ramadan'] = prediction_df['month'].apply(lambda m: 1 if m == 9 else 0)
prediction_df['is_hajj'] = prediction_df['month'].apply(lambda m: 1 if m == 12 else 0)
prediction_df['is_summer'] = prediction_df['month'].apply(lambda m: 1 if 6 <= m <= 8 else 0)
# Use mean values from filtered_df for other columns
for col in ["patient_volume", "er_visits", "antibiotic_use_rate", "hand_hygiene_compliance",
            "cleaning_score", "staffing_level_nurses", "staffing_level_doctors", "bed_occupancy_rate", "icu_patients", "ventilator_days", "central_line_days", "medication_stockout", "equipment_failures"]:
    prediction_df[col] = filtered_df[col].mean()

# Predict
X_future = prediction_df[["avg_temp", "patient_volume", "er_visits", "antibiotic_use_rate", "hand_hygiene_compliance",
                          "cleaning_score", "staffing_level_nurses", "staffing_level_doctors", "bed_occupancy_rate", "icu_patients", "ventilator_days", "central_line_days", "medication_stockout", "equipment_failures"]]
X_future = sm.add_constant(X_future, has_constant='add')
predictions = model1.predict(X_future)
prediction_df["predicted_clabsi_rate"] = predictions

# create proper month names and year to display the data in the right way
prediction_df['month_name'] = prediction_df['month'].apply(lambda x: pd.to_datetime(x, format='%m').month_name())
prediction_df['year'] = 2024 # it is prediction for this year
# merge month and year to get a date
prediction_df['date'] = prediction_df['month_name'] + ' ' + prediction_df['year'].astype(str)

# Line chart for predicted CLABSI
