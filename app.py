import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# --- Data Generation Function (Enhanced with More Parameters) ---
def generate_dummy_data(num_hospitals=5, num_months=36):
    np.random.seed(42)
    data = []
    cluster_id = 1  # Assuming all hospitals belong to one cluster for this demo

    for hospital_id in range(1, num_hospitals + 1):
        for month in range(1, num_months + 1):
            # Environmental Factors
            avg_temp = 20 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 2)
            is_ramadan = 1 if month == 9 else 0
            is_hajj = 1 if month == 12 else 0
            is_summer = 1 if 6 <= month <= 8 else 0
            is_winter = 1 if month in [1, 2, 12] else 0 # Define is_winter here
            is_school_holiday = 1 if month in [7, 8, 12] else 0  # Example: July, August, December

            # Hospital Operational Factors
            num_doctors = int(50 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5))
            num_nurses = int(100 + 20 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 10))
            bed_capacity = 200  # Assume constant for simplicity, but can be made variable
            
            # Patient Demographics & Volume
            patient_volume = 500 + 200 * is_ramadan + 300 * is_hajj + 100 * is_summer + np.random.randint(-50, 50)
            er_visits = int(patient_volume * (0.2 + 0.05 * is_ramadan + 0.1 * is_hajj + 0.02 * is_summer + np.random.normal(0, 0.02)))
            avg_age_er = 40 + 10 * is_ramadan - 5 * is_hajj + np.random.normal(0, 5)
            percentage_saudi = 0.7 + 0.1 * is_ramadan - 0.2 * is_hajj + np.random.normal(0, 0.05)
            
            # Common Diseases (Simplified Example)
            flu_cases = int(patient_volume * (0.05 + 0.03 * is_winter - 0.02 * is_summer + np.random.normal(0, 0.01)))
            gastro_cases = int(patient_volume * (0.03 + 0.02 * is_summer + np.random.normal(0, 0.01)))
            injury_cases = int(patient_volume * (0.02 + 0.01 * is_hajj + np.random.normal(0, 0.005)))
            
            #Infection
            num_clabsi_cases = int(max(0, 2 + 1 * is_ramadan + 2 * is_hajj + 0.5 * is_summer + np.random.normal(0, 1)))
            central_line_days = int(max(0, 200 + 50 * is_ramadan + 80 * is_hajj + 20 * is_summer + np.random.normal(0, 20)))
            clabsi_rate = 0 if central_line_days == 0 else (num_clabsi_cases / central_line_days) * 1000

            # Medical Supplies (Simplified Example)
            antibiotic_use_rate = 400 + 50 * is_ramadan + 80 * is_hajj + 20 * is_summer + np.random.normal(0, 30)
            iv_fluid_use = int(patient_volume * (0.5 + 0.1 * is_summer + np.random.normal(0, 0.05)))
            ppe_use = int(patient_volume * (1 + 0.2 * is_ramadan + 0.3 * is_hajj + np.random.normal(0, 0.1)))

            # Hospital Readiness Indicators
            
            hand_hygiene_compliance = 0.8 + 0.05 * is_ramadan - 0.1 * is_hajj + np.random.normal(0, 0.05)
            cleaning_score = 8 + np.random.normal(0, 0.5)
            medication_stockout = int(max(0, 1 + 0.5 * is_ramadan + 1 * is_hajj + np.random.normal(0, 0.5)))
            equipment_failures = int(max(0, 0 + 0.2 * is_ramadan + 0.5 * is_hajj + np.random.normal(0, 0.2)))
            icu_patients = int(max(0, 10 + 5 * is_ramadan + 8 * is_hajj + 2 * is_summer + np.random.normal(0, 2)))
            ventilator_days = int(max(0, 50 + 20 * is_ramadan + 30 * is_hajj + 10 * is_summer + np.random.normal(0, 10)))
            bed_occupancy_rate = 0.7 + 0.1 * is_ramadan + 0.15 * is_hajj + 0.05 * is_summer + np.random.normal(0, 0.05)
            bed_occupancy_rate = min(bed_occupancy_rate, 1.0)
            surgery_cases = int(patient_volume * (0.05 - 0.02 * is_ramadan + 0.01 * is_hajj + np.random.normal(0, 0.01)))

            data.append([month, hospital_id, cluster_id, avg_temp, is_ramadan, is_hajj, is_summer, is_school_holiday,
                         num_doctors, num_nurses, bed_capacity, patient_volume, er_visits, avg_age_er,
                         percentage_saudi, flu_cases, gastro_cases, injury_cases, antibiotic_use_rate, iv_fluid_use,
                         ppe_use, num_clabsi_cases, central_line_days, clabsi_rate,
                         hand_hygiene_compliance, cleaning_score, medication_stockout, equipment_failures, icu_patients, ventilator_days, bed_occupancy_rate, surgery_cases])

    columns = ["month", "hospital_id", "cluster_id", "avg_temp", "is_ramadan", "is_hajj", "is_summer", "is_school_holiday",
               "num_doctors", "num_nurses", "bed_capacity", "patient_volume", "er_visits", "avg_age_er",
               "percentage_saudi", "flu_cases", "gastro_cases", "injury_cases", "antibiotic_use_rate", "iv_fluid_use",
               "ppe_use", "num_clabsi_cases", "central_line_days", "clabsi_rate",
               "hand_hygiene_compliance", "cleaning_score", "medication_stockout", "equipment_failures", "icu_patients", "ventilator_days", "bed_occupancy_rate", "surgery_cases"]
    df = pd.DataFrame(data, columns=columns)
    return df

# --- Streamlit App ---
st.set_page_config(page_title="Healthcare Cluster Simulation", layout="wide")

st.title("Healthcare Cluster Simulation and Prediction")

# Generate Dummy Data
df = generate_dummy_data()

# --- Sidebar: Parameter Sliders ---
st.sidebar.header("Simulation Parameters")

# Example Sliders (Add more based on the dummy data columns)
num_doctors_adj = st.sidebar.slider("Number of Doctors Adjustment (%)", -50, 50, 0, step=5)
num_nurses_adj = st.sidebar.slider("Number of Nurses Adjustment (%)", -50, 50, 0, step=5)
avg_temp_adj = st.sidebar.slider("Average Temperature Adjustment (Celsius)", -5, 5, 0, step=1)
patient_volume_adj = st.sidebar.slider("Patient Volume Adjustment (%)", -20, 20, 0, step=5)
er_visits_adj = st.sidebar.slider("ER Visits Adjustment (%)", -20, 20, 0, step=5)
antibiotic_use_rate_adj = st.sidebar.slider("Antibiotic Use Rate Adjustment (%)", -20, 20, 0, step=5)
hand_hygiene_compliance_adj = st.sidebar.slider("Hand Hygiene Compliance Adjustment (%)", -20, 20, 0, step=5)
cleaning_score_adj = st.sidebar.slider("Cleaning Score Adjustment (0-10)", -2, 2, 0, step=1)
icu_patients_adj = st.sidebar.slider("ICU Patients Adjustment (%)", -20, 20, 0, step=5)
ventilator_days_adj = st.sidebar.slider("Ventilator Days Adjustment (%)", -20, 20, 0, step=5)

# --- Apply Adjustments to Data ---
# Create a copy of the dataframe to modify
modified_df = df.copy()

# Apply the adjustments based on slider values
modified_df["num_doctors"] = (modified_df["num_doctors"] * (1 + num_doctors_adj / 100)).astype(int)
modified_df["num_nurses"] = (modified_df["num_nurses"] * (1 + num_nurses_adj / 100)).astype(int)
modified_df["avg_temp"] += avg_temp_adj
modified_df["patient_volume"] = (modified_df["patient_volume"] * (1 + patient_volume_adj / 100)).astype(int)
modified_df["er_visits"] = (modified_df["er_visits"] * (1 + er_visits_adj / 100)).astype(int)
modified_df["antibiotic_use_rate"] = (modified_df["antibiotic_use_rate"] * (1 + antibiotic_use_rate_adj / 100))
modified_df["hand_hygiene_compliance"] = np.clip((modified_df["hand_hygiene_compliance"] * (1 + hand_hygiene_compliance_adj / 100)),0,1)
modified_df["cleaning_score"] = np.clip((modified_df["cleaning_score"] + cleaning_score_adj), 0, 10)
modified_df["icu_patients"] = (modified_df["icu_patients"] * (1 + icu_patients_adj / 100)).astype(int)
modified_df["ventilator_days"] = (modified_df["ventilator_days"] * (1 + ventilator_days_adj / 100)).astype(int)

# --- Model Building (Example: CLABSI Rate Prediction) ---
# Select features (X) and target (y)
X = modified_df[["avg_temp", "num_doctors", "num_nurses", "patient_volume", "er_visits", 
                 "antibiotic_use_rate", "hand_hygiene_compliance", "cleaning_score", "icu_patients", "ventilator_days"]]
y = modified_df["clabsi_rate"]

# Add a constant to the features
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# --- Predictions ---
# 1. Predict for Next 12 Months using Modified Data
future_months = list(range(1, 13))
future_hospitals = list(df['hospital_id'].unique())
prediction_df = pd.DataFrame([(month, hospital) for month in future_months for hospital in future_hospitals],
                             columns=['month', 'hospital_id'])

# Use the modified dataframe (with adjustments) for prediction inputs
for col in ["avg_temp", "num_doctors", "num_nurses", "patient_volume", "er_visits",
            "antibiotic_use_rate", "hand_hygiene_compliance", "cleaning_score", "icu_patients", "ventilator_days"]:
    prediction_df[col] = modified_df.groupby('hospital_id')[col].mean().to_dict()

# Predict using the model
X_future = prediction_df[["avg_temp", "num_doctors", "num_nurses", "patient_volume", "er_visits",
                         "antibiotic_use_rate", "hand_hygiene_compliance", "cleaning_score", "icu_patients", "ventilator_days"]]
X_future = sm.add_constant(X_future, has_constant='add')
predictions = model.predict(X_future)
prediction_df["predicted_clabsi_rate"] = predictions

# --- Visualizations ---
st.header("Simulation Results")

# CLABSI Rate Prediction
st.markdown("### Predicted CLABSI Rate")
fig_clabsi = px.line(prediction_df, x='month', y="predicted_clabsi_rate", color='hospital_id',
                     title="Predicted CLABSI Rate for Next 12 Months",
                     labels={"predicted_clabsi_rate": "Predicted CLABSI Rate (per 1000 central line days)"})
st.plotly_chart(fig_clabsi)

# --- Other Predictions & Visualizations (Examples) ---

# Example: Predict ER Visits
# ... (Build a model for ER visits using relevant parameters) ...

# Example: Predict Medical Supply Needs
# ... (Build models for antibiotic use, IV fluids, PPE, etc.) ...

# Example: Hospital Readiness Indicators
# ... (Visualize hand hygiene compliance, cleaning scores, etc.) ...

# Add more visualizations as needed based on the parameters and your prediction goals.
