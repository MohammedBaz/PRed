import streamlit as st
import pandas as pd

# Input Factors and Default Values
st.title("Healthcare Readiness Simulation")

# Input Section
st.sidebar.header("Input Factors")
avg_temp = st.sidebar.slider("Average Temperature (°C)", 10, 50, 30)
is_hajj = st.sidebar.checkbox("Hajj Season")
is_ramadan = st.sidebar.checkbox("Ramadan Season")
is_summer = st.sidebar.checkbox("Summer")
is_winter = st.sidebar.checkbox("Winter")
is_school_holiday = st.sidebar.checkbox("School Holiday")
num_doctors = st.sidebar.number_input("Number of Doctors", min_value=0, value=50)
num_nurses = st.sidebar.number_input("Number of Nurses", min_value=0, value=100)
patient_volume = st.sidebar.number_input("Expected Patient Volume", min_value=0, value=500)
er_visits = st.sidebar.number_input("Expected ER Visits", min_value=0, value=100)
percentage_saudi = st.sidebar.slider("Percentage of Saudi Patients (%)", 0, 100, 70)
top_3_countries = st.sidebar.text_area("Top 3 Countries of Patient Origin", "Country1, Country2, Country3")

# Readiness Indicators Computation
def compute_readiness_indicators():
    # Parse top 3 countries
    countries = [c.strip() for c in top_3_countries.split(",")]

    # Equipment Needs
    equipment_score = (
        (avg_temp > 35) * 1.5 +  # High temperature increases equipment needs
        is_hajj * 2.0 +
        is_winter * 1.2 +
        is_summer * 1.3
    )

    # Medicine Needs
    medicine_score = (
        (avg_temp < 15) * 1.5 +  # Cold weather increases medicine needs
        patient_volume * 0.01 +
        er_visits * 0.02
    )

    # Staffing Adequacy
    staffing_score = (num_doctors + num_nurses) / (patient_volume + er_visits) * 100

    # Testing Capacity
    testing_score = 0
    for country in countries:
        if country == "Country1":
            testing_score += 1.5
        elif country == "Country2":
            testing_score += 1.0
    testing_score += is_hajj * 2.0

    # Infection Control (CLABSI Rate)
    clabsi_rate = (
        1 / ((num_doctors + num_nurses) * 0.05 + 0.1) * 100  # Simplified example
    )

    # Interpret CLABSI Rate
    if clabsi_rate < 5:
        clabsi_interpretation = "Excellent"
    elif clabsi_rate < 10:
        clabsi_interpretation = "Good"
    elif clabsi_rate < 15:
        clabsi_interpretation = "Moderate"
    else:
        clabsi_interpretation = "Needs Improvement"

    return {
        "Equipment Needs": round(equipment_score, 2),
        "Medicine Needs": round(medicine_score, 2),
        "Staffing Adequacy (%)": round(staffing_score, 2),
        "Testing Capacity": round(testing_score, 2),
        "CLABSI Rate (%)": round(clabsi_rate, 2),
        "CLABSI Interpretation": clabsi_interpretation,
    }

indicators = compute_readiness_indicators()

# Display Results
st.subheader("Readiness Indicators")
st.write(pd.DataFrame(indicators, index=["Value"]).T)

# Visualization (Bar Chart)
st.bar_chart(pd.DataFrame({k: v for k, v in indicators.items() if isinstance(v, (int, float))}, index=["Value"]).T)
