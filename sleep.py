import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os

# ===============================
# PAGE CONFIG (DARK MODE FRIENDLY)
# ===============================
st.set_page_config(
    page_title="Sleep Quality Predictor",
    layout="centered"
)

st.markdown("""
<style>
.main { padding: 20px; }
h1, h2, h3 { color: #EAEAEA; }
.stButton>button {
    background-color: #6c63ff;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("😴 Sleep Quality Prediction System")
st.caption("Predict sleep quality using lifestyle & health data")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

df = load_data()
df.drop(columns=["Person ID"], inplace=True)

# Split Blood Pressure
bp = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic BP"] = bp[0].astype(int)
df["Diastolic BP"] = bp[1].astype(int)
df.drop(columns=["Blood Pressure"], inplace=True)

# Encode categorical columns
label_encoders = {}
cat_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & Target
X = df.drop(columns=["Quality of Sleep"])
y = df["Quality of Sleep"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ===============================
# USER INPUT
# ===============================
st.header("👤 Enter Person Details")

gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
age = st.slider("Age", 18, 70, 30)
occupation = st.selectbox("Occupation", label_encoders["Occupation"].classes_)
sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0)
physical_activity = st.slider("Physical Activity Level", 0, 100, 50)
stress_level = st.slider("Stress Level", 1, 10, 5)
bmi_category = st.selectbox("BMI Category", label_encoders["BMI Category"].classes_)
heart_rate = st.slider("Heart Rate", 50, 120, 70)
daily_steps = st.slider("Daily Steps", 1000, 20000, 8000)
sleep_disorder = st.selectbox("Sleep Disorder", label_encoders["Sleep Disorder"].classes_)
systolic_bp = st.slider("Systolic BP", 90, 160, 120)
diastolic_bp = st.slider("Diastolic BP", 60, 100, 80)

# ===============================
# INPUT DATAFRAME
# ===============================
input_data = pd.DataFrame([[ 
    label_encoders["Gender"].transform([gender])[0],
    age,
    label_encoders["Occupation"].transform([occupation])[0],
    sleep_duration,
    physical_activity,
    stress_level,
    label_encoders["BMI Category"].transform([bmi_category])[0],
    heart_rate,
    daily_steps,
    label_encoders["Sleep Disorder"].transform([sleep_disorder])[0],
    systolic_bp,
    diastolic_bp
]], columns=X.columns)

# ===============================
# HISTORY
# ===============================
HISTORY_FILE = "sleep_history.csv"

def save_history(data):
    if not os.path.exists(HISTORY_FILE):
        data.to_csv(HISTORY_FILE, index=False)
    else:
        data.to_csv(HISTORY_FILE, mode="a", header=False, index=False)

# ===============================
# PREDICTION
# ===============================
if st.button("🔮 Predict Sleep Quality"):
    prediction = model.predict(input_data)[0]

    if prediction <= 5:
        category = "Poor"
        st.error(f"Sleep Quality Score: {prediction} → Poor 😴")
    elif prediction <= 7:
        category = "Average"
        st.warning(f"Sleep Quality Score: {prediction} → Average 🙂")
    else:
        category = "Good"
        st.success(f"Sleep Quality Score: {prediction} → Good 🌙")

    # Save history
    history = pd.DataFrame([{
        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sleep Duration": sleep_duration,
        "Physical Activity": physical_activity,
        "Stress Level": stress_level,
        "Sleep Score": prediction,
        "Category": category
    }])
    save_history(history)

    # ===============================
    # HEALTH ADVICE
    # ===============================
    st.subheader("💡 Personalized Advice")

    if sleep_duration < 6:
        st.write("🕒 Increase sleep duration to 7–8 hours.")
    if stress_level >= 7:
        st.write("🧘 Practice meditation or breathing exercises.")
    if physical_activity < 30:
        st.write("🏃 Increase physical activity.")
    if sleep_disorder != "None":
        st.write("📵 Reduce screen time before bed.")
    if systolic_bp > 140 or diastolic_bp > 90:
        st.write("🩺 Control BP with diet & stress management.")

    # ===============================
    # FEATURE IMPORTANCE (SMALL)
    # ===============================
    st.subheader("📊 Feature Importance")

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance")

    fig, ax = plt.subplots(figsize=(5, 4))  # 👈 reduced size
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_title("Key Factors")
    st.pyplot(fig)

    # ===============================
    # LIFESTYLE OVERVIEW (SMALL)
    # ===============================
    st.subheader("📈 Lifestyle Overview")

    fig2, ax2 = plt.subplots(figsize=(4, 3))  # 👈 reduced size
    ax2.bar(
        ["Sleep", "Activity", "Stress"],
        [sleep_duration, physical_activity, stress_level]
    )
    st.pyplot(fig2)

# ===============================
# HISTORY VIEW
# ===============================
st.header("📜 Prediction History")

if os.path.exists(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    st.dataframe(hist, width="stretch")  # ✅ no warning

    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(hist["Sleep Score"], marker="o")
    ax3.set_title("Sleep Score Trend")
    st.pyplot(fig3)
else:
    st.info("No history available yet.")
