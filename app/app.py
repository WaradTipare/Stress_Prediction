import streamlit as st
import joblib
import plotly.graph_objects as go

# ✅ 1. PAGE CONFIG (MOVED TO TOP)
st.set_page_config(
    page_title="Stress Prediction System",
    page_icon="🎯",
    layout="centered"
)

# ✅ 2. FIXED CSS
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    text-align: center;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stSlider {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load(r"C:\Users\warad\OneDrive\Desktop\projects\stress_prediction\model\model.pkl")
scaler = joblib.load(r"C:\Users\warad\OneDrive\Desktop\projects\stress_prediction\model\scaler.pkl")

# ---------------------------
# Title
# ---------------------------
st.title("🎯 Student Stress Prediction System")
st.write("Enter your details to predict stress level")

# ---------------------------
# Input Fields
# ---------------------------
st.markdown("## 📊 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours", 0, 12, 6)
    study = st.slider("📚 Study Hours", 0, 12, 5)
    screen = st.slider("📱 Screen Time", 0, 12, 6)

with col2:
    workload = st.slider("📝 Workload (tasks)", 0, 15, 5)
    mood = st.slider("😊 Mood Score", 1, 10, 5)

# ---------------------------
# Predict Button (IMPROVED)
# ---------------------------
st.markdown("###")
predict_btn = st.button("🚀 Predict Stress Level")

# ---------------------------
# Prediction
# ---------------------------
if predict_btn:

    input_data = [[sleep, study, screen, workload, mood]]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    # ---------------------------
    # RESULT TEXT (CLEANED)
    # ---------------------------
    if prediction == 0:
        st.success("Low Stress 😊")
        st.write("You are doing well! Maintain your routine and healthy habits.")

    elif prediction == 1:
        st.warning("Medium Stress ⚠️")
        st.write("Try improving time management, sleep, and reduce screen time.")

    else:
        st.error("High Stress 🚨")
        st.write("High stress detected. Consider relaxation techniques and seek support if needed.")

        # 🆘 Support Section
        st.markdown("### 🧠 Need someone to talk to?")
        st.info("""
        If you are feeling overwhelmed, talking to someone can help.

        📞 **24/7 Mental Health Helpline (India):**
        - wxyz: **+91-123456789**
        - abcd: **+91-123456789**

        💬 You are not alone. Support is available anytime.
        """)

    st.markdown("### 💡 Personalized Suggestions")

    suggestions = []

    # Sleep
    if sleep < 6:
        suggestions.append("😴 Try to increase your sleep duration to at least 7–8 hours.")

    # Study
    if study > 8:
        suggestions.append("📚 Consider reducing study overload and take breaks.")

    # Screen Time
    if screen > 8:
        suggestions.append("📱 Reduce screen time to avoid mental fatigue.")

    # Workload
    if workload > 10:
        suggestions.append("📝 Try to manage your workload and prioritize tasks.")

    # Mood
    if mood < 5:
        suggestions.append("🧠 Engage in activities that improve your mood (exercise, hobbies, talking to someone).")

    # If no issues
    if not suggestions:
        st.success("✅ Your lifestyle looks balanced. Keep it up!")

    else:
        for s in suggestions:
            st.write("•", s)

    # ---------------------------
    # STRESS SCORE (FIXED INDENT + CLEAN)
    # ---------------------------
    stress_score = (
        (12 - sleep) * 3 +
        study * 2 +
        screen * 2 +
        workload * 2 +
        (10 - mood) * 3
    )

    value = max(0, min(100, int(stress_score)))

    # ---------------------------
    # LABEL (USED PROPERLY)
    # ---------------------------
    labels = {
        0: "Low Stress 😊",
        1: "Medium Stress ⚠️",
        2: "High Stress 🚨"
    }

    st.subheader(labels[prediction])

    # ---------------------------
    # 🔥 IMPROVED GAUGE (PREMIUM LOOK)
    # ---------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Stress Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00ADB5"},
            'bgcolor': "#1E1E1E",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 33], 'color': "#00FF9C"},
                {'range': [33, 66], 'color': "#FFD369"},
                {'range': [66, 100], 'color': "#FF4C4C"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)