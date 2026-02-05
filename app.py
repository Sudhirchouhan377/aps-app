import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="APS Failure Prediction", layout="centered")

st.title("🚛 APS Failure Prediction System")
st.write("Upload sensor data to predict the probability of APS failure.")

# ---------------- LOAD MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("### 🔍 Uploaded Data Preview", data.head())

        # ---------------- PREDICT BUTTON ----------------
        if st.button("🔮 Predict Failure Risk"):
            proba = model.predict_proba(data)[0][1]   # Probability of failure class
            percentage = round(proba * 100, 2)

            st.subheader(f"⚠️ Failure Probability: **{percentage}%**")

            # Risk indicator
            if percentage > 70:
                st.error("🚨 HIGH RISK: Immediate maintenance required!")
            elif percentage > 40:
                st.warning("⚡ MEDIUM RISK: Monitor vehicle condition.")
            else:
                st.success("✅ LOW RISK: Vehicle operating normally.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit | APS Failure ML Model")
