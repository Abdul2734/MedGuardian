import os
os.system("pip install streamlit")

import numpy as np
import joblib
import streamlit as st
from warnings import filterwarnings
filterwarnings("ignore")
import os
import joblib
import pickle
print("Current working directory:", os.getcwd())

print("Files in the current directory:", os.listdir())

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    st.error(" Error loading model:")
    st.code(str(e))

st.title("🩺 MedGuardian - Diabetes Risk Prediction")
st.write("Enter your health details to check your diabetes risk.")

pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1, step=1)

if st.button("Predict"):
    if model is not None:
        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("High Risk of Diabetes!")
            st.info(" Tip: Reduce sugar, exercise regularly, and consult your doctor.")
        else:
            st.success("Low Risk of Diabetes")
            st.balloons()
            st.info("Keep maintaining a healthy lifestyle!")
    else:
        st.warning(" Model not loaded. Cannot make a prediction.")

st.markdown("---")
st.caption(" This is an educational tool. For actual medical diagnosis, consult a doctor.")
