import pickle
import pandas as pd
import streamlit as st

with open('model.pkl','rb') as file:
    model_data=pickle.load(file)

lr=model_data['model']
sc=model_data['scaler']
columns=model_data['columns']

st.title('Diabetes Prediction')
st.write('Enter the patient details to predict diabetes.')

pregnancies=st.number_input('Pregnancies',min_value=0,value=1)
glucose=st.number_input('Glucose',min_value=0,value=120)
blood_pressure=st.number_input('Blood Pressure',min_value=0,value=70)
skin_thickness=st.number_input('Skin Thickness',min_value=0,value=20)
insulin=st.number_input('Insulin',min_value=0,value=80)
bmi=st.number_input('BMI',min_value=0.0,value=25.0)
diabetes_pedigree_function=st.number_input('Diabetes Pedigree Function',min_value=0.0,value=0.5)
age=st.number_input('Age',min_value=1,value=30)

if st.button('Predict'):
    data=pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]],columns=columns)

    data[columns]=sc.transform(data[columns])
    prediction=lr.predict(data)
    probability=lr.predict_proba(data)[:,1]

    if prediction[0]==1:
        st.error(f'Diabetic - Probability: {probability[0]:.2f}')
    else:
        st.success(f'Not Diabetic - Probability: {probability[0]:.2f}')
