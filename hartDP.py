# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:12:26 2023

@author: user
"""

import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle 
import streamlit as st 

# loaded the saved model 
hart_path = r"C:\Users\user\Downloads\pythonProject\hartDiseasePrediction\hartDisease_model.sav"
loaded_model = pickle.load(open(hart_path, 'rb'))

# creating a function
def heart_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is having heart disease'
    else:
        return 'The person does not have any heart disease'
  
def main():
    # giving a title
    st.title('Heart Disease Prediction using ML')

    
    col1, col2 = st.columns(2)
    
    # getting the input data from the user
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col1:
        cp = st.text_input('Chest Pain types')
    with col2:
        trestbps = st.text_input('Resting Blood Pressure')
    with col1:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col1:
        exang = st.text_input('Exercise Induced Angina')
    with col2:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col1:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col2:
        ca = st.text_input('Major vessels colored by fluoroscopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect; 3 = irreversible defect') 
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        # Convert user inputs to numeric types
        input_data = [
            float(age), float(sex), float(cp), float(trestbps), float(chol),
            float(fbs), float(restecg), float(thalach), float(exang),
            float(oldpeak), float(slope), float(ca), float(thal)
        ]
        diagnosis = heart_prediction([input_data])
        
    st.success(diagnosis)
    
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px; color: green;">
            <p><strong>Prepard by Anamul Haque Sayem (1901152)| EEE, RUET</strong> </p>
        </div>

        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
