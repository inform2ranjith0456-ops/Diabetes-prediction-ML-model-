
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(
    open('C:/Users/infor/OneDrive/Desktop/Diabetes/trained_model.sav', 'rb')
)

def diabetes_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():

    st.title('Diabetes Prediction Web App')

    st.write('Enter the following details to check the prediction')

    Pregnancies = st.number_input('Number of Pregnancies', min_value=0)
    Glucose = st.number_input('Glucose Level', min_value=0.0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0.0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0.0)
    Insulin = st.number_input('Insulin Level', min_value=0.0)
    BMI = st.number_input('BMI value', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input(
        'Diabetes Pedigree Function value', min_value=0.0
    )
    Age = st.number_input('Age of the Person', min_value=1)

    if st.button('Diabetes Test Result'):

        input_data = [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]

        diagnosis = diabetes_prediction(input_data)

        if diagnosis == 'The person is not diabetic':
            st.success(diagnosis)
        else:
            st.error(diagnosis)

if __name__ == '__main__':
    main()
