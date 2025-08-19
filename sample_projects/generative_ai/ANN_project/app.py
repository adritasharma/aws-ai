import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st



class App:
    def __init__(self):
        model_path = 'ann_model.h5'
        self.model = load_model(model_path)

        print("Model loaded successfully.")

        ## Load Encoders and Scaler

        self.label_encoder_gender_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "label_encoder_gender.pkl")  
        self.one_hot_encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "one_hot_encoder_geo.pkl")  
        self.scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scaler.pkl")

        with open(self.label_encoder_gender_path, 'rb') as file:   
            self.label_encoder_gender = pickle.load(file)

        with open(self.one_hot_encoder_path, 'rb') as file:
            self.one_hot_encoder = pickle.load(file)    

        with open(self.scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)    

    def run(self):
        st.title("Customer Churn Prediction")

        geography = st.selectbox('Geography', self.one_hot_encoder.categories_[0])
        gender = st.selectbox('Gender', self.label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92)
        balance = st.number_input('Balance')
        credit_score = st.number_input('Credit Score')
        estimated_salary = st.number_input('Estimated Salary')
        tenure = st.slider('Tenure', 0, 10)
        num_of_products = st.slider('Number of Products', 1, 4)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active_member = st.selectbox('Is Active Member', [0, 1])

        # Prepare input data

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [self.label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = self.one_hot_encoder.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=self.one_hot_encoder.get_feature_names_out())

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = self.scaler.transform(input_data)

        # Predict churn
        prediction = self.model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.write(f'Prediction Probability: {prediction_proba:.2f}')

        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')

App().run()

     