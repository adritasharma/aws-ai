## Load pickle files

import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

class Prediction:
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

        input_encoded_scaled = self.encode_input()  

        self.predict(input_encoded_scaled)  

    def encode_input(self):
        # Example input data
        input_data = {
            'CreditScore': 600,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 40,
            'Tenure': 3,
            'Balance': 60000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary':50000
        }

        # Convert Categorical Data to Numerical

        #one-hot encode Geography
        geography_encoded = self.one_hot_encoder.transform([[input_data['Geography']]]).toarray()
        geography_encoded_df = pd.DataFrame(    
            geography_encoded,
            columns=self.one_hot_encoder.get_feature_names_out(['Geography'])
        )

        # Combine with other features
        input_df = pd.DataFrame([input_data])  


        # Encode Gender before concatenation
        input_df['Gender'] = self.label_encoder_gender.transform(input_df['Gender'])

        # Concatenate the one-hot encoded Geography with the input DataFrame
        input_df = pd.concat([input_df.drop('Geography', axis=1), geography_encoded_df], axis=1)

        print("Input DataFrame:")
        print(input_df)


        # Scale the data
        input_scaled = self.scaler.transform(input_df)

        print("Input Scaled DataFrame for prediction:")
        print(input_scaled)

        return input_scaled    
    
    def predict(self, input_encoded_scaled):
        prediction = self.model.predict(input_encoded_scaled)
        print("Prediction completed.", prediction)
