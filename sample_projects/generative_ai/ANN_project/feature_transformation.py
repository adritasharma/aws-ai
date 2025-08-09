import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

class FeatureTransformer:
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Churn_Modelling.csv")
        self.output_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_data.pkl")
        self.scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scaler.pkl")
        self.data = None
        self.label_encoder_gender = LabelEncoder()
        self.oneHotEncoder_geography = OneHotEncoder()
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.run()

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def preprocess(self):
        self.data = self.data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    def encode_gender(self):
        self.data['Gender'] = self.label_encoder_gender.fit_transform(self.data['Gender'])

    def encode_geography(self):
        geography_encoded = self.oneHotEncoder_geography.fit_transform(self.data[['Geography']])
        geography_encoded_df = pd.DataFrame(
            geography_encoded.toarray(),
            columns=self.oneHotEncoder_geography.get_feature_names_out(['Geography'])
        )
        self.data = pd.concat([self.data.drop('Geography', axis=1), geography_encoded_df], axis=1)

    def save_processed_data(self):
        with open(self.output_data_path, 'wb') as f:
            pickle.dump(self.data, f)

    def split_and_scale(self):
        X = self.data.drop('Exited', axis=1)
        y = self.data['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def run(self):
        self.load_data()
        self.preprocess()
        self.encode_gender()
        self.encode_geography()
        self.save_processed_data()
        self.split_and_scale()

