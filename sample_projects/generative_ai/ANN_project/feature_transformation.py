import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Churn_Modelling.csv")
data = pd.read_csv(data_path)


## Feature Transformation using Sklearn

## Preprocessing the data
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

## Here Geography and gender are categorical variables

## Encode categorical variables i.e convert male and female to 0 and 1
label_encoder_gender = LabelEncoder()

data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

print(data.head())

## Encode Geography using OneHot Encoding (OHE)
oneHotEncoder_geography = OneHotEncoder()
geography_encoded = oneHotEncoder_geography.fit_transform(data[['Geography']])

print(geography_encoded.toarray())

geography_encoded_df = pd.DataFrame(geography_encoded.toarray(), columns=oneHotEncoder_geography.get_feature_names_out(['Geography']))

print(geography_encoded_df)

## Combine OHE columns with the original dataframe
data = pd.concat([data.drop('Geography', axis=1), geography_encoded_df], axis=1)

print(data)

## save file using pickle
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_data.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(data, f)                        


## Divide the dataset into independent and dependent variables

X = data.drop('Exited', axis=1)
y = data['Exited']  

## Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Scale the features
scaler = StandardScaler()   
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)  

    