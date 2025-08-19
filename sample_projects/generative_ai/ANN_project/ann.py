from feature_transformation import FeatureTransformer
from ann_training import AnnTraining
from prediction import Prediction


# Initialize and run the feature transformation
featureTransformer = FeatureTransformer()
X_train, X_test, y_train, y_test = featureTransformer.run()
print("Feature transformation completed and data saved.")


# Initialize and run the ANN training
AnnTraining(X_train, X_test, y_train, y_test )
print("ANN training completed.")


# Initialize and run the prediction
prediction = Prediction()   


