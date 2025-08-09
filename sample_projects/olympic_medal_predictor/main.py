import os
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

teams = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teams.csv"))


teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
print(teams)

# Find the corelation between each column and the number of medals won
correlation = teams.select_dtypes(include='number').corr()["medals"].sort_values(ascending=False)
print("Correlation with medals won:", correlation)   


sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
plt.show() 

sns.lmplot(x="prev_medals", y="medals", data=teams, fit_reg=True, ci=None)
plt.show() 

# Check how balanced the data is:

# We will make histogram to look how many countries fall into each bin for no of medals won.
# We can see that lot the countries have won bw 0 to 50 medals and ther's very few countries that have earned lot of medals. So our data is little unbalanced.
# It will impact our accuracy.
teams.plot.hist(y="medals")

# Data cleaning; Find out if there are any null values in the data and drop them
teams = teams.dropna()


## Data splitting

# Make sure we don't use future data to predict past data

train = teams[teams["year"] < 2012]
test = teams[teams["year"] >= 2012]

print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

# Training the model

# Create a Linear Regression model
reg = LinearRegression()

# Define the predictor columns 
predictors = ["athletes", "prev_medals"]

# Fit the model using the training data
reg.fit(train[predictors], train["medals"])

# Make predictions on the test set
predictions = reg.predict(test[predictors])

# Add the predictions as a new column in the test DataFrame
test["predictions"] = predictions

# Print relevant columns to compare actual vs predicted medals
print(test[["team", "country", "year", "athletes", "prev_medals", "medals", "predictions"]])

# Set any negative predictions to zero (since negative medals are not possible)
test.loc[test["predictions"] < 0, "predictions"] = 0

# Round the predictions to the nearest whole number
test["predictions"] = test["predictions"].round(0)

print(test)

error = mean_absolute_error(test["medals"], test["predictions"])
print("Mean Absolute Error:", error)