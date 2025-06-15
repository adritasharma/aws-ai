import os
import pandas as pd
teams = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "teams.csv"))


teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
print(teams)

# Find the corelation between each column and the number of medals won
correlation = teams.select_dtypes(include='number').corr()["medals"].sort_values(ascending=False)
print("Correlation with medals won:", correlation)   