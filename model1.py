# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:50:30 2023

@author: HP
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset from CSV
df = pd.read_csv('parf.csv')

df['Space occupied'] = df['Space occupied']/573
df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], format='%d-%m-%Y %H:%M')

df['year'] = df['Time Stamp'].dt.year
df['month'] = df['Time Stamp'].dt.month
df['day'] = df['Time Stamp'].dt.day
df['day_of_week'] = df['Time Stamp'].dt.weekday
df['hour'] = df['Time Stamp'].dt.hour
df['minute'] = df['Time Stamp'].dt.minute
df['date'] = df['Time Stamp'].dt.date
df['time'] = df['Time Stamp'].dt.strftime('%H:%M')
df['hour_min'] = round(df['hour'] + (df['minute'] / 60), 1)

 
X = df[['day', 'day_of_week', 'hour_min', 'Holiday']].values
y = df['Space occupied'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gradient Boosting Regression model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²) Score:", r2)

d = int(input("Enter the day: "))
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
wda = input("Enter the day of the week: ")
wd = days.index(wda)
hour = int(input("Enter the hour: "))
minute = int(input("Enter the minutes: "))
hm = round(hour + (minute / 60), 1)
hol = input("Enter if it's a holiday: ")

# Create a DataFrame for user input
user_input = pd.DataFrame({'day': [d], 'day_of_week': [wd], 'hour_min': [hm], 'Holiday': [hol]})

# Make prediction on user input using the trained model
user_pred = gbr.predict(user_input)

# Print the predicted output
print(round(user_pred[0] * 100, 2), "% of spaces are occupied.")


filename = 'final_model.pkl'
pickle.dump(gbr, open(filename, 'wb'))