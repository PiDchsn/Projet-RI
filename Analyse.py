import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Loading the data
data = pd.read_csv('./data.csv', sep=";")
# Selecting specific columns and converting to a NumPy array
data = data.iloc[:, [0, 1, 6]].to_numpy()

# Ensure the time series data is numeric and drop any non-numeric values
time_series_data = pd.Series(data[:, 2], dtype='float64')

# Drop any NaN values that may cause issues
time_series_data = time_series_data.dropna()

# Plotting the time series
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label='Valeur de la série temporelle')
plt.legend()
plt.show()

# SARIMA parameters
p, d, q = 1, 1, 1    
P, D, Q, s = 1, 1, 1, 96  # Seasonality every 96 time steps

# Building the SARIMA model
model = SARIMAX(time_series_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

# Display the model summary
print(model_fit.summary())