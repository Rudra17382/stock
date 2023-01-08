import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
import numpy as np

# Fetch data for a stock using pandas_datareader
ticker = "AAPL"
timeframe = "1d"

# Download the stock data
df = yf.download(ticker, period= "5y", interval= "1d")

# Preprocess the data
df = df.dropna()
#df = (df - df.mean()) / df.std())

# Create time windows of past data to use as features
num_windows = 30
df_windows = []
for i in range(num_windows):
    df_windows.append(df.shift(i).dropna())

# Concatenate the time windows into a single dataframe
df_X = pd.concat(df_windows, axis=1).dropna()

# Use the Close value as the label
df_y = df[[i in df_X.index for i in df["Close"].index]]["Close"]

# Split the data into training and test sets
df_X_train = df_X[:-1000]
df_y_train = df_y[:-1000]
df_X_test = df_X[-1000:]
df_y_test = df_y[-1000:]

# Train a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(df_X_train, df_y_train)

# Make predictions on the test set
predictions = model.predict(df_X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df_y_test, predictions)
rmse = np.sqrt(mse)
print('Root mean squared error:', rmse)

# Plot the predicted vs. actual values
import matplotlib.pyplot as plt
plt.plot(predictions, label='Predicted')
plt.plot(df_y_test.values, label='Actual', alpha=0.5)
plt.legend()
plt.show()
