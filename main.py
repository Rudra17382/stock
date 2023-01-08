import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


pd.set_option('mode.use_inf_as_na', True)

# Set the stock ticker and time frame
ticker = "IBM"
timeframe = "1d"

# Download the stock data
data = yf.download(ticker, period= "Max", interval="1m")

data = data.pct_change().dropna()

# Create a target vector by shifting the close price one day into the future
y = data["Close"].shift(-1).dropna()

# Create a feature matrix by selecting some relevant columns from the data
X = data[["Open", "Close", "Volume"]]
X = X[[i in y.index for i in X.index]]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier on the training data
clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train)

clf = tf.keras.models.load_model('Bi-LSTM.hdf5')

# Make predictions on the test set
predictions = clf.predict(X_test)
X_test["predictions"] = predictions
X_test.sort_index(inplace=True)
y_test.sort_index(inplace=True)

# Calculate the accuracy of the predictions
mse = mean_squared_error(y_test, predictions)
print(mse)

# Plot the predicted vs. actual values
import matplotlib.pyplot as plt
plt.plot(X_test["predictions"],label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.show()
