# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

stock = "NDX"

# The start and end date of where nasdaq100 will be analyzed
start = dt.datetime(2020,1,1)
end = dt.datetime(2021,10,1)

# Read the data from yahoo website
data = web.DataReader(stock, 'yahoo', start, end)
print(data.head())

#show graph
#data['Open'].plot(label = 'nasdaq opening price')
#data['Close'].plot(label = 'nasdaq closing price',figsize = (5,5))
#plt.legend()
#plt.title('Nasdaq Price Analysis')
#plt.show()


#Scale the data to fit within 0 to 10
scaler = MinMaxScaler(feature_range=(0,10))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

#Number of previous days the model can look back into
prediction_days = 10

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train) ,np.array(y_train)
x_train = np.reshape(x_train.shape[0], x_train.shape[1], [1])

#Create model
model = Sequential()

model.add(LSTM(units=50, return_sequences= True, input_shape=(x_train.shape[1],[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=2))

model,compile(optimizer='adam', loss='mean_squared_error')
model(x_train,y_train, epochs=25, batch_size=32)

# Testing model Accuracy

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(stock, 'yahoo', test_start,test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make the prediction on test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#plot the predictions
plt.plot(actual_prices, color='black', label=f"Actual{stock} Price" )
plt.plot(predicted_prices, color='green', label=f"Predicted {stock} Price" )
plt.title(f"{stock} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{stock} Share Price")
plt.legend()
plt.show()

#Predict the next day of stock price

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")