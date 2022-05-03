from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf


#################### Load Data #######################

company = 'SPY'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)

################# Prepare Data for NN ####################


scaler = MinMaxScaler(feature_range=(0, 1))  # fit all numbers between 0 and 1

# transform the close prices in scaled format
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60  # of past days

# need 2 x, y lists
x_train = []
y_train = []


for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

# convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape , so it works with NN
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


##################### BUILD THE NN MODEL ########################
model = Sequential()

# create the layers isn LSTM then Dropout format , then Dense layer(the prediction)

model.add(LSTM(units=50, return_sequences=True,
               input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # prediction for next closing value


# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# fit model to the training data
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test the Model Accuracy on existing data '''

##################### Load test Data ############################

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values
# combine both from above
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(
    total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

############## Make Predictions on Test Data ######################

x_test = []

for x in range(prediction_days, len(model_inputs) + 1):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
# reverse the scaler , to get actual numbers
predicted_prices = scaler.inverse_transform(predicted_prices)

################# plot the test predictions #####################
plt.plot(actual_prices, color="black", label=f" Actual {company} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
plt.title(f" {company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

################### Predict next day #############################

real_data = [model_inputs[len(model_inputs) + 1 -
                          prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

print(scaler.inverse_transform(real_data[-1]))
################ BUG #################
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")
