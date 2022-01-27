
# Import the libraries
import math
import pandas_datareader as web
import numpy as np
import plotly.graph_objs as go

from datetime import date
import json
import plotly
import pickle
import os
from os import path

# needs to be removed and improved
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from datetime import date, timedelta


from flask import Flask, redirect, url_for, render_template
import pickle

directory = 'Stocks'
figure = ""


def writeFile(stock_ticker, json_data):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    fileDir = directory + '/' + stock_ticker + '.json'
    with open(fileDir, 'w') as outfile:
        json.dump(json_data, outfile)
    return 0


def toJson(pred):
    json_output = {}
    json_output["predictionDate"] = date.today().strftime("%Y-%m-%d")
    json_output["prediction"] = str(pred)
    json_output["chart"] = str(figure)
    return json_output


def getChart():
    return figure


def readFile(stock_ticker):
    # if no folder exists, create one
    file_name = directory + '/' + stock_ticker + '.json'
    output = {}
    if not os.path.isdir(directory):
        os.mkdir(directory)
        return output
    if not path.isfile(file_name):
        return output
    # read file and parse it into a json
    with open(file_name) as json_file:
        output = json.load(json_file)
    return output


def getPrice(stock_ticker):
    stock_ticker = stock_ticker.upper()
    # Check the stock by it's key
    try:
        web.DataReader(stock_ticker, data_source='yahoo')
    except:
        return 0
    # Get data from previous day
    json_data = readFile(stock_ticker)
    # Check if data was predicted today
    if json_data == {} or json_data["predictionDate"] != date.today().strftime("%Y-%m-%d"):
        output = predict(stock_ticker)
        writeFile(stock_ticker, toJson(output))
        return output
    else:
        output = json_data["prediction"]
    return json_data["prediction"]  # output


def predict(stock_ticker):
    # Get current date
    today = date.today().strftime("%Y-%m-%d")
    # Get tomorrows date
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Get the past stock ticker values (between start and end dates)
    stock_data_frame = web.DataReader(
        stock_ticker, data_source='yahoo', start='2014-01-01', end=today)

    # Filter data frame to contain only Closing Stock Values
    closing_data = stock_data_frame.filter(['Close'])

    # Convert the dataframe to a numpy array
    close_dataset = closing_data.values

    # Set the training data set to 90% of original data values
    training_data_length = math.ceil(len(close_dataset) * 0.90)

    # Scale the data (Normalise Data) - This creates new values that maintain the general distribution
    # and ratios of the source data, while keeping values within a scale applied across all numeric
    # columns used in the model.

    # set the scaler to scale values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # scale stock closing price into the values between 0 and 1
    scaled_data = scaler.fit_transform(close_dataset)
    # Create the training data sets
    # Create the scaled training data set from the first 90% of overall data
    scaled_training_data = scaled_data[0:training_data_length, :]

    # Split the data into independent and dependent training variables

    # independent training variables
    training_set_a = []

    # dependent training variables (value to predict)
    training_set_b = []

    # Populate training sets
    # The model uses past 30 data values(training_set_a) to predict each training_set_b value

    for i in range(30, len(scaled_training_data)):
        # Append past 30 values to the training_set_a data set9
        training_set_a.append(scaled_training_data[i-30:i, 0])
        # append 31st value that the model is to predict
        training_set_b .append(scaled_training_data[i, 0])
        # print data just to check the values
        if i <= 30:
            print(training_set_a)
            print(training_set_b)
            print()
    # Convert training_set_a and training_set_b data sets to numby arrays
    # So that they can be used by the LSTM
    training_set_a, training_set_b = np.array(
        training_set_a), np.array(training_set_b)
    # Reshape the data: Since LSTM expects 3d data model set (currently is 2d: training_set_a and training_set_b)
    # The LSTM expected input is: number of samples, number of steps and number of features
    # training_set_a.shape[0] - number of data rows(samples)
    # training_set_a.shape[1] - number of columns( number of steps) i.e last 60 days used to predict next day price
    # number of features = 1 since we want only one predicted price

    training_set_a = np.reshape(
        training_set_a, (training_set_a.shape[0], training_set_a.shape[1], 1))
    # Build the LSTM model
    model = Sequential()

    # Add layers to the model
    # First LSTM layer with 100 neurons that takes input and returns sequences into another LSTM layer
    model.add(LSTM(100, return_sequences=True,
              input_shape=(training_set_a.shape[1], 1)))
    # Second and last LSTM layer with 50 neurons, doesnt return sequences
    model.add(LSTM(50, return_sequences=False))
    # Dense layer with 10 neurons
    model.add(Dense(10))
    # final layer with the result of prediction
    model.add(Dense(1))
    # Compile the model, adding the optimizer and loss function
    # optimizer: an algorithm/method used to minimize an error functions or to maximize the efficiency of training
    #           Main goal is to improve upon the loss function.
    # loss function is used to measure how well the model did on training
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model:
    # The batch size is a number of samples processed before the model is updated
    # The epoch is the number of complete passes through the training dataset
    # validation_split - The part of the data used to validate the model: in this case is 15%
    model.fit(training_set_a, training_set_b, batch_size=16,
              epochs=32, validation_split=0.15)
    # Create the testing data set that is 10% of overall data
    test_data = scaled_data[training_data_length-30:, :]
    # Create testing data sets
    # testing_set_a will contain first 30 testing values used to predict next value
    testing_set_a = []
    # testing_set_b will contain the value that is to be predicted
    testing_set_b = close_dataset[training_data_length:, :]

    for i in range(30, len(test_data)):
        # append past 30 values
        testing_set_a.append(test_data[i-30:i, 0])
    # Convert the testing_set_a to a numpy array
    testing_set_a = np.array(testing_set_a)

    # Reshape the testing_set_a into a 3d data model
    testing_set_a = np.reshape(
        testing_set_a, (testing_set_a.shape[0], testing_set_a.shape[1], 1))
    # Use model to predict the values using testing_set_a data. If the values correspond to the testing_set_b
    # then the predictions of the model are accurate.
    predictions = model.predict(testing_set_a)
    # unscale the values into actual USD values
    predictions = scaler.inverse_transform(predictions)
    # Check the model accuracy by taking the Root Mean Squared Error (RMSE)
    # The lower the RMSE value, the better the fit and accuracy of the model is
    rmse = np.sqrt(np.mean(predictions - testing_set_b)**2)
    # Calculate the absolute percentage error for the predictions
    mape = mean_absolute_percentage_error(testing_set_b, predictions)
    # Plot the data
    # training data set
    training_data_set = closing_data[0:training_data_length]
    # validation data set
    validation_data_set = closing_data[training_data_length:]
    # predicted values data set
    validation_data_set['Predictions'] = predictions

    # Create graph objects

    trainingSet = go.Scatter(
        x=training_data_set.index,
        y=training_data_set['Close'],
        mode='lines',
        connectgaps=True,
        name="Training Data"
    )
    validationSet = go.Scatter(
        x=validation_data_set.index,
        y=validation_data_set['Close'],
        mode='lines',
        connectgaps=True,
        name="Actual Values"
    )
    testingSet = go.Scatter(
        x=validation_data_set.index,
        y=validation_data_set['Predictions'],
        mode='lines',
        connectgaps=True,
        name="Predicted Values"
    )
    # set graph layout
    graphLayout = go.Layout(
        title="Model",
        xaxis={'title': "Date"},
        yaxis={'title': "Closing Price USD"},
        autosize=False,
        width=900,
        height=600
    )
    # Create graph based on the above data
    model_testing_figure = go.Figure(
        data=[trainingSet, testingSet, validationSet], layout=graphLayout)
    # Use the model to predict tomorrows closing price of a given stock
    # Get the stock data
    figure = model_testing_figure
    print(figure)
    stock_quote = web.DataReader(
        stock_ticker, data_source='yahoo', start='2014-01-01', end=tomorrow)
    # filter the stock data to contain only closing stock price
    closing_prices = stock_quote.filter(['Close'])
    # Get the last 30 day closing price values
    last_30_values = closing_prices[-30:].values
    # Scale the data to be values between 0 and 1
    last_30_values_scaled = scaler.transform(last_30_values)
    # create list and append the past 30 scaled values
    data_set = [last_30_values_scaled]
    # Convert data_set to a numpy array
    data_set = np.array(data_set)
    # Reshape the data so it can be used by the model
    X_test = np.reshape(data_set, (data_set.shape[0], data_set.shape[1], 1))
    # Predict the next day closing price
    pred_price = model.predict(data_set)
    # Undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    # Display stock chart together with the predicted value for tomorrow closing price.

    # stock data
    datasets = go.Scatter(
        x=closing_prices.index,
        y=closing_prices['Close'],
        mode='lines',
        connectgaps=True,
        name="Past Stock Price"
    )
    # predicted price
    predicted_closing_price = go.Scatter(
        x=[tomorrow],
        y=pred_price[0],
        mode='markers',
        name="Predicted Next Day Closing Price"
    )

    # create graph
    prediction_figure = go.Figure(
        data=[datasets, predicted_closing_price], layout=graphLayout)
    return pred_price[0][0]
