from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sklearn_metric
from keras.layers.recurrent import LSTM
from tqdm import  tqdm
from keras.layers import Bidirectional
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import math
import time


def generateCandle(df,numberOfCandles):
    listDf = [df[i:i+numberOfCandles] for i in range(0,df.shape[0],numberOfCandles)]
    newData = []
    for df in tqdm(listDf):
        if len(df) != numberOfCandles:
            continue
        start = df['start'].values[0]
        end = df['end'].values[len(df)-1]
        high = max(df['high'].values)
        low = min(df['low'].values)
        volume = sum(df['volume'].values)
        time = df['time'].values[0]
        newData.append([time,start,end,high,low,volume])
    newData = pd.DataFrame(newData)
    newData.columns = ['time','start','high','low','end','volume']
    return newData

def getFutureDataset(df,future):
    tempdf = df[future:]
    df = df[0:len(tempdf)]
    df['future'] = tempdf['end'].values
    return df

def createLookBack(df, look_back=1,forecast = 1):
    train_x = []
    train_y = []
    for i in tqdm(range(len(df)-(forecast+look_back))):
        previousData = df[0+i:i+look_back+1]
        forecastData = df['end'].values[i+look_back+forecast]
        train_x.append(previousData.values)
        train_y.append(forecastData)
    return np.array(train_x),np.array(train_y)



def initializeModel(window_size, input_shape, dropout_value, activation_function, loss_function, optimizer):
    model = Sequential()
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, input_shape),))
    model.add(Dropout(dropout_value))
    model.add(Bidirectional(LSTM((window_size*2), return_sequences=True)))
    model.add(Dropout(dropout_value))
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    model.add(Dense(units=1))
    model.add(Activation(activation_function))
    model.compile(loss=loss_function, optimizer=optimizer)
    return model

def trainModel(model, X_train, Y_train, batch_num, num_epoch, val_split):
    model.fit(X_train, Y_train, batch_size= batch_num, nb_epoch=num_epoch, validation_split= val_split)
    return model


data = pd.read_csv('3yr4mon5min_bit2.csv')
columns = ['start','high','low','end','volume']

train = data[data['time'] < '2019-01-01']
test = data[data['time'] > '2019-01-01']

train.drop('time',axis = 1,inplace = True)
test.drop('time',axis = 1,inplace = True)
train_shape = train.values.shape
test_shape = test.values.shape
scaler = StandardScaler().fit(train.values.reshape(-1,1))

train_scaled = scaler.transform(train.values.reshape(-1,1))
train_scaled = train_scaled.reshape(train_shape)
train_scaled = pd.DataFrame(train_scaled,columns=columns)

test_scaled = scaler.transform(test.values.reshape(-1,1))
test_scaled = test_scaled.reshape(test_shape)
test_scaled = pd.DataFrame(test_scaled,columns=columns)

lookback = 50
forecast = 12

train_scaled = createLookBack(train_scaled,lookback,forecast)
test_scaled = createLookBack(test_scaled,lookback,forecast)
test = createLookBack(test,lookback,forecast)

X_train = train_scaled[0]
y_train = train_scaled[1]
X_test = test_scaled[0]
y_test = test_scaled[1]

model = initializeModel(lookback+1,5,0.2, 'linear', 'mse', 'adam')
model = trainModel(model,X_train,y_train,1024,10,0.2)

predicted_price = scaler.inverse_transform(model.predict(X_test).reshape(-1,1))
predicted_price = predicted_price.reshape(len(predicted_price))
original_price = test[1]
instance_price = test[0][:,-1,3]

predicted = list(predicted_price-instance_price)
original = list(original_price-instance_price)

for i in range(len(predicted)):
    if predicted[i] <= 0:
        predicted[i] = 0 
    else:
        predicted[i] = 1 

for i in range(len(original)):
    if original[i] <= 0:
        original[i] = 0 
    else:
        original[i] = 1


accuracy = sklearn_metric.accuracy_score(original,predicted)
precision = sklearn_metric.precision_score(original,predicted)
recall = sklearn_metric.recall_score(original,predicted)

accuracy,precision,recall