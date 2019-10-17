#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from numpy import concatenate
import matplotlib.pyplot as plt


# In[161]:


df = pd.read_csv('GoogleStocks.csv',index_col="date",parse_dates=True)
df = df.iloc[1:]
df = df.convert_objects(convert_numeric=True)
col = df.loc[: , "high":"low"]
opencol=  df.loc[: , "open"]
df=df.drop(columns=['high','low','close','open'])
df['average'] = col.mean(axis=1)


# In[162]:


featurescaler = MinMaxScaler(feature_range = (0,1))
df = featurescaler.fit_transform(df)
opencol = np.array(opencol)
opencol = opencol.reshape(-1,1)
openfeaturescaler = MinMaxScaler(feature_range = (0,1))
opencol = openfeaturescaler.fit_transform(opencol)


# In[163]:


data,testdata=np.split(df,[int(.8*len(df))])
opendata,opentest =np.split(opencol,[int(.8*len(opencol))])


# In[164]:


opendata = np.array(opendata)


# # Part 1

# In[165]:


testdata=np.array(testdata)
data=np.array(data)


# In[166]:


timesteps = [20,50,75]
hiddenlayercells = [30,50,80]


# # 2 Hidden Layers

# In[167]:


for ts in timesteps:
    X = []
    Y = []
    Xtest=[]
    Ytest=[]
    for i in range(ts, 604):
        X.append(data[i-ts:i,:])
        Y.append(opendata[i])
    for i in range(ts, 151):
        Xtest.append(testdata[i-ts:i,:])
        Ytest.append(opentest[i])

    X=np.array(X)
    Y=np.array(Y)
    Xtest=np.array(Xtest)
    Ytest=np.array(Ytest)
    for hlc in hiddenlayercells:
        model = Sequential()
        model.add(LSTM(hlc,return_sequences = True,input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(hlc,return_sequences = False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        fitmodel = model.fit(X, Y, epochs=50, batch_size=73, validation_data=(Xtest, Ytest), verbose=0, shuffle=False)
        
        title = "Stock Price Prediction for: (" + str(2) +", "+ str(hlc)+ ", " + str(ts)+ ")"
        print(title)
        
        predicted_price = model.predict(Xtest)
        inv_yhat = openfeaturescaler.inverse_transform(predicted_price)
        real_stock_price = openfeaturescaler.inverse_transform(Ytest)

        plt.plot(real_stock_price,color = 'red', label = 'Real Price')
        plt.plot(inv_yhat, color = 'blue', label = 'Predicted Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()
# plot history


# # 3 Hidden Layers

# In[169]:


for ts in timesteps:
    X = []
    Y = []
    Xtest=[]
    Ytest=[]
    for i in range(ts, 604):
        X.append(data[i-ts:i,:])
        Y.append(opendata[i])
    for i in range(ts, 151):
        Xtest.append(testdata[i-ts:i,:])
        Ytest.append(opentest[i])

    X=np.array(X)
    Y=np.array(Y)
    Xtest=np.array(Xtest)
    Ytest=np.array(Ytest)
    for hlc in hiddenlayercells:
        model = Sequential()
        model.add(LSTM(hlc,return_sequences = True,input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(hlc,return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(hlc,return_sequences = False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        fitmodel = model.fit(X, Y, epochs=50, batch_size=73, validation_data=(Xtest, Ytest), verbose=0, shuffle=False)
        
        title = "Stock Price Prediction for: (" + str(2) +", "+ str(hlc)+ ", " + str(ts)+ ")"
        print(title)
        
        predicted_price = model.predict(Xtest)
        inv_yhat = openfeaturescaler.inverse_transform(predicted_price)
        real_stock_price = openfeaturescaler.inverse_transform(Ytest)

        plt.plot(real_stock_price,color = 'red', label = 'Real Price')
        plt.plot(inv_yhat, color = 'blue', label = 'Predicted Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()
# plot history


# In[ ]:




