#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from tqdm import tqdm
import itertools
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


# In[105]:


df = pd.read_csv('GoogleStocks.csv')
df = df.iloc[1:]
df = df.convert_objects(convert_numeric=True)
df['date'] = pd.to_datetime(df['date'])


# In[106]:


def getfeatureddata(data):
    opencol = data['open'].values
    volcol = data['volume'].values
    highcol = data['high'].values
    lowcol = data['low'].values
    closecol = data['close'].values
    averagecol = (highcol + lowcol)/2
    
    feature1=(opencol-closecol)/closecol
    feature2=(averagecol-closecol)/closecol
    feature3=(closecol-volcol)/closecol
    
    feature1range=np.linspace(-0.1, 0.1, 20)
    feature2range=np.linspace(0, 0.1, 10)
    feature3range=np.linspace(0, 0.1, 10)
   
    featuredata = np.column_stack((feature1,feature2,feature3))
    
    combinationdata =np.array(list(itertools.product(feature1range, feature2range, feature3range)))
    
    return featuredata,combinationdata


# In[107]:


featuredata,_ = getfeatureddata(data)


# In[108]:


def dayprediction(data,day,ts,n):
#     print(ts,n)
    begin =day-ts
    if begin<0:
        begin=0
    
    end = day - 1
    if end<0:
        end=0
    
    trainx = data.iloc[begin:end]
    fetureddata,combinationdata = getfeatureddata(trainx)
    predictions =[]
    model = GaussianHMM(n_components=n)
    model.fit(featuredata)
    
    for comb in combinationdata:
        traindata = np.row_stack((fetureddata, comb))
        prediction = model.score(traindata)
        predictions.append(prediction)

    maxpred = np.argmax(predictions)    
    return combinationdata[maxpred]


# In[109]:


def predict(ts,n):
#     print(ts)
    data,testdata=np.split(df,[int(.8*len(df))])
    predicted = []
    
    for i in range(len(testdata)):
        close = testdata.iloc[i]['close']
        p,_,_ = dayprediction(data,i,ts,n)
        p = close*(1+p)
        predicted.append(p)

    return predicted


# In[110]:


ts = [20,50,75]
n= [4,8,12]
for num in n:
    for t in ts:
        predicted = predict(t,num)
        fig = plt.figure()
        dates = np.array(testdata['date'], dtype="datetime64[ms]")
        actual = testdata['open']

        axes = fig.add_subplot(111)
        axes.plot(dates, actual, 'bo-', label="actual")
        axes.plot(dates, predicted, 'r+-', label="predicted")

        fig.autofmt_xdate()

        plt.legend()
        plt.show()


# In[133]:


predictedhmm=predicted


# In[116]:


df = pd.read_csv('GoogleStocks.csv',index_col="date",parse_dates=True)
df = df.iloc[1:]
df = df.convert_objects(convert_numeric=True)
col = df.loc[: , "high":"low"]
opencol=  df.loc[: , "open"]
df=df.drop(columns=['high','low','close','open'])
df['average'] = col.mean(axis=1)

featurescaler = MinMaxScaler(feature_range = (0,1))
df = featurescaler.fit_transform(df)
opencol = np.array(opencol)
opencol = opencol.reshape(-1,1)
openfeaturescaler = MinMaxScaler(feature_range = (0,1))
opencol = openfeaturescaler.fit_transform(opencol)

data,testdata=np.split(df,[int(.8*len(df))])
opendata,opentest =np.split(opencol,[int(.8*len(opencol))])
opendata = np.array(opendata)
testdata=np.array(testdata)
data=np.array(data)


# In[126]:


ts=75
hlc=80

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


# In[167]:


fig = plt.figure()
predictedhmm = np.array(predictedhmm)
p =predictedhmm[-77:-1]
p= np.append(p,predictedhmm[-1])

plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(inv_yhat, color = 'blue', label = 'HMM')
plt.plot(p, color = 'green', label = 'RNN')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

