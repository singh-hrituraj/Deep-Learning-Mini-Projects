
# coding: utf-8

# In[52]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.layers import Dense, LSTM
from keras.models import Sequential
import numpy as np


# In[7]:

def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

Dataset = pd.read_csv('/home/hrituraj/Desktop/IOP/sales-of-shampoo-over-a-three-ye.csv',skipfooter = 2,index_col = 0, engine = 'python', parse_dates = [0], date_parser = parser )


# In[8]:

plt.close('all')
Dataset.plot()
plt.show()



# In[9]:

X = Dataset.values
train, test = X[0:-12], X[-12:]


# In[10]:

def baseline_model(train, test):
    history = [x for x in train]
    predictions = []
    
    for i in xrange(len(test)):
        #make predictions = last output
        predictions.append(history[-1])
        
        #update history
        history.append(test[i])
        
    return predictions


predictions = baseline_model(train, test)
rmse = sqrt(mean_squared_error(test, predictions))

print "Root Mean Squared Error : {}".format(rmse)

plt.plot(test)
plt.plot(predictions)
plt.show()       


# In[13]:

def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# In[14]:

df = timeseries_to_supervised(X)
df


# In[15]:

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# In[16]:

diff = difference(X)
diff.head()


# In[17]:

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# In[28]:

inverted = list()
for i in range(len(diff)):
    value = inverse_difference(X, diff[i], len(Dataset)-i)
    inverted.append(value)
inverted = pd.Series(inverted)
print(inverted.head())


# In[29]:

from sklearn.preprocessing import MinMaxScaler
X = X.reshape(len(X), 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X)
scaled_X = scaler.transform(X)


# In[30]:


inverted_X = scaler.inverse_transform(scaled_X)


# In[33]:

X, y = train[:, 0:-1], train[:, -1]
X = X.reshape(X.shape[0], 1, X.shape[1])


# In[50]:

def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# In[35]:

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    
    for i in range(nb_epoch):
        model.fit(X, y, epochs = 1, batch_size = batch_size, shuffle = False)
        model.reset_states()
    
    return model

        


# In[36]:

def forecast_LSTM(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    
    return yhat[0,0]


# In[45]:

series = pd.read_csv('/home/hrituraj/Desktop/IOP/sales-of-shampoo-over-a-three-ye.csv', skipfooter = 2,  parse_dates=[0], index_col=0,engine= 'python', date_parser=parser)
series


# In[46]:

raw_values = series.values
diff_values = difference(raw_values, 1)


# In[47]:

supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values


# In[48]:

train, test = supervised_values[0:-12], supervised_values[-12:]


# In[51]:

scaler, train_scaled, test_scaled = scale(train, test)


# In[53]:

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# In[ ]:

lstm_model = fit_lstm(train_scaled, 1, 3000, 4)


# In[ ]:



