
# coding: utf-8

# # LSTM For Time Series
# 
# We will first start with the basic example of sine series and will then extend the concepts to Stock Market

# In[8]:

import numpy as np
import time
import matplotlib.pyplot as plt
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
sine_wave_path = '/home/hrituraj/Desktop/IOP/sinwave.csv'


# In[9]:

def load_data(filepath, seq_len, normalise_window):
    f = open(filepath, 'rb').read()
    data = f.decode().split('\n')
    
    sequence_length = seq_len + 1
    result = []
    
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
        
    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) 
    return [x_train, y_train, x_test, y_test]


# In[10]:

def build_model(layers):
    model = Sequential()
    
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


# In[11]:

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


# In[12]:

epochs  = 1
seq_len = 50

print('> Loading data... ')
X_train, y_train, X_test, y_test = load_data(sine_wave_path, seq_len, True)
print('> Data Loaded. Compiling...')

model = build_model([1, 50, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

predicted = predict_point_by_point(model, X_test)


# In[13]:

plt.plot(predicted)
plt.show()


# In[ ]:



