# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset. The prediction accuracy of the model will be assessed by comparing its output with the true stock prices from the test dataset.
Dataset: The dataset consists of two CSV files:

trainset.csv: This file contains historical stock price data of Google, which will be used for training the RNN model. It includes features such as the opening price of the stock.

testset.csv: This file contains additional historical stock price data of Google, which will be used for testing the trained RNN model. Similarly, it includes features such as the opening price of the stock.

Both datasets contain multiple columns, but for this task, only the opening price of the stock (referred to as 'Open') will be used as the feature for predicting future stock prices.

The objective is to build a model that can effectively learn from the patterns in the training data to make accurate predictions on the test data.

![image](https://github.com/RoopakCS/rnn-stock-price-prediction/assets/139228922/6ebf1686-7828-4bb2-b8f1-a48fdc000f7b)

## DESIGN STEPS

### STEP 1:

Import the necessary tensorflow modules

### STEP 2:

Load the stock dataset.

### STEP 3:

Fit the model and then predict.

## PROGRAM

**Name: Roopak C S**

**Register number: 212223220088**
## Importing modules
````python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
````
## Loading the training dataset
````python
dataset_train = pd.read_csv('trainset.csv')
````
## Reading only columns
````python
dataset_train.columns
````
## Displaying the first five rows of the dataset
````python
dataset_train.head()
````
## Selecting all rows and the column with index 1
````python
train_set = dataset_train.iloc[:,1:2].values
````
## Displaying the type of the training dataset
````python
type(train_set)
````
## Displaying the shape of the training dataset
````python
train_set.shape
````
## Scaling the dataset using MinMaxScaler
````python
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
````
## Displaying the shape of the scaled training data set
````python
training_set_scaled.shape
````
````python
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape
length = 60
n_features = 1
````
## Creating the model
````python
model = Sequential()
model.add(layers.SimpleRNN(50, input_shape=(length, n_features)))
model.add(layers.Dense(1))
````
## Compiling the model
````python
model.compile(optimizer='adam', loss='mse')
````
## Printing the summary of the model
````python
model.summary()
````
## Fitting the model
````python
model.fit(X_train1,y_train,epochs=100, batch_size=32)
````
## Reading the testing dataset
````python
dataset_test = pd.read_csv('testset.csv')
````
## Selecting all rows and the column with index 1
````python
test_set = dataset_test.iloc[:,1:2].values
````
## Displaying the shape of the testing data
````python
test_set.shape
````
## Concatenating the 'Open' columns from testing dataset and training dataset
````python
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
````
## Transforming the inputs
````python
inputs_scaled=sc.transform(inputs)
````
````python
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
````
## Plotting the graph between True Stock Price, Predicted Stock Price vs time
````python
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Roopak C S\n212223220088\nGoogle Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
````

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/RoopakCS/rnn-stock-price-prediction/assets/139228922/7eda16ef-7d2d-4c8e-8ac6-6a9044a90354)

### Mean Square Error

![image](https://github.com/RoopakCS/rnn-stock-price-prediction/assets/139228922/c6d4d855-dab6-4073-9ddc-a2034d8549c3)

## RESULT

Thus a Recurrent Neural Network model for stock price prediction is developed and implemented successfully.


