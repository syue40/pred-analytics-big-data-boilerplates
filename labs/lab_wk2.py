# from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Dense

import numpy  as np
import pandas as pd

# # Load the flower feature data into a DataFrame.
# df   = pd.DataFrame(columns=['Length', 'Width', 'IsRed'])
# data = [
#     {'Length':3,    'Width':1.5, 'IsRed': 1},
#     {'Length':2,    'Width':1,   'IsRed': 0},
#     {'Length':4,    'Width':1.5, 'IsRed': 1},
#     {'Length':3,    'Width':1,   'IsRed': 0},
#     {'Length':3.5,  'Width':.5,  'IsRed': 1},
#     {'Length':2,    'Width':.5,  'IsRed': 0},
#     {'Length':5.5,  'Width':1,   'IsRed': 1},
#     {'Length':1,    'Width':1,   'IsRed': 0}]

# for i in range(0, len(data)):
#     df = df.append(data[i], ignore_index=True)
# print(df)

# ROW_DIM = 0
# COL_DIM = 1

# # Convert DataFrame columns to vertical columns of features (as mentioned earlier).
# dfX = df.iloc[:, 0:2]
# ROW_DIM = 0
# COL_DIM = 1

# x_array = dfX.values
# x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
#                                   x_array.shape[COL_DIM])

# # Convert DataFrame columns to vertical columns of target variables values.
# dfY = df.iloc[:, 2:3]
# y_array = dfY.values
# y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM],
#                                   y_array.shape[COL_DIM])

# # Build a network model of sequential layers.
# model = Sequential()

# # Add 1st hidden layer. Note 1st hidden layer also receives data from input layer.
# # The input array must contain two feature columns and any number of rows.
# model.add(Dense(10, activation='sigmoid',
#                 input_shape=(x_arrayReshaped.shape[COL_DIM],)))


# # Add output layer.
# model.add(Dense(1,  activation='sigmoid'))

# # Compile the model.
# # Binary cross entropy is used to measure error cost for binary predictions.
# model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# # Fit the model
# # An epoch is one iteration for all samples through the network. 
# # verbose can be set to 1 to show detailed output during training.
# model.fit(x_arrayReshaped, y_arrayReshaped, epochs=1000, verbose=1)

# # Evaluate the model
# loss, acc = model.evaluate(x_arrayReshaped, y_arrayReshaped, verbose=0)
# print('Test Accuracy: %.3f' % acc)

# # Make a prediction
# row = [4.5,1]
# yhat = model.predict([row])
# print('Predicted: %.3f' % yhat)

import pandas                as     pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from tensorflow.keras        import Sequential
from tensorflow.keras.layers import Dense

PATH       = "/Users/pm/Desktop/DayDocs/data/"
df         = pd.read_csv('iris_old.csv')
df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']
print(df)

# Convert text to numeric category.
# 0 is setosa, 1 is versacolor and 2 is virginica
df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

# Prepare the data.
dfX = df.iloc[:, 0:4] # Get X features only from columns 0 to 3
dfY = df.iloc[:, 5:6] # Get X features only from column 5

ROW_DIM = 0
COL_DIM = 1

x_array = dfX.values
x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                  x_array.shape[COL_DIM])

# Convert DataFrame columns to vertical columns of target variables values.
y_array = dfY.values
y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM],
                                  y_array.shape[COL_DIM])

# Build a network model of sequential layers.
model = Sequential()

# Add 1st hidden layer. Note 1st hidden layer also receives data from input layer.
# The input array must contain two feature columns and any number of rows.
model.add(Dense(12, activation='sigmoid',
                input_shape=(x_arrayReshaped.shape[COL_DIM],)))


# Add output layer.
model.add(Dense(3,  activation='softmax'))

# Compile the model.
# Binary cross entropy is used to measure error cost for binary predictions.
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
# An epoch is one iteration for all samples through the network. 
# verbose can be set to 1 to show detailed output during training.
model.fit(x_arrayReshaped, y_arrayReshaped, epochs=1000, verbose=1)

# Evaluate the model
loss, acc = model.evaluate(x_arrayReshaped, y_arrayReshaped, verbose=0)
print('Test Accuracy: %.3f' % acc)

# Make a prediction
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print(f'Predicted: {yhat}')