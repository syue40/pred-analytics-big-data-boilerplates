# from keras.layers import Dense
# from sklearn.metrics import accuracy_score
# from numpy import argmax
# from sklearn.datasets import make_blobs
# from keras.models import Sequential
# from sklearn.model_selection import train_test_split
# import numpy as np
# import tensorflow as tf
# from keras.utils import to_categorical
# # fit model on dataset
# def fitModel(trainX, trainy):
#     # define model
#     model = Sequential()
#     model.add(Dense(15, input_dim=4, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#     # fit model
#     model.fit(trainX, trainy, epochs=200, verbose=0)
#     return model


# from sklearn.preprocessing   import LabelEncoder
# import pandas as pd

# def getData():
#         PATH = "/Users/pm/Desktop/DayDocs/data/"
#         df = pd.read_csv('iris_old.csv')
#         df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']

#         # Convert text to numeric category.
#         # 0 is setosa, 1 is versacolor and 2 is virginica
#         df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

#         # Prepare the data.
#         X = df[['Sepal L', 'Sepal W', 'Petal L', 'Petal W']]
#         y = df['y']
#         ROW_DIM = 0
#         COL_DIM = 1

#         x_array = X.values
#         x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
#                                           x_array.shape[COL_DIM])

#         y_array = y.values
#         y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM], 1)

#         trainX, testX, trainy, testy = train_test_split(x_arrayReshaped,
#                                                         y_arrayReshaped, 
#                                                         test_size=0.33)
#         trainy = to_categorical(trainy)
#         return trainX, testX, trainy, testy


# def buildAndEvaluateIndividualModels():
#     trainX, testX, trainy, testy = getData()
#     NUM_MODELS  = 11
#     yhats       = []
#     scores      = []
#     models      = []
#     print("\n**** Single model results:")
#     for i in range(0, NUM_MODELS):
#         model                   = fitModel(trainX, trainy)
#         models.append(model)
#         predictions             = model.predict(testX)
#         yhats.append(predictions)

#         # Converts multi-column prediction set back to single column
#         # so accuracy score can be calculated.
#         singleColumnPredictions = argmax(predictions, axis=1)
#         accuracy = accuracy_score(singleColumnPredictions, testy)
#         scores.append(accuracy)
#         print("Single model " + str(i) + "   accuracy: " + str(accuracy))

#     print("Average model accuracy:      " + str(np.mean(scores)))
#     print("Accuracy standard deviation: " + str(np.std(scores)))
#     return models
# models = buildAndEvaluateIndividualModels()


# # Evaluate ensemble
# def buildAndEvaluateEnsemble(models):
#     scores = []
#     print("\n**** Ensemble model results: ")
#     for trial in range(0, 11):
#         # Generate new test data.
#         _, testX, _, testy = getData()

#         yhats  = []
#         # Get predictions with pre-built models.
#         for model in models:
#             predictions = model.predict(testX)
#             yhats.append(predictions)

#         # Sum predictions for all models.
#         # [[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]...], # Model 1 results
#         #  [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]...], # Model 2 results
#         #  [0.2, 0.2, 0.6], [0.3, 0.3, 0.4]...], # Model 3 results
#         # Becomes
#         # [[0.7, 0.8, 1.5],[0.7, 0.7, 1.6]...] # Summed results
#         summed = np.sum(yhats, axis=0)

#         # Converts multi-column prediction set back to single column
#         # so accuracy score can be calculated. For example;
#         # [[0.7, 0.8, 1.5],[0.7, 0.7, 1.6]...]
#         # Becomes
#         # [2, 2,....]
#         singleColumnPredictions = argmax(summed, axis=1)

#         accuracy = accuracy_score(singleColumnPredictions, testy)
#         scores.append(accuracy)
#         print("Ensemble model accuracy during trial " + str(trial) +\
#               ": " + str(accuracy))

#     print("Average model accuracy:      " + str(np.mean(scores)))
#     print("Accuracy standard deviation: " + str(np.std(scores)))

# buildAndEvaluateEnsemble(models)
from keras.models     import Sequential
from keras.layers     import Dense
from os               import makedirs
from os import path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
import numpy as np

PATH = './models/'

# fit model on dataset
def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(25, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=500, verbose=0)
    return model

from sklearn.preprocessing   import LabelEncoder
def generateData():
    PATH = "/Users/pm/Desktop/DayDocs/data/"
    df = pd.read_csv('iris_old.csv')
    df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']

    # Convert text to numeric category.
    # 0 is setosa, 1 is versacolor and 2 is virginica
    df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

    # Prepare the data.
    X = df[['Sepal L', 'Sepal W', 'Petal L', 'Petal W']]
    y = df['y']

    # split into train and test
    trainX, tempX, trainy, tempy = train_test_split(X, y, test_size=0.6)
    testX, valX, testy, valy = train_test_split(tempX, tempy, test_size=0.5)
    return trainX, testX,  valX, trainy, testy, valy


def generateModels(trainX, trainy):
    # create directory for models
    if(not path.exists(PATH)):
        makedirs('./models')

    # fit and save models
    numModels = 5
    print("\nFitting models with training data.")
    for i in range(numModels):
        # fit model
        model = fit_model(trainX, trainy)
        # save model
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        model.save(filename)
        print('>Saved %s' % filename)

trainX, testX,  valX, trainy, testy, valy = generateData()

# one hot encode output variable
trainy = to_categorical(trainy)
generateModels(trainX, trainy)

# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = PATH + 'model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of models
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

#trainX, testX, trainy, testy = generateData()

# load all models
numModels = 5
models    = load_all_models(numModels)
print('Loaded %d models' % len(models))

print("\nEvaluating single models with validation data.")
# evaluate standalone models on test dataset
# individual ANN models are built with one-hot encoded data.
for model in models:
    oneHotEncodedY = to_categorical(valy)
    _, acc = model.evaluate(valX, oneHotEncodedY, verbose=0)
    print('Model Accuracy: %.3f' % acc)

# Make predictions with the stacked model
def stacked_prediction(models, stackedModel, inputX):
    # create dataset using ensemble
    stackedX = getStackedData(models, inputX)
    # make a prediction
    yhat = stackedModel.predict(stackedX)
    return yhat

# fit a model based on the outputs from the ensemble models
def fit_stacked_model(models, inputX, inputy):
    # create dataset using ensemble
    stackedX = getStackedData(models, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model

# fit stacked model using the ensemble
# Stacked model build with LogisticRegression.
# y for LogisticRegression is not one-hot encoded.
print("\nFitting stacked model with test data.")
stackedModel = fit_stacked_model(models, testX, testy)

# evaluate model on test set
print("\nEvaluating stacked model with validation data.")
yhat = stacked_prediction(models, stackedModel, valX)
acc  = accuracy_score(valy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)
