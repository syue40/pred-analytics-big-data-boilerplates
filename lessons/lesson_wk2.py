import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from   sklearn               import metrics

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

PATH     = "/Users/pm/Desktop/DayDocs/data/"
FILE     = "heart_disease.csv"
data     = pd.read_csv(FILE)
x_data   = data.drop("target", axis=1)
y_values = data["target"]

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_values, test_size=0.3, random_state=42
)

# Stochastic gradient descent models are sensitive to differences
from sklearn.preprocessing import StandardScaler
scaler        = StandardScaler()
scaler.fit(X_train)
X_trainScaled = scaler.transform(X_train)
X_testScaled  = scaler.transform(X_test)

clf     = LogisticRegression(max_iter=1000)
clf.fit(X_trainScaled, y_train)
lr_pred = clf.predict(X_testScaled)

print("Accuracy:{} ".format(clf.score(X_testScaled, y_test) * 100))
print("Error Rate:{} ".format((1 - clf.score(X_testScaled, y_test)) * 100))

# Show confusion matrix and accuracy scores.
confusion_matrix = pd.crosstab(y_test, lr_pred,
                               rownames=['Actual'],
                               colnames=['Predicted'])

print('\nAccuracy: ',metrics.accuracy_score(y_test, lr_pred))
print("\nConfusion Matrix")
print(confusion_matrix)

COLUMN_DIMENSION = 1
#######################################################################
# Part 2
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# shape() obtains rows (dim=0) and columns (dim=1)
n_features = X_trainScaled.shape[COLUMN_DIMENSION]

#######################################################################
# Model tuning section.
from tensorflow.keras.optimizers import SGD #for adam optimizer

def create_model(numNeurons=35, initializer='uniform', activation='softmax'):
    # create model
    model = Sequential()

    model.add(Dense(numNeurons, kernel_initializer=initializer,  
              activation=activation))

    opt = SGD(lr=0.001)
    # Compile model
    model.compile(loss='mse', metrics=['accuracy'], optimizer=opt)
    return model



# summarize results
estimator = KerasRegressor(build_fn=create_model, epochs=100,
                           batch_size=60, verbose=1)
kfold   = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
print("Baseline RMSE: " + str(np.sqrt(results.std())))

# So then we build the model.
model = create_model()
history = model.fit(X_train, y_train, epochs=100,
                    batch_size=60, verbose=1,
                    validation_data=(X_test, y_test))



