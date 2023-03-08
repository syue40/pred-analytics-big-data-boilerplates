# Pandas is used for data manipulation
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from   sklearn.metrics import mean_squared_error
# Creating a DataFrame of given iris dataset.
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
iris['target_names']
print(data.head())

# Import train_test_split function
from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc_x            = StandardScaler()
X_train_scaled  = sc_x.fit_transform(X_train)
X_test_scaled = sc_x.transform(X_test)

from sklearn              import metrics
from sklearn.ensemble     import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def buildModelAndPredict(clf, X_train_scaled, X_test_scaled, y_train, y_test, title):
    print("\n**** " + title)
    #Train the model using the training sets y_pred=rf.predict(X_test)
    clf_fit = clf.fit(X_train_scaled,y_train)
    y_pred = clf_fit.predict(X_test_scaled)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # For explanation see:
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
    print(metrics.classification_report(y_test, y_pred, digits=3))

    # Predict species for a single flower.
    # sepal length = 3, sepal width = 5
    # petal length = 4, petal width = 2
    prediction = clf_fit.predict([[3, 5, 4, 2]])

    # 'setosa', 'versicolor', 'virginica'
    print(prediction)

lr = LogisticRegression(fit_intercept=True, solver='liblinear')
buildModelAndPredict(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(X_train_scaled, y_train)

# # Use the forest's predict method on the test data
# predictions = rf.predict(X_test_scaled)

# # Calculate the absolute errors
# errors = abs(predictions - y_test)

# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / y_test)

# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)

# print('Accuracy:', round(accuracy, 2), '%.')

# # Print out the mean square error.
# mse = mean_squared_error(y_test, predictions)
# print('RMSE:', np.sqrt(mse))

buildModelAndPredict(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest Regression")
