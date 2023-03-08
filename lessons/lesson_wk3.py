import  pandas as pd
from    sklearn.model_selection import train_test_split
PATH    = "/Users/pm/Desktop/DayDocs/data/"
from    keras.models import Sequential
from    keras.layers import Dense
import  matplotlib.pyplot as plt
import tensorflow as tf
# load the dataset
df = pd.read_csv('fluDiagnosis.csv')
# split into input (X) and output (y) variables
print(df)

X = df[['A','B']]
y = df[['Diagnosed']]
# Split into train and test data sets.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33)

def buildModel(num_nodes):
    # define the keras model
    model = Sequential()
    model.add(Dense(num_nodes, input_dim=2, activation='relu',
                    kernel_initializer='he_normal'))
    
    NUM_LAYERS=11
    for i in range(0, NUM_LAYERS-1):
        model.add(Dense(num_nodes, activation='relu', kernel_initializer='he_normal'))
        
    model.add(Dense(1, activation='sigmoid'))

    opitimizer = tf.keras.optimizers.SGD(
        learning_rate=0.005, momentum=0.8, name="SGD",
    )

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                  metrics=['accuracy'])

    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=110, batch_size=81, validation_data=(X_test,
                        y_test))
    # evaluate the keras model

    # Evaluate the model.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: ' + str(acc) + ' Num nodes: ' + str(num_nodes))
    return history

def showLoss(history, numNodes):
    # Get training and test loss histories
    training_loss       = history.history['loss']
    validation_loss     = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history for training data.
    actualLabel = str(numNodes) + " nodes"
    plt.subplot(1, 2, 1)
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actualLabel)
    plt.legend()

def showAccuracy(history, numNodes):
    # Get training and test loss histories
    training_loss       = history.history['accuracy']
    validation_loss     = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 2)

    actualLabel = str(numNodes) + " nodes"
    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, label=actualLabel)
    plt.legend()

nodeCounts = [210]
plt.subplots(nrows=1, ncols=2,  figsize=(14,7))

for i in range(0, len(nodeCounts)):
    history = buildModel(nodeCounts[i])
    showLoss(history, nodeCounts[i])
    showAccuracy(history, nodeCounts[i])

plt.subplots(nrows=1, ncols=2,  figsize=(14,7))
batchSizes = [64, len(y_train)]
for i in range(0, len(batchSizes)):
    history = buildModel(batchSizes[i])
    showLoss(history, batchSizes[i])
    showAccuracy(history, batchSizes[i])


plt.show()
