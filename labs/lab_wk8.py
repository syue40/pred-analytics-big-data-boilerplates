from sklearn.linear_model import RidgeClassifier
from pycaret.utils import enable_colab
enable_colab()

from pycaret.datasets import get_data
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Show all columns in same row.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

dataset = get_data('credit')
print(dataset.T) # Transpose for a reader-friendly display

# Redo this for test_train_split
data        = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

ridge = RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=123, solver='auto',
                tol=0.001)

tuned_ridge = tune_model(ridge)
