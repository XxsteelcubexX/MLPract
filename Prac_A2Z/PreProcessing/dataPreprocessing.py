import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = '/home/piyush/Projects/MLPract/dataSets/data_A2Z/'
fileName = 'Data.csv'

dataSet = pd.read_csv(path+fileName)
dataSet

# seperating independent(Features) and dependet Varaibles

X = dataSet.iloc[:,:-1].values #.values converts the DataFrame to an array
X
Y = dataSet.iloc[:,-1].values
Y
## handling the missing Data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])
X

# Encoding Categorical Data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough') # passthrough will keep all the column on which transform is not applied
X = np.array(ct.fit_transform(X))
X
# Encoding the dependent Variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)
Y

# train test slpit

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)# random_state is added to remove randomness
X_train
X_test
Y_train
Y_test

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train[:,-2:] = sc.fit_transform(X_train[:,-2:])
X_train
X_test[:,-2:] = sc.transform(X_test[:,-2:]) # we didn't use fit_transform as we need to use scaler for train.
X_test
