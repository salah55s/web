import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy
import pandas as np
import seaborn as sns

from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
#df.head()

#use required features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)

X = cdf.iloc[:, :3]
Y = cdf.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(
     X,Y, test_size=0.2, random_state=0)

regressor = LinearRegression()

regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)


train_prediction = regressor.predict(X_test)
train_prediction2 = regressor.predict(X_train)
print("r_square score (train dataset): ", r2_score(y_train,train_prediction2))
# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))






