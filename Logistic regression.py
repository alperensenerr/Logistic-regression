import numpy as np
import matplotlib as plt
import pandas as pd

veriler =pd.read_csv('veriler.csv')

data = veriler.iloc[:,1:4].values
cinsiyet = veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split

data_train, data_test, cinsiyet_train, cinsiyet_test = train_test_split(data,cinsiyet,test_size=0.33,random_state=0)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(data_train,cinsiyet_train)

y_pred = logr.predict(data_test)

print(y_pred)
print(cinsiyet_test)
