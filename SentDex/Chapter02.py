import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle
quandl.ApiConfig.api_key = '76eCnz6z9XTH8nfLWeQU'

# Config
isLoadFromLocal = True

# Loading data
if isLoadFromLocal:
    df = pickle.load(open("DataFromQuandl_Stock_Chap2.pickle", "rb"))
else:
    df = quandl.get('WIKI/GOOGL')
    pickle.dump(df, open("DataFromQuandl_Stock_Chap2.pickle", "wb+"))

# Data pre-processing
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecastCol = 'Adj. Close'
df.fillna('-99999', inplace = True)
forecastOut = int(math.ceil(0.0002*len(df)))
df['label'] = df[forecastCol].shift(-forecastOut)

x = np.array(df.drop(['label'],1))
x = x[:-forecastOut]
xLately = x[:-forecastOut]
df.dropna(inplace = True)
y = np.array(df['label'])

x = preprocessing.scale(x)

# Regression
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1)

# classifier = svm.SVR(kernel='linear') # SVM SVR

classifier = LinearRegression(n_jobs=3) # Linear Regression

classifier.fit(x_train, y_train)

accuracy = classifier.score(x_test, y_test)

print(classifier.predict(xLately))

print(accuracy)

