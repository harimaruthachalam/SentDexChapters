import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle
import datetime
from matplotlib import style
import matplotlib.pyplot as plot

# Config
isLoadFromLocal = True
quandl.ApiConfig.api_key = '76eCnz6z9XTH8nfLWeQU'
style.use('ggplot')


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
forecastOut = int(math.ceil(0.01*len(df)))
df['label'] = df[forecastCol].shift(-forecastOut)

# df['label'].plot()
# df[forecastCol].plot()
# plot.legend(loc = 4)
# plot.show()

x = np.array(df.drop(['label'], 1))
print(x)
x = preprocessing.scale(x)
print(x)
xLately = x[-forecastOut:]
x = x[:-forecastOut]
df.dropna(inplace = True)
y = np.array(df['label'])


# Regression
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1)

# classifier = svm.SVR(kernel='linear') # SVM SVR

classifier = LinearRegression(n_jobs=3) # Linear Regression
classifier.fit(x_train, y_train)
accuracy = classifier.score(x_test, y_test)
forecastSet = classifier.predict(xLately)

print('Accuracy is ', accuracy, '\nForecasted values are ', forecastSet, '\nNumber of values is ', forecastOut)

df['Forecast'] = np.nan
lastDate = df.iloc[-1].name
print(lastDate)
lastTime = lastDate.timestamp()
print(lastTime)
oneDay = 24 * 60 * 60 # seconds in a day
nextTime = lastTime + oneDay

for iter in forecastSet:
    nextDate = datetime.datetime.fromtimestamp(nextTime)
    nextTime += oneDay
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns) - 1)] + [iter]

df['Adj. Close'].plot()
df['Forecast'].plot()
plot.legend(loc = 4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()