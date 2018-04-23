from statistics import mean
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
import random

def generateData(numberOfPoints, variance, step = 1, correlation = False):
    start = 1
    valueX = []
    valueY = []
    for iter in range(numberOfPoints):
        valueX.append(iter)
        valueY.append(random.randrange(-variance, variance) + start)
        if correlation and correlation == 'positive':
            start += step
        elif correlation and correlation == 'negative':
            start -= step
    return valueX, valueY

x, y = generateData(50, 150, 20, 'negative')
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

style.use('fivethirtyeight')
plot.scatter(x, y)

def best_fit(x, y):
    m = ((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) ** 2) - mean(x ** 2))
    c = mean(y) - m * mean(x)
    return m, c


m, c = best_fit(x, y)
print(m,c)

regressionLine = [(m * x_ + c) for x_ in x]
# The above equation is equivalent to the following
# for x_ in x:
#     regressionLine.append(m*x_ + c)

predictX = 10
predictY = m * predictX + c

plot.scatter(predictX, predictY, s=100, color='r')

plot.plot(x, regressionLine, color='g')
plot.show()