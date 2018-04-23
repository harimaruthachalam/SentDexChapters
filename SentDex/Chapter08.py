# We are going to implement a linear regression with degree one.
# Simple equation would be y = mx + c
#
# Formula to find m,
#       mean(x).mean(y) - mean(xy)
#  m = ----------------------------
#       sqr(mean(x)) - mean(sqr(x))
#
# Formula to find c,
#  c = mean(y) - m.mean(x)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style

x = [1,2,3,4,5,6,7,8]
y = [9,8,5,5,3,3,3,2]
x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

style.use('fivethirtyeight')
plot.scatter(x, y)
# plot.show()
# plot.hold()

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

plot.scatter(predictX, predictY, edgecolors='g')

plot.plot(x, regressionLine)
plot.show()