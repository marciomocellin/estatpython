import pandas as pd
import statistics as stat

cars = pd.read_csv('cars.csv')

stat.covariance(cars.speed, cars.dist)/(stat.stdev(cars.speed)*stat.stdev(cars.dist))

stat.correlation(cars.speed, cars.dist*(-1))

slope, intercept = stat.linear_regression(cars.speed, cars.dist*(-1))

round(slope * 30 + intercept)

test = stat.NormalDist(mu=700, sigma=10)

test.cdf(700)

from scipy.stats import f_oneway



