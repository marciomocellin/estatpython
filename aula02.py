import pandas as pd
import statistics as stat

cars = pd.read_csv('cars.csv')

stat.covariance(cars.speed, cars.dist)/(stat.stdev(cars.speed)*stat.stdev(cars.dist) )

stat.correlation(cars.speed, cars.dist)

slope, intercept = stat.linear_regression(cars.speed, cars.dist)

round(slope * 15 + intercept)
