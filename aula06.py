import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Gaussian

algodao = pd.DataFrame({'percent' : [15, 20, 25, 30, 35, 
15, 20, 25, 30, 35, 
15, 20, 25, 30, 35, 
15, 20, 25, 30, 35, 
15, 20, 25, 30, 35 ],
'resist' : [7, 12, 14, 19, 7,
7, 17, 18, 25, 10,
15, 12, 18, 22, 11,
11, 18, 19, 19, 15,
9, 18, 19, 23, 11]})
algodao['intercept'] = 1


line_fit = sm.OLS(algodao['resist'], sm.add_constant(algodao['percent'], prepend=True)).fit()
print(line_fit.summary())

gauss = sm.GLM(algodao['resist'], algodao[['percent', 'intercept']], family=sm.families.Gaussian())
gauss_results = gauss.fit()
print(gauss_results.summary())

algodao['percent^2'] = np.power(algodao['percent'], 2)
gauss = sm.GLM(algodao['resist'], algodao[['percent', 'intercept', 'percent^2']], family=sm.families.Gaussian())
gauss_results = gauss.fit()
print(gauss_results.summary())

cypemethrin = pd.DataFrame({ 'Dose' : np.concatenate([
np.repeat(1, 20), np.repeat(2, 20), np.repeat(4, 20), np.repeat(8, 20), np.repeat(16, 20), np.repeat(32, 20),
  np.repeat(1, 20), np.repeat(2, 20), np.repeat(4, 20), np.repeat(8, 20), np.repeat(16, 20), np.repeat(32, 20)]),
'mortos' : np.concatenate([
np.repeat(1, 1) , np.repeat(0, 19),
  np.repeat(1, 4) , np.repeat(0, 16),
  np.repeat(1, 9) , np.repeat(0, 20-9),
  np.repeat(1, 13), np.repeat(0, 20-13),
  np.repeat(1, 18), np.repeat(0, 20-18),
  np.repeat(1, 20), np.repeat(0, 20-20),
  np.repeat(1, 0), np.repeat(0, 20-0),
  np.repeat(1, 2), np.repeat(0, 20-2),
  np.repeat(1, 6), np.repeat(0, 20-6),
  np.repeat(1, 10), np.repeat(0, 20-10),
  np.repeat(1, 12), np.repeat(0, 20-12),
  np.repeat(1, 16), np.repeat(0, 20-16)]),
'sexo' : np.concatenate([np.repeat('F', 20*6), np.repeat('M', 20*6)])})

cypemethrin['logDose'] = np.log2(cypemethrin.Dose)
cypemethrin['sexo'] = pd.get_dummies(data=cypemethrin['sexo'], drop_first=True)

glm_binom = sm.GLM(cypemethrin['mortos'], sm.add_constant(cypemethrin[['logDose', 'sexo']]), family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())

Quedas = pd.DataFrame({
'Individuo' : [1, 2, 3, 4, 5],
'Quedas' : [1, 1, 2, 0, 2],
'Intervencao' : [1, 1, 1, 1, 1],
'Sexo' : [0, 0, 1, 1, 0],
'Balanco' : [45, 62, 43, 76, 51],
'Forca' : [70, 66, 64, 48, 72]})

Quedas = pd.DataFrame(np.loadtxt('https://www.ime.usp.br/~giapaula/geriatra.dat', unpack = True).T, columns = ['Quedas'
,'Intervencao'
,'Sexo'
,'Balanco'
,'Forca'])


