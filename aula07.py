#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot

# %%
print(sm.datasets.sunspots.NOTE)

#%%

dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]
#print(dta)
dta.plot(figsize=(12, 8))

# %%

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
# Onda sinoidal amortecida na acf


#%%

arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()
print(arma_mod20.params)

#%%
arma_mod30 = ARIMA(dta, order=(3, 0, 0)).fit()
print(arma_mod30.params)

#%%
print('arma 2, 0, 0:')
print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)
print('arma 3, 0, 0:')
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)

# %%

#Nosso modelo obedece à teoria?

sm.stats.durbin_watson(arma_mod30.resid.values)
# A hipótese nula do teste de é que não há autocorrelação serial nos resíduos.
# Essa estatística sempre estará entre 0 e 4. Quanto mais próxima de 0 a estatística, maior será a evidência de correlação serial positiva. Quanto mais próximo de 4, mais evidências de correlação serial negativa.

# %% Como fica a serie sem o afeito da auto correlação?

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax)
resid = arma_mod30.resid

# %%

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# %% predicao

predict_sunspots = arma_mod30.predict("1990", "2012", dynamic=True)
print(predict_sunspots)

# %%
arma_mod30.summary()

# %% 
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
# .SARIMAX(endog, exog = None, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend=None

modsaz = sm.tsa.statespace.SARIMAX(dta, trend='n', order=(2,0,0), seasonal_order=(1, 0, 0, 9)).fit()

# %%

print('arma 3, 0, 0:')
print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
print('SARIMA 3, 0, 0:')
print(modsaz.aic, modsaz.bic, modsaz.hqic)

# %%
print(modsaz.summary())



# %%
