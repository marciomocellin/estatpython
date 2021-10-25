import numpy as np
import pandas as pd
import statsmodels.api as sm

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

# >>> algodao.head()
#    percent  resist
# 0       15       7
# 1       20      12
# 2       25      14
# 3       30      19
# 4       35       7


line_fit = sm.OLS(algodao['resist'], sm.add_constant(algodao['percent'], prepend=True)).fit()
print(line_fit.summary())

# >>> print(line_fit.summary())
#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                 resist   R-squared:                       0.053
# Model:                            OLS   Adj. R-squared:                  0.012
# Method:                 Least Squares   F-statistic:                     1.282
# Date:                Mon, 25 Oct 2021   Prob (F-statistic):              0.269
# Time:                        18:47:47   Log-Likelihood:                -75.269
# No. Observations:                  25   AIC:                             154.5
# Df Residuals:                      23   BIC:                             157.0
# Df Model:                           1                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         10.9400      3.764      2.907      0.008       3.154      18.726
# percent        0.1640      0.145      1.132      0.269      -0.136       0.464
# ==============================================================================
# Omnibus:                        2.066   Durbin-Watson:                   1.932
# Prob(Omnibus):                  0.356   Jarque-Bera (JB):                1.176
# Skew:                          -0.166   Prob(JB):                        0.556
# Kurtosis:                       1.991   Cond. No.                         95.6
# ==============================================================================
# 
# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

gauss = sm.GLM(algodao['resist'], sm.add_constant(algodao['percent'], prepend=True), family=sm.families.Gaussian())
gauss_results = gauss.fit()
print(gauss_results.summary())

# >>> print(gauss_results.summary())
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                 resist   No. Observations:                   25
# Model:                            GLM   Df Residuals:                       23
# Model Family:                Gaussian   Df Model:                            1
# Link Function:               identity   Scale:                          26.232
# Method:                          IRLS   Log-Likelihood:                -75.269
# Date:                Mon, 25 Oct 2021   Deviance:                       603.34
# Time:                        18:55:24   Pearson chi2:                     603.
# No. Iterations:                     3   Pseudo R-squ. (CS):            0.05318
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         10.9400      3.764      2.907      0.004       3.563      18.317
# percent        0.1640      0.145      1.132      0.258      -0.120       0.448
# ==============================================================================

algodao['percent^2'] = np.power(algodao['percent'], 2)

# >>> algodao.head()
#    percent  resist  percent^2
# 0       15       7        225
# 1       20      12        400
# 2       25      14        625
# 3       30      19        900
# 4       35       7       1225

gauss = sm.GLM(algodao['resist'], sm.add_constant(algodao[['percent', 'percent^2']], prepend=True), family=sm.families.Gaussian())
gauss_results = gauss.fit()
print(gauss_results.summary())

# >>> print(gauss_results.summary())
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                 resist   No. Observations:                   25
# Model:                            GLM   Df Residuals:                       22
# Model Family:                Gaussian   Df Model:                            2
# Link Function:               identity   Scale:                          11.824
# Method:                          IRLS   Log-Likelihood:                -64.752
# Date:                Mon, 25 Oct 2021   Deviance:                       260.13
# Time:                        19:02:22   Pearson chi2:                     260.
# No. Iterations:                     3   Pseudo R-squ. (CS):             0.7227
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const        -39.9886      9.785     -4.087      0.000     -59.166     -20.811
# percent        4.5926      0.828      5.549      0.000       2.970       6.215
# percent^2     -0.0886      0.016     -5.388      0.000      -0.121      -0.056
# ==============================================================================



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

# >>> cypemethrin.head()
#    Dose  mortos sexo
# 0     1       1    F
# 1     1       0    F
# 2     1       0    F
# 3     1       0    F
# 4     1       0    F

cypemethrin['logDose'] = np.log2(cypemethrin.Dose)

# >>> cypemethrin.tail()
#      Dose  mortos sexo  logDose
# 235    32       1    M      5.0
# 236    32       0    M      5.0
# 237    32       0    M      5.0
# 238    32       0    M      5.0
# 239    32       0    M      5.0

cypemethrin['sexo'] = pd.get_dummies(data=cypemethrin['sexo'], drop_first=True)

# >>> cypemethrin.head()
#    Dose  mortos  sexo  logDose
# 0     1       1     0      0.0
# 1     1       0     0      0.0
# 2     1       0     0      0.0
# 3     1       0     0      0.0
# 4     1       0     0      0.0

glm_binom = sm.GLM(cypemethrin['mortos'], sm.add_constant(cypemethrin[['logDose', 'sexo']]), family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())

# >>> print(res.summary())
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                 mortos   No. Observations:                  240
# Model:                            GLM   Df Residuals:                      237
# Model Family:                Binomial   Df Model:                            2
# Link Function:                  logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -106.62
# Date:                Mon, 25 Oct 2021   Deviance:                       213.24
# Time:                        19:25:46   Pearson chi2:                     213.
# No. Iterations:                     5   Pseudo R-squ. (CS):             0.3887
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -2.3724      0.386     -6.154      0.000      -3.128      -1.617
# logDose        1.0642      0.131      8.119      0.000       0.807       1.321
# sexo          -1.1007      0.356     -3.093      0.002      -1.798      -0.403
# ==============================================================================


cypemethrin = pd.DataFrame({
'mortos' :  [1, 4, 9, 13, 18, 20, 0, 2, 6, 10, 12, 16],
'Nmortos' : [19, 16, 11, 7, 2, 0, 20, 18, 14, 10, 8, 4],
'sexo': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ],
'Dose' : [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32 ]})

# >>> cypemethrin
#     mortos  Nmortos  sexo  Dose
# 0        1       19     1     1
# 1        4       16     1     2
# 2        9       11     1     4
# 3       13        7     1     8
# 4       18        2     1    16
# 5       20        0     1    32
# 6        0       20     0     1
# 7        2       18     0     2
# 8        6       14     0     4
# 9       10       10     0     8
# 10      12        8     0    16
# 11      16        4     0    32

cypemethrin['logDose'] = np.log2(cypemethrin.Dose)

glm_binom = sm.GLM(cypemethrin[['mortos', 'Nmortos']], sm.add_constant(cypemethrin[['logDose', 'sexo']]), family=sm.families.Binomial())
res = glm_binom.fit()
print(res.summary())

# >>> print(res.summary())
#                    Generalized Linear Model Regression Results                   
# =================================================================================
# Dep. Variable:     ['mortos', 'Nmortos']   No. Observations:                   12
# Model:                               GLM   Df Residuals:                        9
# Model Family:                   Binomial   Df Model:                            2
# Link Function:                     logit   Scale:                          1.0000
# Method:                             IRLS   Log-Likelihood:                -18.434
# Date:                   Mon, 25 Oct 2021   Deviance:                       6.7571
# Time:                           19:56:12   Pearson chi2:                     5.31
# No. Iterations:                        5   Pseudo R-squ. (CS):             0.9999
# Covariance Type:               nonrobust                                         
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -3.4732      0.469     -7.413      0.000      -4.391      -2.555
# logDose        1.0642      0.131      8.119      0.000       0.807       1.321
# sexo           1.1007      0.356      3.093      0.002       0.403       1.798
# ==============================================================================


# Quedas = pd.DataFrame({
# 'Quedas' : [1, 1, 2, 0, 2],
# 'Intervencao' : [1, 1, 1, 1, 1],
# 'Sexo' : [0, 0, 1, 1, 0],
# 'Balanco' : [45, 62, 43, 76, 51],
# 'Forca' : [70, 66, 64, 48, 72]})

Quedas = pd.DataFrame(np.loadtxt('https://www.ime.usp.br/~giapaula/geriatra.dat', unpack = True).T, columns = ['Quedas'
,'Intervencao'
,'Sexo'
,'Balanco'
,'Forca'],  dtype= np.int8)

# >>> Quedas
#     Quedas  Intervencao  Sexo  Balanco  Forca
# 0        1            1     0       45     70
# 1        1            1     0       62     66
# 2        2            1     1       43     64
# 3        0            1     1       76     48
# 4        2            1     0       51     72
# ..     ...          ...   ...      ...    ...
# 95       5            0     1       76     46
# 96       2            0     1       33     55
# 97       4            0     0       69     48
# 98       4            0     1       50     52
# 99       2            0     0       37     56
# 
# [100 rows x 5 columns]

glm_poisson = sm.GLM(Quedas['Quedas'], sm.add_constant(Quedas[['Intervencao' ,'Sexo' ,'Balanco' ,'Forca']]),
 family=sm.families.Poisson())
res = glm_poisson.fit()
print(res.summary())

# >>> print(res.summary())
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                 Quedas   No. Observations:                  100
# Model:                            GLM   Df Residuals:                       95
# Model Family:                 Poisson   Df Model:                            4
# Link Function:                    log   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -183.64
# Date:                Mon, 25 Oct 2021   Deviance:                       108.79
# Time:                        20:13:51   Pearson chi2:                     106.
# No. Iterations:                     5   Pseudo R-squ. (CS):             0.5951
# Covariance Type:            nonrobust                                         
# ===============================================================================
#                   coef    std err          z      P>|z|      [0.025      0.975]
# -------------------------------------------------------------------------------
# const           0.4895      0.337      1.453      0.146      -0.171       1.150
# Intervencao    -1.0694      0.133     -8.031      0.000      -1.330      -0.808
# Sexo           -0.0466      0.120     -0.388      0.698      -0.282       0.189
# Balanco         0.0095      0.003      3.207      0.001       0.004       0.015
# Forca           0.0086      0.004      1.986      0.047       0.000       0.017
# ===============================================================================

# Como prever
res.predict([ 1, 1, 0, 45, 70])
#array([1.56177349])

# a matematica continua funcionando?
np.exp(((res.params)*[ 1, 1, 0, 45, 70]).sum())
# 1.561773493060502

# >>> res.params
# const          0.489467
# Intervencao   -1.069403
# Sexo          -0.046606
# Balanco        0.009470
# Forca          0.008566
# dtype: float64

# >>> (res.params)*[ 1, 1, 0, 45, 70]
# const          0.489467
# Intervencao   -1.069403
# Sexo          -0.000000
# Balanco        0.426149
# Forca          0.599608
# dtype: float64

# >>> ((res.params)*[ 1, 1, 0, 45, 70]).sum()
# 0.44582203005765614
