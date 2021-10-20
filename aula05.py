import numpy as np

# scipy.stats.cramervonmises

# Suponha que desejamos testar se os dados gerados por
# scipy.stats.norm.rvs foram, de fato, extraídos da
# distribuição normal padrão. Escolhemos um nível de
# significância alfa = 0,05.

from scipy import stats
rng = np.random.default_rng()
x = stats.norm.rvs(size=500, random_state=rng)
res = stats.cramervonmises(x, 'norm')
res.statistic, res.pvalue
#CramerVonMisesResult(statistic=0.1276613786697622, pvalue=0.46556649116631343)

#O valor de p excede nosso nível de significância escolhido,
# portanto, não rejeitamos a hipótese nula de que a amostra observada
# é extraída da distribuição normal padrão.

# Agora, suponha que desejamos verificar se as mesmas amostras
# deslocadas em 2,1 são consistentes com o fato de terem sido tiradas
# de uma distribuição normal com uma média de 2.

y = x + 2.1
res = stats.cramervonmises(y, 'norm', args=(2,))
#CramerVonMisesResult(statistic=0.7040268563073291, pvalue=0.012420322007088758)

#Aqui, usamos a palavra-chave args para especificar a média (loc)
# da distribuição normal para testar os dados. Isso é equivalente ao
# seguinte, em que criamos uma distribuição normal com média 2,1 e,
# em seguida, passamos seu cdf método como um argumento.

frozen_dist = stats.norm(loc=2)
res = stats.cramervonmises(y, frozen_dist.cdf)
res.statistic, res.pvalue
#(0.7040268563073291, 0.012420322007088758)

#Em todos dos casos, rejeitaríamos a hipótese nula de que a amostra
# observada é retirada de uma distribuição normal com uma média de 2
# (e variância padrão de 1) porque o valor de p 0,01 é menor do que
# nosso nível de significância escolhido.

# scipy.stats.cramervonmises_2samp

# Suponha que desejamos testar se duas amostras geradas por scipy.stats.norm.rvstêm a mesma distribuição. Escolhemos um nível de significância alfa = 0,05.

from scipy import stats
rng = np.random.default_rng()
x = stats.norm.rvs(size=100, random_state=rng)
y = stats.norm.rvs(size=70, random_state=rng)
res = stats.cramervonmises_2samp(x, y)
res.statistic, res.pvalue #(0.12726890756302467, 0.47115054777270216)

#O valor p excede nosso nível de significância escolhido, portanto,
# não rejeitamos a hipótese nula de que as amostras observadas são
# retiradas da mesma distribuição.
#Para tamanhos de amostra pequenos, pode-se calcular os valores p exatos:

x = stats.norm.rvs(size=7, random_state=rng)
y = stats.t.rvs(df=2, size=6, random_state=rng)
res = stats.cramervonmises_2samp(x, y, method='exact')
res.statistic, res.pvalue #(0.042124542124541975, 0.9801864801864801)

# O valor p com base na distribuição assintótica é uma boa aproximação,
# embora o tamanho da amostra seja pequeno.

res = stats.cramervonmises_2samp(x, y, method='asymptotic')
res.statistic, res.pvalue #(0.042124542124541975, 0.9937806294485269)

#Independentemente do método, não se rejeitaria a hipótese nula no
# nível de significância escolhido neste exemplo.

x = stats.norm.rvs(size=700, random_state=rng)
y = stats.t.rvs(df=2, size=600, random_state=rng)
res = stats.cramervonmises_2samp(x, y)
print(res) #CramerVonMisesResult(statistic=0.6771188644688664, pvalue=0.014472209121915047)

#scipy.stats.kstest

from scipy import stats
rng = np.random.default_rng()
x = np.linspace(-15, 15, 9)
stats.kstest(x, 'norm')
# KstestResult(statistic=0.4443560271592436, pvalue=0.03885014008678778)
stats.kstest(stats.norm.rvs(size=100, random_state=rng), stats.norm.cdf)

#As linhas acima são equivalentes a:

stats.kstest(stats.norm.rvs, 'norm', N=100)

#Testando variáveis ​​aleatórias t distribuídas em relação à distribuição normal

# Com 100 graus de liberdade, a distribuição t parece próxima da distribuição normal,
# e o teste KS não rejeita a hipótese de que a amostra veio da distribuição normal:

stats.kstest(stats.t.rvs(100, size=100, random_state=rng), 'norm')
# KstestResult(statistic=0.10694118810178882, pvalue=0.18878890547885985)

#Com 3 graus de liberdade, a distribuição t é suficientemente diferente da distribuição normal,
# de modo que podemos rejeitar a hipótese de que a amostra veio da distribuição normal no
# nível de 10%:

stats.kstest(stats.t.rvs(3, size=100, random_state=rng), 'norm')
#KstestResult(statistic=0.11786287323060995, pvalue=0.11456645992107758)

#scipy.stats.ks_2samp

from scipy import stats
rng = np.random.default_rng()
n1 = 200  # tamanho da primeira amostra
n2 = 300  # tamanho da segunda amostra

#Para uma distribuição diferente, podemos rejeitar a hipótese nula uma vez que o valor p está abaixo de 1%:

rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1, random_state=rng)
rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5, random_state=rng)
stats.ks_2samp(rvs1, rvs2)
# KstestResult(statistic=0.24, pvalue=1.5876939054582095e-06)

#Para uma distribuição ligeiramente diferente, não podemos rejeitar a hipótese nula em um alfa de 10% ou inferior,
# uma vez que o valor de p em 0,219 é superior a 10%

rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0, random_state=rng)
stats.ks_2samp(rvs1, rvs3)
# KstestResult(statistic=0.095, pvalue=0.2192140768654085)

#Para uma distribuição idêntica, não podemos rejeitar a hipótese nula uma vez que o valor p é alto, 41%:

rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0, random_state=rng)
stats.ks_2samp(rvs1, rvs4)

#scipy.stats.anderson_ksamp

from scipy import stats
rng = np.random.default_rng()

# A hipótese nula de que as duas amostras aleatórias vêm da mesma distribuição pode ser rejeitada
# no nível de 5% porque o valor de teste retornado é maior do que o valor crítico para 5% (1,961),
# mas não no nível de 2,5%. A interpolação dá um nível de significância aproximado de 3,2%:

stats.anderson_ksamp([rng.normal(size=50),
rng.normal(loc=0.5, size=30)])
# p valor = significance_level = 0.07396028404997687

# A hipótese nula não pode ser rejeitada para três amostras de uma distribuição idêntica.
# O valor p relatado (25%) foi limitado e pode não ser muito preciso (uma vez que corresponde ao valor 0,449,
# enquanto a estatística é -0,731):

stats.anderson_ksamp([rng.normal(size=50),
rng.normal(size=30), rng.normal(size=20)])
#Anderson_ksampResult(statistic=-0.5917988120678772, critical_values=array([0.44925884, 1.3052767 , 1.9434184 , 2.57696569, 3.41634856, 4.07210043, 5.56419101]), significance_level=0.25)

#scipy.stats.ansari

from scipy.stats import ansari
rng = np.random.default_rng()
#Para esses exemplos, criaremos três conjuntos de dados aleatórios. Os dois primeiros, com tamanhos 35 e 25,
# são extraídos de uma distribuição normal com média 0 e desvio padrão 2. O terceiro conjunto de dados tem
# tamanho 25 e é extraído de uma distribuição normal com desvio padrão 1,25.

x1 = rng.normal(loc=0, scale=2, size=35)
x2 = rng.normal(loc=0, scale=2, size=25)
x3 = rng.normal(loc=0, scale=1.25, size=25)

# Primeiramente, aplicamos ansari para x1 e x2 . Essas amostras são retiradas da mesma distribuição, portanto,
# esperamos que o teste de Ansari-Bradley não nos leve a concluir que as escalas das distribuições são diferentes.

ansari(x1, x2)
#AnsariResult(statistic=534.0, pvalue=0.811752031516162)

# Com um valor de p próximo de 1, não podemos concluir que existe uma diferença significativa nas escalas (conforme o esperado).

# Agora aplique o teste a x1 e x3 :

ansari(x1, x3)
# AnsariResult(statistic=464.0, pvalue=0.01846645873767982)

# A probabilidade de observar tal valor extremo da estatística sob a hipótese nula de escalas iguais é de apenas 1,84%.
# Tomamos isso como evidência contra a hipótese nula em favor da alternativa: as escalas das distribuições das quais as
# amostras foram retiradas não são iguais.

# Podemos usar o parâmetro alternativo para realizar um teste unilateral. No exemplo acima, a escala de x1 é maior do que x3 e,
# portanto, a proporção das escalas de x1 e x3 é maior do que 1. Isso significa que o valor p quando alternative='greater'deve estar próximo de 0 e,
# portanto, devemos ser capazes de rejeitar o nulo hipótese:

ansari(x1, x3, alternative='greater')

#Como podemos ver, o valor p é de fato bastante baixo. O uso de alternative='less'deve,
# portanto, produzir um grande valor p:

ansari(x1, x3, alternative='less')

# scipy.stats.fligner

#Testa se as listas de a , b e c vêm de populações com variâncias iguais.

from scipy.stats import fligner
a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
stat, p = fligner(a, b, c)
p #pvalue=0.00450826080004775

#O pequeno valor de p sugere que as populações não têm variâncias iguais.

#Isso não é surpreendente, dado que a variância da amostra de b é muito maior do que
# a de a e c :

[np.var(x, ddof=1) for x in [a, b, c]] #[0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

#outros testes de variancia

from scipy.stats import bartlett, levene
bartlett(a, b, c)
#BartlettResult(statistic=22.789434813726768, pvalue=1.1254782518834628e-05)
levene(a, b, c)
#LeveneResult(statistic=7.584952754501659, pvalue=0.002431505967249681)

#scipy.stats.jarque_bera

from scipy import stats
rng = np.random.default_rng()
x = rng.normal(0, 1, 100000)
jarque_bera_test = stats.jarque_bera(x)
jarque_bera_test
# Jarque_beraResult(statistic=3.3415184718131554, pvalue= 0.18810419594996775)
jarque_bera_test.statistic
# 3.3415184718131554
jarque_bera_test.pvalue
# 0.18810419594996775

# scipy.stats.kurtosistest
from scipy.stats import kurtosistest
kurtosistest(list(range(20)))
# KurtosistestResult(statistic=-1.7058104152122062, pvalue=0.08804338332528348)
kurtosistest(list(range(20)), alternative='less')
# KurtosistestResult(statistic=-1.7058104152122062, pvalue=0.04402169166264174)
kurtosistest(list(range(20)), alternative='greater')
# KurtosistestResult(statistic=-1.7058104152122062, pvalue=0.9559783083373583)
rng = np.random.default_rng()
s = rng.normal(0, 1, 1000)
kurtosistest(s)
#KurtosistestResult(statistic=-0.3188545786000282, pvalue=0.7498367888656665)

# Pacote statsmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))

print(X[0:3])
beta = np.array([1, 0.1, 10])

e = np.random.normal(size=nsample)
print(e[0:3])

X = sm.add_constant(X)
print(X[0:3])

y = np.dot(X, beta) + e
print(y[0:3])

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

#                             OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       1.000
# Model:                            OLS   Adj. R-squared:                  1.000
# Method:                 Least Squares   F-statistic:                 4.409e+06
# Date:                Wed, 20 Oct 2021   Prob (F-statistic):          3.21e-241
# Time:                        19:47:11   Log-Likelihood:                -141.92
# No. Observations:                 100   AIC:                             289.8
# Df Residuals:                      97   BIC:                             297.6
# Df Model:                           2                                         
# Covariance Type:            nonrobust                                         
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          1.2849      0.299      4.302      0.000       0.692       1.878
# x1            -0.0167      0.138     -0.121      0.904      -0.291       0.257
# x2            10.0099      0.013    749.389      0.000       9.983      10.036
# ==============================================================================
# Omnibus:                        3.186   Durbin-Watson:                   2.138
# Prob(Omnibus):                  0.203   Jarque-Bera (JB):                1.927
# Skew:                          -0.061   Prob(JB):                        0.382
# Kurtosis:                       2.331   Cond. No.                         144.
# ==============================================================================

print("Parameters: ", results.params)
print("R2: ", results.rsquared)