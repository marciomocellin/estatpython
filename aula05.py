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

#O valor de p excede nosso nível de significância escolhido,
# portanto, não rejeitamos a hipótese nula de que a amostra observada
# é extraída da distribuição normal padrão.

# Agora, suponha que desejamos verificar se as mesmas amostras
# deslocadas em 2,1 são consistentes com o fato de terem sido tiradas
# de uma distribuição normal com uma média de 2.

y = x + 2.1
res = stats.cramervonmises(y, 'norm', args=(2,))
res.statistic, res.pvalue

#Aqui, usamos a palavra-chave args para especificar a média ( loc)
# da distribuição normal para testar os dados. Isso é equivalente ao
# seguinte, em que criamos uma distribuição normal  com média 2,1 e,
# em seguida, passamos seu cdf método como um argumento.

frozen_dist = stats.norm(loc=2)
res = stats.cramervonmises(y, frozen_dist.cdf)
res.statistic, res.pvalue

#Em qualquer dos casos, rejeitaríamos a hipótese nula de que a amostra
# observada é retirada de uma distribuição normal com uma média de 2
# (e variância padrão de 1) porque o valor de p 0,04 é menor do que
# nosso nível de significância escolhido.

# scipy.stats.cramervonmises_2samp

# Suponha que desejamos testar se duas amostras geradas por scipy.stats.norm.rvstêm a mesma distribuição. Escolhemos um nível de significância alfa = 0,05.

from scipy import stats
rng = np.random.default_rng()
x = stats.norm.rvs(size=100, random_state=rng)
y = stats.norm.rvs(size=70, random_state=rng)
res = stats.cramervonmises_2samp(x, y)
res.statistic, res.pvalue #(0.09171428571428564, 0.6335531231888883)

#O valor p excede nosso nível de significância escolhido, portanto,
# não rejeitamos a hipótese nula de que as amostras observadas são
# retiradas da mesma distribuição.
#Para tamanhos de amostra pequenos, pode-se calcular os valores p exatos:

x = stats.norm.rvs(size=7, random_state=rng)
y = stats.t.rvs(df=2, size=6, random_state=rng)
res = stats.cramervonmises_2samp(x, y, method='exact')
res.statistic, res.pvalue

# O valor p com base na distribuição assintótica é uma boa aproximação,
# embora o tamanho da amostra seja pequeno.

res = stats.cramervonmises_2samp(x, y, method='asymptotic')
res.statistic, res.pvalue
#Independentemente do método, não se rejeitaria a hipótese nula no
# nível de significância escolhido neste exemplo.

#scipy.stats.kstest

from scipy import stats
rng = np.random.default_rng()
x = np.linspace(-15, 15, 9)
stats.kstest(x, 'norm')
# KstestResult(statistic=0.444356027159..., pvalue=0.038850140086...)
stats.kstest(stats.norm.rvs(size=100, random_state=rng), stats.norm.cdf)

#As linhas acima são equivalentes a:

stats.kstest(stats.norm.rvs, 'norm', N=100)

#Testando variáveis ​​aleatórias t distribuídas em relação à distribuição normal

# Com 100 graus de liberdade, a distribuição t parece próxima da distribuição normal,
# e o teste KS não rejeita a hipótese de que a amostra veio da distribuição normal:

stats.kstest(stats.t.rvs(100, size=100, random_state=rng), 'norm')

#Com 3 graus de liberdade, a distribuição t é suficientemente diferente da distribuição normal,
# de modo que podemos rejeitar a hipótese de que a amostra veio da distribuição normal no
# nível de 10%:

stats.kstest(stats.t.rvs(3, size=100, random_state=rng), 'norm')

#scipy.stats.kstest

from scipy import stats
rng = np.random.default_rng()
n1 = 200  # tamanho da primeira amostra
n2 = 300  # tamanho da segunda amostra

#Para uma distribuição diferente, podemos rejeitar a hipótese nula uma vez que o valor p está abaixo de 1%:

rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1, random_state=rng)
rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5, random_state=rng)
stats.ks_2samp(rvs1, rvs2)

#Para uma distribuição ligeiramente diferente, não podemos rejeitar a hipótese nula em um alfa de 10% ou inferior,
# uma vez que o valor de p em 0,144 é superior a 10%

rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0, random_state=rng)
stats.ks_2samp(rvs1, rvs3)

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

# A hipótese nula não pode ser rejeitada para três amostras de uma distribuição idêntica.
# O valor p relatado (25%) foi limitado e pode não ser muito preciso (uma vez que corresponde ao valor 0,449,
# enquanto a estatística é -0,731):

stats.anderson_ksamp([rng.normal(size=50),
rng.normal(size=30), rng.normal(size=20)])

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

# Com um valor de p próximo de 1, não podemos concluir que existe uma diferença significativa nas escalas (conforme o esperado).

# Agora aplique o teste a x1 e x3 :

ansari(x1, x3)

# A probabilidade de observar tal valor extremo da estatística sob a hipótese nula de escalas iguais é de apenas 0,03087%.
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
p

#O pequeno valor de p sugere que as populações não têm variâncias iguais.

#Isso não é surpreendente, dado que a variância da amostra de b é muito maior do que
# a de a e c :

[np.var(x, ddof=1) for x in [a, b, c]]

#scipy.stats.jarque_bera

from scipy import stats
rng = np.random.default_rng()
x = rng.normal(0, 1, 100000)
jarque_bera_test = stats.jarque_bera(x)
jarque_bera_test
# Jarque_beraResult(statistic=3.3415184718131554, pvalue=0.18810419594996775)
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

X = sm.add_constant(X)
print(X[0:3])
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

print("Parameters: ", results.params)
print("R2: ", results.rsquared)