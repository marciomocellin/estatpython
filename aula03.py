# scipy.stats.ttest_1samp

from scipy import stats
import numpy as np
rng = np.random.default_rng()
rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2), random_state=rng)
print(rvs)

# Será teste se a média da amostra aleatória é igual à média verdadeira e uma média diferente.
# Rejeitamos a hipótese nula no segundo caso e não a rejeitamos no primeiro caso.

stats.ttest_1samp(rvs, 5.0)

stats.ttest_1samp(rvs, 0.0)

#Exemplos usando eixo e dimensão não escalar para a média da população.

result = stats.ttest_1samp(rvs, [5.0, 0.0])
print(result.statistic)
print(result.pvalue)

result = stats.ttest_1samp(rvs.T, [5.0, 0.0], axis=1)
print(result.statistic)
print(result.pvalue)

result = stats.ttest_1samp(rvs, [[5.0], [0.0]])
print(result.statistic)
print(result.pvalue)

# scipy.stats.ttest_ind

# Teste com amostra com médias idênticas:

rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
rvs2 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
stats.ttest_ind(rvs1, rvs2)
stats.ttest_ind(rvs1, rvs2, equal_var=False)

#subestima p para variâncias desiguais:

rvs3 = stats.norm.rvs(loc=5, scale=20, size=500, random_state=rng)
stats.ttest_ind(rvs1, rvs3)
stats.ttest_ind(rvs1, rvs3, equal_var=False)

# Quando n1! = N2, a estatística t de variância igual não é mais igual à
# estatística t de variância diferente:

rvs4 = stats.norm.rvs(loc=5, scale=20, size=100, random_state=rng)
stats.ttest_ind(rvs1, rvs4)
stats.ttest_ind(rvs1, rvs4, equal_var=False)

#Teste t com diferentes médias, variâncias e n:
rvs5 = stats.norm.rvs(loc=8, scale=20, size=100, random_state=rng)
stats.ttest_ind(rvs1, rvs5)
stats.ttest_ind(rvs1, rvs5, equal_var=False)

## scipy.stats.ttest_ind_from_stats
# Suponha que temos os dados de resumo para duas amostras, como segue:
#
#                 Sample   Sample
#           Size   Mean   Variance
#Sample 1    13    15.0     87.5
#Sample 2    11    12.0     39.0

#Aplique o teste t a estes dados (com a suposição de que as variâncias da população são iguais):

from scipy.stats import ttest_ind_from_stats
ttest_ind_from_stats(mean1=15.0, std1=np.sqrt(87.5), nobs1=13,
                     mean2=12.0, std2=np.sqrt(39.0), nobs2=11)

# Para comparação, aqui estão os dados dos quais essas estatísticas de resumo foram obtidas.
# Com esses dados, podemos calcular o mesmo resultado usando scipy.stats.ttest_ind:

a = np.array([1, 3, 4, 6, 11, 13, 15, 19, 22, 24, 25, 26, 26])
b = np.array([2, 4, 6, 9, 11, 13, 14, 15, 18, 19, 21])
from scipy.stats import ttest_ind
ttest_ind(a, b)

# Suponha que, em vez disso, tenhamos dados binários e
# gostaríamos de aplicar um teste t para comparar a proporção
# de 1s em dois grupos independentes:
# 
#                   Number of    Sample     Sample
#             Size    ones        Mean     Variance
# Sample 1    150      30         0.2        0.16
# Sample 2    200      45         0.225      0.174375
# 
# A média da amostra p é a proporção de uns na amostra e
# a variância para uma observação binária é estimada por p(1-p).

ttest_ind_from_stats(mean1=0.2, std1=np.sqrt(0.16), nobs1=150,
                     mean2=0.225, std2=np.sqrt(0.17437), nobs2=200)

# Para comparação, poderíamos calcular a estatística t e
# o valor p usando matrizes de 0s e 1s e scipy.stat.ttest_ind , como acima.
group1 = np.array([1]*30 + [0]*(150-30))
group2 = np.array([1]*45 + [0]*(200-45))
ttest_ind(group1, group2)

# scipy.stats.ttest_rel

from scipy import stats
rng = np.random.default_rng()
rvs1 = stats.norm.rvs(loc=5, scale=10, size=5, random_state=rng)
rvs2 = (rvs1
        + stats.norm.rvs(scale=0.2, size=5, random_state=rng))
stats.ttest_rel(rvs1, rvs2)

rvs3 = (stats.norm.rvs(loc=8, scale=10, size=500, random_state=rng)
        + stats.norm.rvs(scale=0.2, size=500, random_state=rng))

stats.ttest_rel(rvs1, rvs3)