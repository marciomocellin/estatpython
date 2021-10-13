## scipy.stats.ttest_1samp

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

## scipy.stats.ttest_ind

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

##scipy.stats.mannwhitneyu

# Exemplo: nove adultos jovens de uma amostra aleatória foram diagnosticados
# com diabetes tipo II nas idades abaixo.

males = [19, 22, 16, 29, 24]
females = [20, 11, 17, 12]

# Usamos o teste U de Mann-Whitney para avaliar se há uma diferença estatisticamente
# significativa na idade de diagnóstico de homens e mulheres.
# A hipótese nula é que a distribuição das idades de diagnóstico masculino é igual à
# distribuição das idades de diagnóstico feminino.
# Decidimos que um nível de confiança de 95% é necessário para rejeitar a hipótese nula
# em favor da alternativa de que as distribuições são diferentes.
# Como o número de amostras é muito pequeno e não há empates nos dados,
# podemos comparar a estatística de teste observada com a distribuição exata da estatística
# de teste sob a hipótese nula.

from scipy.stats import mannwhitneyu
mannwhitneyu(males, females, method="exact")

#A distribuição exata da estatística de teste é assintoticamente normal,
# então o exemplo continua comparando o valor p exato com o valor p produzido
# usando a aproximação normal.

_, pnorm = mannwhitneyu(males, females, method="asymptotic")
print(pnorm)



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

##scipy.stats.wilcoxon

# Exemplo: as diferenças de altura entre plantas de milho autofecundadas
# e cruzadas são dadas como segue:

d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

# As plantas com fertilização cruzada parecem ser mais altas.
# Para testar a hipótese nula de que não há diferença de altura,
# podemos aplicar o teste bilateral:

from scipy.stats import wilcoxon
w, p = wilcoxon(d)
w, p

#Assim, rejeitaríamos a hipótese nula com um nível de confiança de 5%,
# concluindo que há diferença de altura entre os grupos.
# Para confirmar que a mediana das diferenças pode ser considerada positiva, usamos:

w, p = wilcoxon(d, alternative='greater')
w, p

#Isso mostra que a hipótese nula de que a mediana é negativa pode ser rejeitada
# a um nível de confiança de 5% em favor da alternativa de que a mediana é maior que zero.
# Os valores de p acima são exatos. O uso da aproximação normal fornece valores muito
# semelhantes:

w, p = wilcoxon(d, mode='approx')
w, p

#Observe que a estatística mudou para 96 ​​no caso unilateral (a soma das classificações das
# diferenças positivas), enquanto é 24 no caso bilateral (o mínimo da soma das
# classificações acima e abaixo de zero).



# scipy.stats.f_oneway

from scipy.stats import f_oneway

#Aqui estão alguns dados sobre a medição da concha
# (o comprimento da cicatriz do músculo adutor anterior,
# padronizado pela divisão pelo comprimento) no mexilhão Mytilus trossulus
# de cinco locais: Tillamook, Oregon; Newport, Oregon; Petersburgo, Alasca; Magadan, Rússia; e Tvarminne, Finlândia,
# extraídos de um conjunto de dados muito maior usado em McDonald et al. (1991).

tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,
             0.0659, 0.0923, 0.0836]
newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835,
           0.0725]
petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,
           0.0689]
tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
f_oneway(tillamook, newport, petersburg, magadan, tvarminne)

# f_onewayaceita matrizes de entrada multidimensionais. Quando as entradas são
# multidimensionais e o eixo não é fornecido, o teste é executado ao longo do primeiro
# eixo das matrizes de entrada. Para os dados a seguir, o teste é realizado três vezes,
# uma para cada coluna.

a = np.array([[9.87, 9.03, 6.81],
              [7.18, 8.35, 7.00],
              [8.39, 7.58, 7.68],
              [7.45, 6.33, 9.35],
              [6.41, 7.10, 9.33],
              [8.00, 8.24, 8.44]])
b = np.array([[6.35, 7.30, 7.16],
              [6.65, 6.68, 7.63],
              [5.72, 7.73, 6.72],
              [7.01, 9.19, 7.41],
              [7.75, 7.87, 8.30],
              [6.90, 7.97, 6.97]])
c = np.array([[3.31, 8.77, 1.01],
              [8.25, 3.24, 3.62],
              [6.32, 8.81, 5.19],
              [7.48, 8.83, 8.91],
              [8.59, 6.01, 6.07],
              [3.07, 9.72, 7.48]])
F, p = f_oneway(a, b, c)
F
p

# scipy.stats.kruskal

from scipy import stats
x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8, 10]
stats.kruskal(x, y)

x = [1, 1, 1]
y = [2, 2, 2]
z = [2, 2]
stats.kruskal(x, y, z)