import numpy as np

#Quando apenas f_obs é dado, assume-se que as frequências esperadas são
#uniformes e dadas pela média das frequências observadas.

from scipy.stats import chisquare
chisquare([132, 98, 95, 98, 105, 133, 158])

#Com f_exp as frequências esperadas podem ser fornecidas.
chisquare([132, 98, 95, 98, 105, 133, 158],
    f_exp=[117, 117, 117, 117, 117, 117, 117])

# Quando f_obs é 2-D, por padrão, o teste é aplicado a cada coluna.
chisquare([16, 18, 16, 14, 12, 12])

obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
obs.shape

chisquare(obs)

# scipy.stats.contingency.margins

from scipy.stats.contingency import margins

obs = np.array([[491, 377, 31],[213, 112, 8]])
m0, m1 = margins(obs)
m0, m1

obs.sum()

# scipy.stats.contingency.expected_freq

from scipy.stats.contingency import expected_freq
expected_freq(obs)

b = np.arange(18).reshape(3, 6)
expected_freq(b)

## scipy.stats.chi2_contingency

from scipy.stats import chi2_contingency

chi2_contingency(obs)

# Execute o teste usando o log da razão de verossimilhança (ou seja, o “teste G”) em vez da estatística qui-quadrado de Pearson.

g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")
g, p

# Um exemplo de quatro vias (2 x 2 x 2 x 2) (cubo):

obs = np.array(
    [[[[12, 17],
       [11, 16]],
      [[11, 12],
       [15, 16]]],
     [[[23, 15],
       [30, 22]],
      [[14, 17],
       [15, 16]]]])
chi2_contingency(obs)

# scipy.stats.power_divergence

# Quando apenas f_obs é dado, assume-se que as frequências esperadas são uniformes e
# dadas pela média das frequências observadas. Aqui, realizamos um teste G (ou seja,
# usamos a estatística de razão de verossimilhança):

from scipy.stats import power_divergence
power_divergence([16, 18, 16, 14, 12, 12], lambda_='log-likelihood')

# As frequências esperadas podem ser fornecidas com o argumento f_exp :

power_divergence([16, 18, 16, 14, 12, 12],
                 f_exp=[16, 16, 16, 16, 16, 8],
                 lambda_='log-likelihood')

# Quando f_obs é 2-D, por padrão, o teste é aplicado a cada coluna.

obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
obs.shape #(6, 2)
power_divergence(obs, lambda_="log-likelihood")

# ddof é a alteração a ser feita nos graus de liberdade padrão.

power_divergence([16, 18, 16, 14, 12, 12], ddof=1)

# O cálculo dos valores p é feito transmitindo a estatística de teste com ddof .

power_divergence([16, 18, 16, 14, 12, 12], ddof=[0,1,2])

# f_obs e f_exp também são transmitidos.
# A seguir, f_obs tem forma (6,) e f_exp tem forma (2, 6),
# então o resultado da transmissão de f_obs e f_exp tem forma (2, 6).
# Para calcular as estatísticas qui-quadradas desejadas, devemos usar axis=1:

power_divergence([16, 18, 16, 14, 12, 12],
                 f_exp=[[16, 16, 16, 16, 16, 8],
                        [8, 20, 20, 16, 12, 12]],
                 axis=1)

# scipy.stats.fisher_exact
# Digamos que passemos alguns dias contando baleias e tubarões nos oceanos Atlântico e Índico. No oceano Atlântico encontramos 8 baleias e 1 tubarão, no oceano Índico 2 baleias e 5 tubarões. Então, nossa tabela de contingência é:
# 
#          Atlântico  Índico
# baleias      8         2
# tubarões     1         5
# Usamos esta tabela para encontrar o valor p:

from scipy.stats import fisher_exact
oddsratio, pvalue = fisher_exact([[8, 2], [1, 5]])
pvalue

#A probabilidade de observarmos isso ou uma razão ainda mais desequilibrada ao acaso é
# de cerca de 3,5%. Um nível de significância comumente usado é 5% - se o adotarmos,
# podemos, portanto, concluir que nosso desequilíbrio observado é estatisticamente
# significativo; as baleias preferem o Atlântico, enquanto os tubarões preferem o oceano
# Índico.