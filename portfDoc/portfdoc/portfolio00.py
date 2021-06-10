import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import data as pdr

pd.plotting.register_matplotlib_converters()

#Activos
plt.style.use("seaborn-colorblind")
tickers=["GGAL","BMA","YPF","GLOB","MELI"]
prices=pdr.get_data_yahoo(tickers,start="2015-01-01", end=dt.date.today())["Adj Close"]
returns=prices.pct_change()#porcentaje de cambio entre un t-1 y un t, al pedo lo pone

#Grafiquemos
plt.figure(figsize=(14,7))
for i in prices.columns.values:
    plt.plot(prices.index, np.log(prices[i]),lw=2, alpha=0.8, label=i)
plt.legend(loc="upper left", fontsize=12)
plt.ylabel("log price")

#plotting monthly returns, BM="business month end frequency"
monthly_prices = prices.asfreq('BM').ffill()
monthly_returns = monthly_prices.pct_change()

plt.figure(figsize=(14,7))
for i in monthly_returns.columns.values:
    plt.plot(monthly_returns.index, monthly_returns[i], lw=2, alpha=0.8, label=i)
plt.legend(loc='lower left', fontsize = 14)
plt.ylabel('Monthly returns')

from pypfopt import expected_returns
from pypfopt import risk_models

mu = expected_returns.mean_historical_return(monthly_prices,frequency=12)#media historica
covmat = risk_models.sample_cov(monthly_prices, frequency = 12)#covarianza muestral

sd = np.sqrt(np.diag(covmat))#de la matriz de varianzas y covarianzas se queda con la diagonal y con la raiz cuadrada se arma los desvios estandars 

fig = plt.figure()#esto devuelve la grafica esa de las cruces
plt.plot(sd, mu.to_numpy(), 'x', markersize = 5)
plt.ylabel('Retorno esperado')
plt.xlabel('Volatilidad')
plt.title('Mean-Variance Analysis')


#--------------------hasta acá tenemos portafolios armados por nosotros------
#--------------------ahora armemos algo aleatorio

# Calculamos volatilidad y retorno de un portafolio
def portfolio_metrics(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights)#ponderaciones
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))#desvio estandar, es una multiplicacion matricial, esa T es de transpuesta
    return ret, std

# armamos los portafolios random
def random_portfolios(num_port, mean_returns, cov_matrix):
    metrics = np.zeros((2,num_port))
    weights_matrix = []#ponderaciones aleatorias
    
    for i in range(num_port):
        weights = np.random.random(len(mean_returns))#tantas ponderaciones como activos individuales
        weights /= np.sum(weights)#lo normalizamos a 1
        weights_matrix.append(weights)
        port_mu, port_std = portfolio_metrics(weights, mean_returns, cov_matrix)
        metrics[0,i] = port_mu
        metrics[1,i] = port_std
    return metrics, weights_matrix


# Número de portafolios
n_port = 100000
metrics, weights_matrix = random_portfolios(n_port, mu, covmat)

plt.figure(figsize=(16,9))
plt.plot(metrics[1,:], metrics[0,:], 'o')#col 1=sd col 0=mu
plt.scatter(sd, mu, marker = 'x', color = 'r')
plt.title('Random Portfolios')
plt.xlabel('Volatilidad anualizada')
plt.ylabel('Retorno esperado anualizado')
plt.show()

