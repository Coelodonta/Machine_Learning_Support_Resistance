import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt
import mplfinance as mpf
from sklearn.metrics import silhouette_score

def load_prices(ticker, period="6mo",interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data.dropna()


symbol = 'AAPL'

data = load_prices(symbol, "1mo")
#data = load_prices(symbol, "1wk","5m")
#data = load_prices(symbol, "3mo")
#data = load_prices(symbol, "6mo")
#data = load_prices(symbol, "1y")

lows = pd.DataFrame(data=data, index=data.index, columns=["Low"])
highs = pd.DataFrame(data=data, index=data.index, columns=["High"])

low_clusters = AffinityPropagation(max_iter=200, random_state=None).fit(lows)
low_centers = low_clusters.cluster_centers_
low_centers = np.sort(low_centers, axis=0)

high_clusters = AffinityPropagation(max_iter=200,random_state=None).fit(highs)
high_centers = high_clusters.cluster_centers_
high_centers = np.sort(high_centers, axis=0)

# How good are the clusters?
low_score=silhouette_score(lows,low_clusters.labels_)
high_score=silhouette_score(highs,high_clusters.labels_)
print(f"Silhouette score Lows: {low_score} Highs: {high_score}")

lowss = []
highss = []
finals = []

rounding_factor = 2

for i in low_centers:
  i = round(float(i),rounding_factor)
  lowss.append(i)

for i in high_centers:
  i = round(float(i),rounding_factor)
  highss.append(i)

print('lows/support: ', lowss)
print('highs/resistance: ', highss)


# Plotting
plt.style.use('fast')
ohlc = data.loc[:, ['Open', 'High', 'Low', 'Close']]
fig, ax = mpf.plot(ohlc.dropna(), type = 'candle', style='charles', show_nontrading = False,returnfig = True,
                  ylabel='Price',title=symbol)

for low in low_centers[:9]:
    ax[0].axhline(low[0], color='green', ls='-', alpha=.2)

for high in high_centers[-9:]:
    ax[0].axhline(high[0], color='red', ls='-', alpha=.1)

plt.show()
