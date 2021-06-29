import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
import mplfinance as mpf
from sklearn.metrics import silhouette_score

def load_prices(ticker, period="6mo",interval="1d"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data.dropna()

def get_optimum_clusters(df, saturation_point=0.05):
    wcss = []
    k_models = []
    size = min(11, len(df.index))
    for i in range(1, size):
        kmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=None)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # View inertia - good for electing the saturation point
    print(wcss)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss)-1
    for i in range(0, len(wcss)-1):
        diff = abs(wcss[i+1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    print("Optimum K is " + str(optimum_k + 1))
    optimum_clusters = k_models[optimum_k]

    return optimum_clusters

symbol = 'TSLA'

data = load_prices(symbol, "1mo")
#data = load_prices(symbol, "1wk","5m")
#data = load_prices(symbol, "3mo")
#data = load_prices(symbol, "6mo")
#data = load_prices(symbol, "1y")

lows = pd.DataFrame(data=data, index=data.index, columns=["Low"])
highs = pd.DataFrame(data=data, index=data.index, columns=["High"])

low_clusters = get_optimum_clusters(lows)
low_centers = low_clusters.cluster_centers_
low_centers = np.sort(low_centers, axis=0)

high_clusters = get_optimum_clusters(highs)
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
