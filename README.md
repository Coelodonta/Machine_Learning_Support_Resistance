# Machine_Learning_Support_Resistance
Finding support and resistance levels for trading using unsupervised learning with sklearn and mplfinance

Here are several sample scripts demonstrating ways to automatically find support/resistance levels for stocks using unsupervised machine learning. 

Most implementations I've seen use the K-Means clustering algorithm. It seems to be a decent default algorithm for many use cases, but the quality of the results will vary greatly with number of samples, variance, bias, selection of k and other factors. For that reason, I created alternative implementations using other clustering algorithms that may be more appropriate for some use cases:

- Mean Shift
- Birch
- Affinity Propagation
- Mini-batch K-Means

In addition to generating graphs, the quality of the clusters created by the various algorithms, i.e. how "tight" and well separated they are, is evaulated using silhouette scoring.

This project was inspired by https://github.com/judopro/Stock_Support_Resistance_ML and https://github.com/james-ott-csuglobal/KNN_Auto_Supp_Resis.

I have also updated the graph plotting to use mplfinance, since some of the dependencies used in the above projects have been deprecated.

Imports:
- numpy
- pandas
- matplotlib.pyplot
- sklearn.cluster
- sklearn.metrics
- yfinance
- mplfinance
