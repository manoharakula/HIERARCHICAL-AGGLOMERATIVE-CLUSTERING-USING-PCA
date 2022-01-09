#Manohar Akula
#ASU ID: 1223335191
#EEE 511- Extra Credit Assignment

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics


data = np.array(pd.read_csv(r'C:\Users\akula\OneDrive\Desktop\yourdata.csv'))
scaler = StandardScaler()
data = scaler.fit_transform(data)
pca = PCA(2)
data = pca.fit_transform(data)

for no_of_clusters in [3,5]:
    X = data
    model = AgglomerativeClustering(n_clusters= no_of_clusters, affinity='euclidean', linkage='ward')
    model.fit(X)
    labels = model.labels_
    ax1 = plt.subplot(2, 1, 1)
    for k in range(0,no_of_clusters):
        ax1.scatter(X[labels== k, 0], X[labels==k, 1])
    ax1.legend()
    ax1.set_title("AgglomerativeClustering")

    X = data
    kmeans = KMeans(n_clusters=no_of_clusters)
    kmeans.fit(X)
    predictions = kmeans.predict(X)
    frame = pd.DataFrame(X)
    frame['C'] = predictions
    frame.columns = ['X1', 'X2', 'C']
    ax2 = plt.subplot(2, 1, 2)
    for k in range(0,no_of_clusters):
        X = frame[frame["C"]==k]
        ax2.scatter(X["X1"],X["X2"])
    ax2.legend()
    ax2.set_title("KMeans")
    plt.show()
