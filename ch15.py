import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

utilities_df = pd.read_csv('Utilities.csv')
# set row names to the utilities column
utilities_df.set_index('Company', inplace=True)
# while not required, the conversion of integer data to float
# will avoid a warning when applying the scale function
utilities_df = utilities_df.apply(lambda x: x.astype('float64'))
# compute Euclidean distance
d = pairwise.pairwise_distances(utilities_df,
metric='euclidean')
pd.DataFrame(d, columns=utilities_df.index,
index=utilities_df.index)

# code for normalizing data and computing distance
# scikit-learn uses population standard deviation
utilities_df_norm = utilities_df.apply(preprocessing.scale,
axis=0)
# pandas uses sample standard deviation
utilities_df_norm = (utilities_df - utilities_df.mean())/utilities_df.std()
# compute normalized distance based on Sales and Fuel Cost
utilities_df_norm[['Sales', 'Fuel_Cost']]
d_norm = pairwise.pairwise_distances(utilities_df_norm[['Sales', 'Fuel_Cost']], metric='euclidean')
pd.DataFrame(d_norm, columns=utilities_df.index, index=utilities_df.index)

# code for running hierarchical clustering and generating a dendrogram
# in linkage() set argument method =
# 'single', 'complete', 'average', 'weighted', centroid', 'median', 'ward'
Z = linkage(utilities_df_norm, method='single')
dendrogram(Z, labels=utilities_df_norm.index,
color_threshold=2.75)
Z = linkage(utilities_df_norm, method='average')
dendrogram(Z, labels=utilities_df_norm.index,
color_threshold=3.6)


# Single Linkage (output modified for clarity)
memb = fcluster(linkage(utilities_df_norm, method='single'), 6, criterion='maxclust')
memb = pd.Series(memb, index=utilities_df_norm.index)

# Average Linkage (output modified for clarity)
memb = fcluster(linkage(utilities_df_norm,method='average'), 6, criterion='maxclust')
memb = pd.Series(memb, index=utilities_df_norm.index)


# code for creating heatmap
# set labels as cluster membership and utility name
utilities_df_norm.index = ['{}: {}'.format(cluster, state) for cluster, state in zip(memb,utilities_df_norm.index)]
# plot heatmap
# the '_r' suffix reverses the color mapping to large = dark
sns.clustermap(utilities_df_norm, method='average',
col_cluster=False, cmap='mako_r')




# code for k-means
# Load and preprocess data
utilities_df = pd.read_csv('Utilities.csv')
utilities_df.set_index('Company', inplace=True)
utilities_df = utilities_df.apply(lambda x:
x.astype('float64'))
# Normalize distances
utilities_df_norm = utilities_df.apply(preprocessing.scale,
axis=0)
kmeans = KMeans(n_clusters=6,
random_state=0).fit(utilities_df_norm)
# Cluster membership
memb = pd.Series(kmeans.labels_, index=utilities_df_norm.index)
for key, item in memb.groupby(memb):
  print(key, ': ', ', '.join(item.index))

# Within-cluster sum of squared distances and cluster count
# calculate the distances of each data point to the cluster centers
distances = kmeans.transform(utilities_df_norm)
# find closest cluster for each data point
minSquaredDistances = distances.min(axis=1) ** 2
# combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 'cluster': kmeans.labels_}, index=utilities_df_norm.index)
# group by cluster and print information
for cluster, data in df.groupby('cluster'):
  count = len(data)
  withinClustSS = data.squaredDistance.sum()
  print(f'Cluster cluster (count members):  withinClustSS:.2f within cluster ')

  
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=utilities_df_norm.columns)
pd.set_option('precision', 3)
# calculate the distances of each data point to the cluster centers
distances = kmeans.transform(utilities_df_norm)
# find closest cluster for each data point
minSquaredDistances = distances.min(axis=1) ** 2
# combine with cluster labels into a data frame
df = pd.DataFrame({'squaredDistance': minSquaredDistances, 'cluster': kmeans.labels_}, index=utilities_df_norm.index)
# group by cluster and print information
for cluster, data in df.groupby('cluster'):
  count = len(data)
  withinClustSS = data.squaredDistance.sum()
  print(f'Cluster cluster (count members):withinClustSS:.2f within cluster ')
  
  
  
# code for plotting profile plot of centroids
centroids['cluster'] = ['Cluster '.format(i) for i in centroids.index]
plt.figure(figsize=(10,6))
parallel_coordinates(centroids, class_column='cluster',
colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

## code for preparing Figure 15.6
inertia = []
for n_clusters in range(1, 7):
  kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(utilities_df_norm)
  inertia.append(kmeans.inertia_ / n_clusters)
inertias = pd.DataFrame({'n_clusters': range(1, 7), 'inertia': inertia})
ax = inertias.plot(x='n_clusters', y='inertia')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.ylim((0, 1.1 * inertias.inertia.max()))
ax.legend().set_visible(False)
plt.show()

