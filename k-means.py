from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_wine()

X = iris.data[:, :1]
y = iris.target

km = KMeans(n_clusters = 3, random_state=20)
km.fit(X,y)
resultado = km.predict([[1]])
centers = km.cluster_centers_

#print(resultado)
print(iris.target_names[resultado])