from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


iris = datasets.load_wine()



X = iris.data[:, :12]
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,random_state=42)


knn.fit(X_train, y_train)
result = knn.predict(X_test)

print(iris.target_names[result])



print(confusion_matrix(y_test,result))  
print(classification_report(y_test,result))