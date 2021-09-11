from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_wine()


X = iris.data[:, :1]
y = iris.target

from sklearn.naive_bayes import GaussianNB
r = GaussianNB()

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25,random_state=42)


r.fit(X_train, y_train)
result = r.predict(X_test)
print(iris.target_names[result])

#print(result)
print(confusion_matrix(y_test,result))  
print(classification_report(y_test,result))