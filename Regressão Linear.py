import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

wine = datasets.load_wine() 
wine.target.shape
wine.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:,:1], wine.target, test_size=0.4, random_state=0)

model = LinearRegression()

model.fit(X_train,y_train)

model.score(X_test,y_test)

model.coef_ 

model.intercept_

model.predict(X_test)

y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')


x = np.linspace(0, 3, 3)
y = x
plt.plot(x, y)
plt.show()


