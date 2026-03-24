import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns


data_iris = load_iris()
iris = pd.DataFrame(data=data_iris.data, columns=data_iris.feature_names)

iris['iris_kind'] = pd.Categorical.from_codes(data_iris.target, data_iris.target_names)
iris['target'] = data_iris.target 


iris = iris.query("iris_kind == 'versicolor' or iris_kind == 'virginica'").copy()


X = iris.drop(columns=['iris_kind', 'target']).values
y = iris['target'].values


y = np.where(y == 1, 0, 1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def fit_gd(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    for epoch in range(epochs):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
    
        gradient = np.dot(X.T, (y_pred - y)) / n_samples
        weights -= lr * gradient
        
    return weights

def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights))
    return (probabilities >= 0.5).astype(int)


final_weights = fit_gd(X_train, y_train, lr=0.1, epochs=1000)
y_pred = predict(X_test, final_weights)
accuracy = accuracy_score(y_test, y_pred)

print(f"Вес: {final_weights}")
print(f"Точность(Accuracy) : {accuracy:.4f}")
