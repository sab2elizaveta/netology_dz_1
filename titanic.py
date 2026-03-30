import pandas as pd
import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#считываем файлы  

titanic_test = pd.read_csv('C:\\Users\\hia14\\OneDrive\\Рабочий стол\\Python\\titanic_test.csv',encoding='utf-8')
titanic_train = pd.read_csv('C:\\Users\\hia14\\OneDrive\\Рабочий стол\\Python\\titanic_train.csv',encoding='utf-8')

#функцией отбираем только числовые значения и пишем в переменную 

num_train = titanic_train.select_dtypes(include=[np.number]).dropna()# удаляем пустые значения

X = num_train.drop('Survived', axis=1 ) # удаляем столбик Survived, чтобы модель его не подсмотрела
y = num_train['Survived'] #определяем переменную, которую будет угадывать модель 

#обучаем модель

LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(X, y)
LR_score = LR_model.score(X, y)
print(f"Точность LogisticRegression на сырых данных: {LR_score:.4f}")

print(titanic_train.isna().sum(), titanic_test.isna().sum())  #смотрим количество nan 

#удаляем все ненужные данные

colums_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin'] # PassengerId - ID пассажира не несет полезной информации, 
                                                         # Name/Ticket - уникальные значения, которые не представляют знчимости при спасении человека
                                                         # Cabin - много пропущенных значений (687 nan/ 204 непустых)
train_drop = titanic_train.drop(columns=colums_drop)
test_drop = titanic_test.drop(columns=colums_drop)                                                          

# считаем сколько данных уйдет если удалить пропуски

lost = ((1 - len(titanic_train.dropna()) / len(titanic_train) ) * 100)
print(f"Количество потерянных данных при удалении NaN: {lost:.2f}")


# заполняем пропуски

age_median = titanic_train['Age'].median()
fare_median = titanic_train['Fare'].median()
embarked_mode = titanic_train['Embarked'].mode()[0]

for data in (test_drop, train_drop):
    data['Age'] = data['Age'].fillna(age_median)
    data['Fare'] = data['Fare'].fillna(fare_median)
    data['Embarked'] = data['Embarked'].fillna(embarked_mode)

# переводим строчные данные в числовые

test = pd.get_dummies(test_drop, columns=['Sex', 'Embarked'], drop_first=True, dtype=int)
train = pd.get_dummies(train_drop, columns=['Sex', 'Embarked'], drop_first=True, dtype=int)

y_2 = train['Survived']
X_2 = train.drop('Survived', axis=1 )

X_test = test.reindex(columns=X_2.columns, fill_value=0)

LR_model_2 = LogisticRegression(max_iter=1000)
LR_model_2.fit(X_2, y_2)
LR_score_2 = accuracy_score(y_2, LR_model_2.predict(X_2)) * 100

print(f"Точность LogisticRegression на обработанных данных: {LR_score_2:.4f}")
