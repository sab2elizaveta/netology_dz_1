import numpy as np
import pandas as pd 
import scipy.stats 
import matplotlib.pyplot as plt  
import seaborn as sns

data_sport = pd.read_csv('C:\\Users\\hia14\\OneDrive\\Рабочий стол\\Python\\athletes.csv',encoding='utf-8')

data_sport.drop(columns=['name', 'dob'], inplace=True) # удаляем переменную имени т.к она не влияет на обучение и переменную даты, чтобы модель корректно работала

print(data_sport.isna().sum()) #смотрим сколько и где пропущены значения

# заполняем пропуски медианным значением 

median_height = data_sport['height'].median()
data_sport['height'].fillna(median_height, inplace=True)
median_weight = data_sport['weight'].median()
data_sport['weight'].fillna(median_weight, inplace=True)

# преобразуем категор. переменные в числовые и подготавливаем X y
data_sport['sex'] = data_sport['sex'].map({'male': 0, 'female': 1}) # меняем значение целевой переменной на 0/1
y = data_sport['sex']
X_data = data_sport.drop('sex', axis=1)

X = pd.get_dummies(X_data, columns=['nationality', 'sport'], drop_first=True, dtype=int)

# для разбиения данных на тестовые и обучающие данные

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression # импортируем библиотеку с логистической регрессией
from sklearn.preprocessing import StandardScaler # для масштабирования данных

# масштабируем данные

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучаем модель

log_re = LogisticRegression()  
log_re.fit(X_train_scaled, y_train)

from sklearn.metrics import roc_curve, roc_auc_score

y_scores = log_re.predict_proba(X_test_scaled)[:,1]

# Получаем точки для графика
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label='Scikit-learn', color='blue')
plt.plot([0,1],[0,1], '--', label='Random predictions', color='red')
plt.legend()
plt.grid()
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic Curve by Scikit-learn')
plt.show()

# Считаем auc
auc_sk = roc_auc_score(y_test, y_scores)
print(auc_sk)

# Считаем "вручную"
def manual_metrics(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    
    manual_fpr = [0.0]
    manual_tpr = [0.0]
    manual_precision = [1.0] 
    
    for thr in thresholds:

        preds = (y_scores >= thr).astype(int)
        
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        
        manual_tpr.append(tpr)
        manual_fpr.append(fpr)
        manual_precision.append(precision)
        
    return np.array(manual_fpr), np.array(manual_tpr), np.array(manual_precision)

m_fpr, m_tpr, m_prec = manual_metrics(y_test, y_scores)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label='Scikit-learn', color='blue')
plt.plot(m_fpr, m_tpr, '--', lw=5, label='Scikit-learn', color='orange')
plt.plot([0,1],[0,1], '--', label='Random predictions', color='red')
plt.legend()
plt.grid()
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic Curve by Scikit-learn')
plt.show()
