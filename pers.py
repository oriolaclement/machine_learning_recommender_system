'''
import seaborn as sns
iris = sns.load_dataset('iris')
import seaborn as sns; sns.set()
sns.pairplot(iris, hue='species', height=3);
#from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size = 0.4, random_state=1
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
# Finding accuracy by comparing actual response values(y_test)with predicted response value(y_pred)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Providing sample data and the model will make prediction out of that data

sample = [[5, 5, 3, 2], [2, 4, 3, 5]]
preds = classifier_knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species)
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
dataset =pd.read_csv('C:/Users/user/Downloads/raw_data.zip')
dataset.head()
print(dataset)
zero_not_accepted = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna = True))
    dataset[column] = dataset[column].replace(np.NaN, mean)
    print(dataset)
X = dataset.iloc[:,0:3]
y = dataset.iloc[:,3]
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state = 0, test_size =0.2)
print(X_train,X_test,y_train,y_test)