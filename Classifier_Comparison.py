import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Learning_Dataset/Social_Network_Ads.csv')
print(df.head())

X = df.iloc[:, :-1]
y = df.iloc[:,-1]

#Splitting into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def LogisticRegression_model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    Confusion_matrix(y_test, y_pred)

    Graph_Title = 'Logistic Regression (Test set)'
    Test_Visualization(X_test, y_test, classifier, Graph_Title)


def NaiveBayes_model(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    Confusion_matrix(y_test, y_pred)

    Graph_Title = 'SVM Classification (Test set)'
    Test_Visualization(X_test, y_test, classifier, Graph_Title)

def KNN_model(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    Confusion_matrix(y_test, y_pred)

    Graph_Title = 'K-NN Classification (Test set)'
    Test_Visualization(X_test, y_test, classifier, Graph_Title)

def SVM_model(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    Confusion_matrix(y_test, y_pred)

    Graph_Title = 'SVM Classification (Test set)'
    Test_Visualization(X_test, y_test, classifier, Graph_Title)

def DecisionTree_model(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion= 'entropy', random_state= 10)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    Confusion_matrix(y_test, y_pred)

    Graph_Title = 'Decision Tree Classification (Test set)'
    Test_Visualization(X_test, y_test, classifier, Graph_Title)

def RandomForest_model(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    Confusion_matrix(y_test, y_pred)

    Graph_Title = 'SVM Classification (Test set)'
    Test_Visualization(X_test, y_test, classifier, Graph_Title)

def Confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

def Test_Visualization(X, y, classifier, Title):
    from matplotlib.colors import ListedColormap
    X_set, y_set = sc.inverse_transform(X), y
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(("salmon", "lightblue")))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(("red", "blue"))(i), label=j)
    plt.title(Title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

DecisionTree_model(X_train, X_test, y_train, y_test)
LogisticRegression_model(X_train, X_test, y_train, y_test)
KNN_model(X_train, X_test, y_train, y_test)
SVM_model(X_train, X_test, y_train, y_test)
RandomForest_model(X_train, X_test, y_train, y_test)
NaiveBayes_model(X_train, X_test, y_train, y_test)


