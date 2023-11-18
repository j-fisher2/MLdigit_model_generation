from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import joblib
X,y=fetch_openml('mnist_784',version=1,return_X_y=True,parser='auto')

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2) #make training and test set

try:
    mlp=joblib.load('mlp_model.pkl')
    print("loaded existing model")

except FileNotFoundError:
    mlp=MLPClassifier()
    mlp.fit(X_train,y_train)
    joblib.dump(mlp,'mlp_model.pkl')

def score_model(X_test,y_test):
    return mlp.score(X_test,y_test)

print(score_model(X_test,y_test))