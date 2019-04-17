from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from prep import loadData

def makeConfusionMat(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    return tn,fp,fn,tp

def makePearsonCorrMat(X):
    return np.corrcoef(X)
if __name__ == '__main__':
    pass