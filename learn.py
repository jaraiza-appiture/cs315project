from sklearn.tree import DecisionTreeClassifier
from sklearn.
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import seaborn as sns
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix

sns.set(rc={'figure.figsize':(10,10)})


def loadDF(filename='creditcard.csv'):
    return pd.read_csv(filename)


def makeScatterMat(data):

    attri = ['Time','V1','V2','V3',
             'V4','V5','V6','V7',
             'V8','V9','V10','V11',
             'V12','V13','V14','V15',
             'V16','V17','V18','V19',
             'V20','V21','V22','V23',
             'V24','V25','V26','V27',
             'V28','Amount','Class']

    scatter_matrix(data[attri],figsize=(30,20))

    plt.savefig('./scatter_mat.png')


def makeFeatureHistPlot(data):

    creditcard.hist(bins=50, figsize=(30,20),ylabelsize=5,xlabelsize=5,rwidth=5)

    plt.savefig('./featurehistplot.png')

    plt.show()

    return creditcard


def makeCorrelationHeatMap(data):

    cor = data.corr()

    # plot the heatmap
    htMap = sns.heatmap(cor,
        xticklabels=cor.columns,
        yticklabels=cor.columns,vmin=-1, vmax=1).set_title('Credit Card Correlation Heat Map')
    fig = htMap.get_figure()
    fig.savefig("./CorrHeatMap.png")



def confusionMatValues(y_true,y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tn,fp,fn,tp


def makePearsonCorrMat(X):

    return np.corrcoef(X)

def getImportantFeats(data):

    feats = ['Time','V1','V2','V3',
             'V4','V5','V6','V7',
             'V8','V9','V10','V11',
             'V12','V13','V14','V15',
             'V16','V17','V18','V19',
             'V20','V21','V22','V23',
             'V24','V25','V26','V27',
             'V28','Amount']

    dt = DecisionTreeClassifier()

    dt.fit(data[feats],data['Class'])

    N = 30
    ind = np.arange(N)
    width = 0.60

    plt.figure(figsize=(18, 10))
    plt.bar(ind, dt.feature_importances_, width)

    plt.ylabel('Gini Score')
    plt.title('Feature Importance')
    plt.xticks(ind,feats)

    plt.savefig('./featureImportance.png')
    plt.show()

def confusionMatGraph(tn,fp,fn,tp):

    array = np.array( [[tp, fn],
                       [fp, tn]])

    df_cm = pd.DataFrame(array, index=['fraud','non-fraud'], columns=['fraud','non-fraud'])
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)


if __name__ == '__main__':
    creditcard = loadDF()
    # makeScatterMat(creditcard)
    # makeCorrelationHeatMap(creditcard)
    # makeFeatureHistPlot(creditcard)
    # makeCorrelationMat()
    # makeConfusionMat()
    # getImportantFeats(creditcard)
    # confusionMatGraph(50,10,5,100)