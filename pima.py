import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Get the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
df = pd.read_csv(url, sep=',', header=None,  engine='python')

# Separate data from labels
X = df.drop(columns=8, axis=1).to_numpy()
y = df[8].to_numpy()

print(X)


# Separate training from test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#print(train_X)
#print(train_Y)
# SVM
svm = SVC()
svm.fit(X_train, y_train)

# K-Neighbors
kn = KNeighborsClassifier(15, weights="distance")
kn.fit(X_train, y_train)

# LogisticRegression
lgr = LogisticRegression(
    C=0.01, penalty="elasticnet", solver="saga", l1_ratio=0.5, tol=0.01
)
lgr.fit(X_train, y_train)

# Naive Bayesian
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# MLP
mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(
    130, 250), alpha=0.01, solver="lbfgs", random_state=1, max_iter=500))
mlp.fit(X_train, y_train)

# Plot data and predictions curves
fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 12, figure=fig)
ax1 = fig.add_subplot(gs[0, 0:3])
ax2 = fig.add_subplot(gs[0, 4:7])
ax3 = fig.add_subplot(gs[0, 8:11])
ax4 = fig.add_subplot(gs[1, 2:5])
ax5 = fig.add_subplot(gs[1, 6:9])
ax1.set_title("Support Vector Machine")
ConfusionMatrixDisplay.from_estimator(svm, X_test, y_test, ax=ax1)
ax2.set_title("K-Vecinos")
ConfusionMatrixDisplay.from_estimator(kn, X_test, y_test, ax=ax2)
ax3.set_title("Regresion logística")
ConfusionMatrixDisplay.from_estimator(lgr, X_test, y_test, ax=ax3)
ax4.set_title("Naive Bayesian")
ConfusionMatrixDisplay.from_estimator(gnb, X_test, y_test, ax=ax4)
ax5.set_title("Perceptrón multicapa")
ConfusionMatrixDisplay.from_estimator(mlp, X_test, y_test, ax=ax5)
fig.suptitle("Test dataset")
plt.show()
