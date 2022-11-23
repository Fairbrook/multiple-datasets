import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Get the dataset
url = 'https://raw.githubusercontent.com/sharmaroshan/Zoo-Dataset/master/zoo.csv'
df = pd.read_csv(url, sep=',', header='infer',  engine='python')

# Separate data from labels
X = df.drop(columns=["class_type", "animal_name"], axis=1).to_numpy()
y = df["class_type"].to_numpy()

# Separate training from test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

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

# Metrics
svm_prediction = svm.predict(X_test)
kn_prediction = kn.predict(X_test)
lgr_prediction = lgr.predict(X_test)
gnb_prediction = gnb.predict(X_test)
mlp_prediction = mlp.predict(X_test)

# Accuary
svm_accuracy = accuracy_score(y_test, svm_prediction)
kn_accuracy = accuracy_score(y_test, kn_prediction)
lgr_accuracy = accuracy_score(y_test, lgr_prediction)
gnb_accuracy = accuracy_score(y_test, gnb_prediction)
mlp_accuracy = accuracy_score(y_test, mlp_prediction)
print("Accuracy Maquinas de vector soporte: ", svm_accuracy)
print("Accuracy K-Vecinos", kn_accuracy)
print("Accuracy Regresión logística", lgr_accuracy)
print("Accuracy Bayesian", gnb_accuracy)
print("Accuracy Perceptrón", mlp_accuracy)
print("")

# Precision
svm_precision = precision_score(
    y_test, svm_prediction, average='weighted', zero_division=0)
kn_precision = precision_score(
    y_test, kn_prediction, average='weighted', zero_division=0)
lgr_precision = precision_score(
    y_test, lgr_prediction, average='weighted', zero_division=0)
gnb_precision = precision_score(
    y_test, gnb_prediction, average='weighted', zero_division=0)
mlp_precision = precision_score(
    y_test, mlp_prediction, average='weighted', zero_division=0)
print("Precision Maquinas de vector soporte: ", svm_precision)
print("Precision K-Vecinos", kn_precision)
print("Precision Regresión logística", lgr_precision)
print("Precision Bayesian", gnb_precision)
print("Precision Perceptrón", mlp_precision)
print("")

# F1
svm_f1 = f1_score(y_test, svm_prediction, average='weighted')
kn_f1 = f1_score(y_test, kn_prediction, average='weighted')
lgr_f1 = f1_score(y_test, lgr_prediction, average='weighted')
gnb_f1 = f1_score(y_test, gnb_prediction, average='weighted')
mlp_f1 = f1_score(y_test, mlp_prediction, average='weighted')
print("F1 Maquinas de vector soporte: ", svm_f1)
print("F1 K-Vecinos", kn_f1)
print("F1 Regresión logística", lgr_f1)
print("F1 Bayesian", gnb_f1)
print("F1 Perceptrón", mlp_f1)
print("")

# Recall
svm_recall = recall_score(y_test, svm_prediction, average='weighted')
kn_recall = recall_score(y_test, kn_prediction, average='weighted')
lgr_recall = recall_score(y_test, lgr_prediction, average='weighted')
gnb_recall = recall_score(y_test, gnb_prediction, average='weighted')
mlp_recall = recall_score(y_test, mlp_prediction, average='weighted')
print("Sensitivity Maquinas de vector soporte: ", svm_recall)
print("Sensitivity K-Vecinos", kn_recall)
print("Sensitivity Regresión logística", lgr_recall)
print("Sensitivity Bayesian", gnb_recall)
print("Sensitivity Perceptrón", mlp_recall)


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
