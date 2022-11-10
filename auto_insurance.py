import pandas as pd
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np

# Get the dataset
url = 'https://raw.githubusercontent.com/hargurjeet/MachineLearning/Swedish-Auto-Insurance-Dataset/insurance.csv'
df_raw = pd.read_csv(url, sep='delimiter', header=None,  engine='python')
df_data = df_raw.drop([0, 1, 2, 3], axis=0).reset_index(drop=True)

# Split it in columns
df: pd.DataFrame = df_data[0].str.split(
    ',', expand=True).rename(columns={0: 'Claims', 1: 'Payment'})
df.Claims = pd.to_numeric(df.Claims, errors="coerce")
df.Payment = pd.to_numeric(df.Payment, errors="coerce")

# Shape it to sklearn required shape
x = np.reshape(df.Claims.to_numpy(), (-1, 1))
y = df.Payment.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# Support vector regression
svr = SVR(kernel="linear", C=500, gamma="auto", epsilon=0.1)
svr.fit(X_train, y_train)

# K-Neighbors
kn = KNeighborsRegressor(15, weights="distance")
kn.fit(X_train, y_train)

# MLP
mlp = MLPRegressor((100, 100), random_state=1, max_iter=500)
mlp.fit(X_train, y_train)

# Plot data and predictions curves
plt.figure(figsize=(9, 9))
plt.suptitle("Evalución de dataset de prueba")
plt.scatter(X_test,
            y_test, facecolor="none", edgecolor="b", label="Otros puntos")
range = np.arange(X_test.min(), X_test.max()).reshape(-1, 1)
plt.plot(range, svr.predict(range),
         color="c", label="Regresión vector soporte")
plt.plot(range, kn.predict(range),
         color="b", label="K-Vecinos")
plt.plot(range, mlp.predict(range),
         color="r", label="Perceptrón multi capa")
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=1,
    fancybox=True,
    shadow=True,
)
plt.show()
