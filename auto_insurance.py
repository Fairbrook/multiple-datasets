import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Metrics
svr_prediction = svr.predict(X_test)
kn_prediction = kn.predict(X_test)
mlp_prediction = mlp.predict(X_test)

# R2
svr_r2 = r2_score(y_test, svr_prediction)
kn_r2 = r2_score(y_test, kn_prediction)
mlp_r2 = r2_score(y_test, mlp_prediction)
print("R2 SVR ", svr_r2)
print("R2 K-Vecinos ", kn_r2)
print("R2 Perceptrón ", mlp_r2)
print("")

# MSE
svr_mse = mean_squared_error(y_test, svr_prediction)
kn_mse = mean_squared_error(y_test, kn_prediction)
mlp_mse = mean_squared_error(y_test, mlp_prediction)
print("Error Medio SVR ", svr_mse)
print("Error Medio K-Vecinos ", kn_mse)
print("Error Medio Perceptrón ", mlp_mse)
print("")

# MAE
svr_mae = mean_absolute_error(y_test, svr_prediction)
kn_mae = mean_absolute_error(y_test, kn_prediction)
mlp_mae = mean_absolute_error(y_test, mlp_prediction)
print("Error Absoluto SVR ", svr_mae)
print("Error Absoluto K-Vecinos ", kn_mae)
print("Error Absoluto Perceptrón ", mlp_mae)
print("")

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
