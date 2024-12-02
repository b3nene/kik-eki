import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Data loading and inspection
# --------------------------

#TODO Load data from 'data.csv'
data = pd.read_csv('/home/bene/Documents/Code/kik-eki/Ü7/Material zu Übung 07-20241202/data.csv')
X = data['X']
y = data['y']
# ODER
# X = np.loadtext('data_csv')
plt.figure()
plt.scatter(X, y, label='Datenpunkte', color='blue')
plt.plot(X, y, label='Verbindungslinie', color='red')
plt.title('Plot von X und y')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
# Note: You have to close the figure manually in order to continue the execution of the remaining code


# Brute force approach
# --------------------

def sum_of_squared_error(y_hat, y):
    #TODO calculate the sum of squared error
    sse = np.sum((np.array(y_hat) - np.array(y)) ** 2)
    return sse

#TODO find the best m and t that minimize the error
m_candidates = np.linspace(-5.0, 5.0, 100)
t_candidates = np.linspace(-5.0, 5.0, 100)
min_error = float('inf')  # Start with a large error value to find the minimum
best_m, best_t = 0, 0  # Initial values for m and t

for m_candidate in m_candidates:
    #TODO make a second loop over the candidate t values
    for t_candidate in t_candidates:
        y_hat = m_candidate * X + t_candidate
        error = sum_of_squared_error(y_hat, y)
        #TODO something with the error
        if error < min_error:
            min_error = error
            best_m = m_candidate
            best_t = t_candidate

#TODO find smallest of all the errors and know the corresponding m and t
m_hat = best_m
t_hat = best_t
print(f"Brute force: m_hat = {m_hat}, t_hat = {t_hat}")


# Library approach
# ----------------

# Linear regression with scikit-learn
# see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
model = LinearRegression()  # REPLACE THIS with LinearRegression model
#TODO fit the model
model.fit(X.values.reshape(-1, 1), y)  # Reshaping X for the model, as sklearn expects a 2D array for the feature
m_hat = model.coef_[0]  # UNCOMMENT THIS
t_hat = model.intercept_  # UNCOMMENT THIS
print(f"Library approach: m_hat = {m_hat}, t_hat = {t_hat}")  # same thing as above


# BLUE (best linear unbiased estimator)
# ------------------------------------ 

# Change X from data matrix to the so-called design matrix
X = np.array([X, np.ones_like(X)]).T  # Note: T necessary because otherwise the shape of X would be (2, N) instead of (N, 2). Numpy is not always easy and intuitive
#TODO calculate the BLUE (Beta_hat)
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y  # This is the formula for the Best Linear Unbiased Estimator (BLUE)
t_hat, m_hat = beta_hat
print(f"BLUE: m_hat = {m_hat}, t_hat = {t_hat}")  # same thing as above


# Usage of the model
# ------------------

x_q = 1.2
y_hat = m_hat * x_q + t_hat
print("Interpolation (x_q, y_hat):", x_q, y_hat)

x_q = 22.0
y_hat = m_hat * x_q + t_hat
print("Extrapolation (x_q, y_hat):", x_q, y_hat)