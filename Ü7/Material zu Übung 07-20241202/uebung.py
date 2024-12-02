import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Data loading and inpection
# --------------------------

#TODO Load data from 'data.csv'
X = np.array([np.loadtxt("/home/bene/Documents/Code/kik-eki/Ü7/Material zu Übung 07-20241202/data.csv", delimiter=",", skiprows=1, usecols=0)])# REPLACE THIS
y = np.array([np.loadtxt("/home/bene/Documents/Code/kik-eki/Ü7/Material zu Übung 07-20241202/data.csv", delimiter=",", skiprows=1, usecols=1)])  # REPLACE THIS
print(X)
print(y)

plt.figure()
#TODO plot the figure
plt.show()
# Note: You have to close the figure manually in order to continue the execution of the remaining code


# Brute force approach
# --------------------

def sum_of_squared_error(y_hat, y):
    #TODO calculate the sum of squared error
    sose =(y_hat-y)**2 
    return sose

#TODO find the best m and t that minimize the error
m_candidates = np.linspace(-5.0, 5.0, 100)
t_candidates = np.linspace(-5.0, 5.0, 100)
for m_candidate in m_candidates:
    #TODO ... make a second loop over the candidate t values
    for t_candidate in t_candidates:  # REPLACE THIs
        
    y_hat = m_candidate * X + t_candidate
    error = sum_of_squared_error(y_hat, y)
    #TODO ... something with the error
#TODO find smallest of all the errors and know the corresponding m and t
m_hat = 0  # REPLACE THIS
t_hat = 0  # REPLACE THIS
print(f"TODO: f-string print statement with m_hat and t_hat")


# Library approach
# ----------------

# Linear regression with scikit-learn
# see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
model = None  # REPLACE THIS
#TODO fit the model
# Note: It might be that you get an error message telling you something about the shape of X. Follow the instructions given in the error message.
#m_hat = model.coef_[0]  # UNCOMMENT THIS
#t_hat = model.intercept_  # UNCOMMENT THIS
print(f"TODO: f-string print statement with m_hat and t_hat")  # same thing as above


# BLUE (best linear unbiased estimator
# ------------------------------------ 

# Change X from data matrix to the so-called design matrix
X = np.array([X, np.ones_like(X)]).T  # Note: T necessary because otherwise the shape of X would be (2, N) instead of (N, 2). Numpy is not always easy and intuitive
#TODO calculate the BLUE
beta_hat = np.array([0,1])  # REPLACE THIS
t_hat, m_hat = beta_hat
print(f"TODO: f-string print statement with m_hat and t_hat")  # same thing as above


# Usage of the model
# ------------------

x_q = 1.2
y_hat = m_hat * x_q + t_hat
print("Interpolation (x_q, y_hat):", x_q, y_hat)

x_q = 22.0
y_hat = m_hat * x_q + t_hat
print("Extrapolation (x_q, y_hat):", x_q, y_hat)
