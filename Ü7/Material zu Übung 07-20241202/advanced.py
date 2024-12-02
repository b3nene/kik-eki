import numpy as np
from matplotlib import pyplot as plt

# See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# Prepare the plot
# ----------------
plt.figure()
plt.axis([-10,10,-10,10])
plt.xlabel('x')
plt.ylabel('y')

# Plot the initial forecast by our model
# - - - - - - - - - - - - - - - - - - - 
measurement_noise = 1.0
model = GaussianProcessRegressor(kernel=RBF(), alpha = measurement_noise**2)
x_q = np.linspace(-10,10, 1000).reshape(-1, 1)  # query points
y_hat, unc = model.predict(x_q, return_std=True)
plt.plot(x_q, y_hat, '-r')
# Plot a red area around the model line, between y_hat - unc and y_hat + unc to indicate the uncertainty:
#TODO


# Interactivity
# -------------
def callback(event):
    pass
        
#TODO Put all this correctly into the the callback function and use plt.connect('button_press_event', callback)
x = np.array([])
y = np.array([])
# Suppose the user clicks on the plot at the following points:
xdata = 4.5
ydata = 7.5

# Update data
# - - - - - -
x = np.append(x, xdata)
y = np.append(y, ydata)

# Update model
# - - - - - - -
model.fit(x.reshape(-1, 1), y)
x_q = np.linspace(-10,10, 1000).reshape(-1, 1)  # query points
y_hat, unc = model.predict(x_q, return_std=True)

# Update plot
# - - - - - -
plt.cla()  # clear axes
plt.plot(x, y, 'ob')
#TODO: Replace plot by errorbar plot in order to show the measurement noise as error bars
plt.plot(x_q, y_hat, '-r')
# Plot a red area around the model line, between y_hat - unc and y_hat + unc to indicate the uncertainty:
#TODO
plt.axis([-10,10,-10,10])  # "forgotten" by figure, because axes were cleared
plt.xlabel('x') # "forgotten" by figure, because axes were cleared
plt.ylabel('y') # "forgotten" by figure, because axes were cleared
plt.draw()  # refresh the plot


# Show the figure on screen and keep the python script running
# ------------------------------------------------------------
plt.show()


#TODO Re-structure the whole code into several functions, so that it is more modular and easier to understand
