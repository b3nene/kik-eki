import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

XLIM = (-10, 10)
YLIM = (-10, 10)
RES = 1000  # Resolution: number of points for the model line in the plot

# Prepare the global data and model variables:
x = np.array([])
y = np.array([])
# See https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
measurement_noise = 1.0
model = GaussianProcessRegressor(kernel=RBF(), alpha = measurement_noise**2)

def prepare_plot():
    plt.figure()
    plt.axis([*XLIM, *YLIM])
    add_gp_to_plot()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.connect('button_press_event', callback)

def add_data_to_plot():
    plt.errorbar(x, y, yerr=measurement_noise, fmt='ob')

def add_gp_to_plot():
    x_q = np.linspace(*XLIM, RES).reshape(-1, 1)  # query points
    y_hat, unc = model.predict(x_q, return_std=True)
    plt.plot(x_q, y_hat, '-r')
    # Plot a red area around the model line to indicate the uncertainty:
    plt.fill_between(x_q.flatten(), y_hat - unc, y_hat + unc, color='red', alpha=0.3, edgecolor='none')

def update_plot():
    plt.cla()  # clear axes
    add_data_to_plot()
    add_gp_to_plot()
    plt.axis([*XLIM, *YLIM])  # "forgotten" by figure, because axes were cleared
    plt.xlabel('x')
    plt.ylabel('y')
    plt.draw()  # refresh the plot

def update_model():
    model.fit(x.reshape(-1, 1), y)

def callback(event):
    global x, y  # Comment this out to see the error message which would occur without this line
    if event.inaxes:
        # Update the data:
        x = np.append(x, event.xdata)
        y = np.append(y, event.ydata)
        update_model()
        update_plot()


def main():
    prepare_plot()
    # Show the figure on screen and keep the python script running:
    plt.show()

if __name__ == '__main__':
    main()
