# pip install numpy scikit-learn matplotlib
import sklearn.neighbors
import numpy as np
import sklearn

MODEL = 'knn'  # 'knn' or 'nb'

X = []  # dummy until code is implemented
y = []  # dummy until code is implemented
X_q = []  # dummy until code is implemented
#TODO Load X, y and x_q from csv files.
# Hint: Use np.loadtxt() https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html

X = np.loadtxt("/home/bene/Documents/Code/kik-eki/Ü4/X.csv", skiprows=1)
y = np.loadtxt("/home/bene/Documents/Code/kik-eki/Ü4/y.csv", skiprows=1)
X_q = np.loadtxt("/home/bene/Documents/Code/kik-eki/Ü4/test.csv")

print(X)
print(y)
print(X_q)

print("#####################################")

from sklearn.neighbors import KNeighborsClassifier

#TODO Fit a kNN classifier to (X,y)
# Hint: Use KNeighborsClassifier from sklearn https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=2)
model.fit(X,y)


y_hat = [model.predict(X_q)] # the German "Dach" symbol ^ is used to denote the predicted value and is called "hat" in English
#TODO Predict the class of the query points x_q
# Hint: Use the predict() method of the model

print(y_hat)

# TODO Print the species names corresponding to the y_hat values
# Hint: If you don't know the names, read them in the header line in y.csv
print("species: 0=Setosa, 1=Versicolor, 2=Virginica")

#exit()  # Remove this command when you implemented everything above
# -----------------------------------------------------------------
# Below this line: Prepared stuff for visualization,
# with options we might need in the next lecture.
# Is it Clean Code? No.
#
# Nothing to do here for the students.

if 'MODEL' not in locals():
    MODEL = 'unk'

# IMPORTANT: pip install PyQt6 before plotting!

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# Define function predict_with_doubt() if such a function has not been defined yet:
if MODEL == 'nb':
    if 'predict_with_doubt' not in locals():
        def predict_with_doubt(model, X_q, threshold):
            probs = model.predict_proba(X_q)
            # In each line, if the highest probability is below the threshold, we return 3, else the class index (argmax):
            y_hat = np.where(probs.max(axis=1) < threshold, 3, probs.argmax(axis=1))
            return y_hat

# Own function for equal aspect ratio, because axis('equal') leaves the axis limits unchanged
def axisequal():
    # Get the current limits of the plot
    data_xlim = plt.gca().get_xlim()
    data_ylim = plt.gca().get_ylim()

    # Calculate the data range for both axes
    x_range = data_xlim[1] - data_xlim[0]
    y_range = data_ylim[1] - data_ylim[0]

    # Determine the central point and the maximum range to make the plot square
    x_center = (data_xlim[0] + data_xlim[1]) / 2
    y_center = (data_ylim[0] + data_ylim[1]) / 2
    max_range = max(x_range, y_range) / 2

    # Adjust limits to make the aspect equal
    adjusted_xlim = (x_center - max_range, x_center + max_range)
    adjusted_ylim = (y_center - max_range, y_center + max_range)

    # Apply the adjusted limits
    plt.gca().set_xlim(adjusted_xlim)
    plt.gca().set_ylim(adjusted_ylim)

# For models such as NB, get likelihood rather than posterior:
def calculate_likelihoods(X_q, model):
    likelihoods = []
    for x in X_q:
        class_likelihoods = []
        for i, _ in enumerate(model.classes_):
            mean = model.theta_[i]
            var = model.var_[i] 
            # Gaussian likelihood:
            likelihood = np.prod((1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var))
            class_likelihoods.append(likelihood)
        likelihoods.append(class_likelihoods)
    return np.array(likelihoods)

plt.figure()

# Create a scatter plot of the datapoints and the query points:
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='s', label='Versicolor')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='blue', marker='d', label='Virginica')
plt.scatter(X_q[:, 0], X_q[:, 1], color='black', marker='x', label='?')
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()

# Embellishments:
plt.grid(True, which='both')
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.1))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
axisequal()
plt.gca().set_axisbelow(True)  # Ensure gridlines are below markers
# Get axes limits from plot to keep for future plots:
xlim = plt.xlim()
ylim = plt.ylim()

# Create a meshgrid with those limits and predict the classes of its points:
RES = 0.01
x0x0_q, x1x1_q = np.meshgrid(
    np.arange(xlim[0], xlim[1], RES),
    np.arange(ylim[0], ylim[1], RES)
)
if MODEL == 'nb' or MODEL == 'mvb':  # assuming we have then also implemented the fourth class "don't know" and the custom prediciton function
    cc = predict_with_doubt(model, np.c_[x0x0_q.ravel(), x1x1_q.ravel()], threshold=0.5).reshape(x0x0_q.shape)
else:
    cc = model.predict(np.c_[x0x0_q.ravel(), x1x1_q.ravel()]).reshape(x0x0_q.shape)

PRED_NOT_PROB = True  # Show prediction and not probabilistic details
CLASS_TO_SHOW = 0  # class idx, used if probability output is shown. -1 to have the maximum
POSTERIOR_NOT_LIKELIHOOD = True  # used if probability output is shown: which one

if PRED_NOT_PROB:
    # Show the predictions over the feature space:
    cmap = ListedColormap(['red', 'green', 'blue', 'white'])
    plt.contourf(x0x0_q, x1x1_q, cc, alpha=0.3, cmap=cmap, vmin=0, vmax=3)
else:
    # Alternatively, show probability of specific class:
    if MODEL == 'nb' or MODEL == 'mvb':
        if POSTERIOR_NOT_LIKELIHOOD:
            if CLASS_TO_SHOW < 0:
                # Take maximum probability of the three classes:
                pp = model.predict_proba(np.c_[x0x0_q.ravel(), x1x1_q.ravel()]).max(axis=1).reshape(x0x0_q.shape)
            else:  # The usual case
                # Select the CLASS_TO_SHOW probability:
                pp = model.predict_proba(np.c_[x0x0_q.ravel(), x1x1_q.ravel()])[:, CLASS_TO_SHOW].reshape(x0x0_q.shape)
            plt.title(f"Posterior probability of class {CLASS_TO_SHOW} (black to white)")
        else:  # LIKELIHOOD
            if MODEL == 'nb':
                # Here, we use the helper function defined above
                if CLASS_TO_SHOW < 0:
                    # Take maximum likelihood of the three classes:
                    pp = calculate_likelihoods(np.c_[x0x0_q.ravel(), x1x1_q.ravel()], model).max(axis=1).reshape(x0x0_q.shape)
                else:  # The usual case
                    # Select the CLASS_TO_SHOW likelihood:
                    pp = calculate_likelihoods(np.c_[x0x0_q.ravel(), x1x1_q.ravel()], model)[:, CLASS_TO_SHOW].reshape(x0x0_q.shape)
            if MODEL == 'mvb':
                # Here, we implemented it directly in the model
                if CLASS_TO_SHOW < 0:
                    # Take maximum likelihood of the three classes:
                    pp = model.predict_likelihoods(np.c_[x0x0_q.ravel(), x1x1_q.ravel()]).max(axis=1).reshape(x0x0_q.shape)
                else:  # The usual case
                    # Select the CLASS_TO_SHOW likelihood:
                    pp = model.predict_likelihoods(np.c_[x0x0_q.ravel(), x1x1_q.ravel()])[:, CLASS_TO_SHOW].reshape(x0x0_q.shape)
            plt.title(f"Likelihood of class {CLASS_TO_SHOW} (black to white)")
        plt.contourf(x0x0_q, x1x1_q, pp, alpha=1.0, cmap='gray', vmin=0, vmax=1, levels=32, zorder=-1)
    else:
        print("\033[93mNot implemented.\033[0m")

plt.show()
