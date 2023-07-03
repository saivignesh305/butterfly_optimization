import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from butterflyoptim import ButterflyOptimization


# Load the diabetes dataset
data = np.loadtxt('diabetes.csv', delimiter=',')

# Split the dataset into features (X) and target variable (y)
X = data[:, :-1]
y = data[:, -1]

# Define the SVM classifier and its parameters
svm = SVC(kernel='linear', C=1, random_state=42)

# Define the objective function to optimize
def objective_function(X, y, svm):
    score = np.mean(cross_val_score(svm, X, y, cv=5))
    return -score  # Negative sign because butterfly optimization minimizes the objective function

# Define the bounds for the search space (min and max values for each feature)
bounds = [(0, 1)] * X.shape[1]

# Define the butterfly optimization algorithm
bo = ButterflyOptimization(bounds, objective_function, n_iter=50, n_butterflies=20)

# Run the optimization
best_params, best_score = bo.optimize(X, y, svm)

# Print the best parameters and score
print('Best parameters:', best_params)
print('Best score:', -best_score)  # Convert back to positive value
