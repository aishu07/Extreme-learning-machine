



import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class ELM:
    def __init__(self, input_size, hidden_size, output_size, pop_size):
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        np.random.seed(42)  # Set random seed for reproducibility
        self.input_weights = np.random.randn(input_size, hidden_size)
        self.bias = np.random.randn(hidden_size)
        self.output_weights = None

    def train(self, X_train, y_train):
        # Add bias to input
        X_train_biased = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

        # Calculate hidden layer output
        hidden_output = self.activation_function(np.dot(X_train_biased, self.input_weights) + self.bias)

        # Calculate output weights using Moore-Penrose pseudoinverse
        self.output_weights = np.linalg.pinv(hidden_output) @ y_train

    def predict(self, X_test):
        # Add bias to input
        X_test_biased = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

        # Calculate hidden layer output
        hidden_output = self.activation_function(np.dot(X_test_biased, self.input_weights) + self.bias)

        # Predict output
        y_pred = hidden_output @ self.output_weights
        return y_pred

    def activation_function(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))


diab = pd.read_csv("diabetes.csv")

# Standardize the data
X_mean = diab.mean()
X_std = diab.std()
Z = (diab - X_mean) / X_std

# Perform PCA
pca1 = PCA(n_components=5)
pca1.fit(Z)
x_pca = pca1.transform(Z)

# Extract labels
y = diab['Outcome']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.4, random_state=0)

# Initialize and train ELM model
elm = ELM(input_size=x_train.shape[1] + 1, hidden_size=250, output_size=1, pop_size=50)
elm.train(x_train, y_train)

# Predict on test set
y_pred = elm.predict(x_test)

# Convert predictions to binary classes
y_pred_classes = np.round(y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)

# Calculate true positives, true negatives, false positives, false negatives
tp = np.sum((y_test == 1) & (y_pred_classes == 1))
tn = np.sum((y_test == 0) & (y_pred_classes == 0))
fp = np.sum((y_test == 0) & (y_pred_classes == 1))
fn = np.sum((y_test == 1) & (y_pred_classes == 0))

# Calculate precision, recall, and F1 score
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

# Print metrics
print("Accuracy:", accuracy * 100)
print("Specificity:", precision*100)
print("Sensitivity:", recall*100)
print("F1 Score:", f1*100)
