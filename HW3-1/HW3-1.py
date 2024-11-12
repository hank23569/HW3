import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Step 1: Generate 300 random variables X in the range [0, 1000]
np.random.seed(42)
X = np.random.randint(0, 1001, 300)

# Step 2: Create binary classification labels Y
Y = np.where((X > 500) & (X < 800), 1, 0)

# Step 3: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2, random_state=42)

# Step 4: Implement Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y1 = logreg.predict(X_test)

# Step 5: Implement Support Vector Machine (SVM)
svm = SVC(probability=True)  # Using probability=True to get predicted probabilities
svm.fit(X_train, Y_train)
y2 = svm.predict(X_test)

# Step 6 and 7: Visualize the results with decision boundaries
plt.figure(figsize=(15, 6))

# Figure 1: X, Y, and X, Y1 with decision boundary for Logistic Regression
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='gray', label='True Labels')
plt.scatter(X_test, y1, color='blue', marker='x', label='Logistic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y and Logistic Regression Prediction')
plt.legend()

# Decision boundary for Logistic Regression
x_boundary = np.linspace(0, 1000, 300)
y_boundary = logreg.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
plt.plot(x_boundary, y_boundary, color='blue', linestyle='--', label='Logistic Regression Boundary')
plt.legend()

# Figure 2: X, Y, and X, Y2 with decision boundary for SVM
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='gray', label='True Labels')
plt.scatter(X_test, y2, color='green', marker='s', label='SVM')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X vs Y and SVM Prediction')
plt.legend()

# Decision boundary for SVM
x_boundary = np.linspace(0, 1000, 300)
y_boundary = svm.predict_proba(x_boundary.reshape(-1, 1))[:, 1]
plt.plot(x_boundary, y_boundary, color='green', linestyle='--', label='SVM Boundary')
plt.legend()

plt.tight_layout()
plt.show()
