import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def rbf_kernel(x, gamma=0.1):
    return np.exp(-gamma * np.abs(x - np.mean(x))**2)

def exp_decay(x, tau=1):
    return np.exp(-x/tau)

df = pd.read_csv("data/post_processed/row_game_lineups_mmm.csv", index_col=0)
y = df['y']
del df['y']

X = df.values
Y = np.array(y)
gamma = np.random.rand(X.shape[1])  # random gamma for each feature
tau = np.random.rand(X.shape[1])  # random gamma for each feature

# transform each feature using its own RBF kernel
X_transformed = np.empty_like(X)
for i in range(X.shape[1]):
    # X_transformed[:, i] = rbf_kernel(X[:, i], gamma=gamma[i])
    X_transformed[:, i] = exp_decay(X[:, i], tau=tau[i])
    # X_transformed[:, i] = X[:, i]

# Add a column of ones to include an intercept in the model
X_transformed = sm.add_constant(X_transformed)

# fit ordinary least squares regression on transformed features
model = sm.OLS(Y, X_transformed)
results = model.fit()

# print out the summary
print(results.summary())

# Predicted values
Y_pred = results.predict(X_transformed)

# Create scatter plot of Actual vs. Predicted values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(Y, Y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")

# Residual plot
residuals = Y - Y_pred
plt.subplot(1, 2, 2)
plt.scatter(Y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()

# Plot coefficients
coef = results.params
plt.figure(figsize=(8, 6))
plt.bar(range(len(coef)), coef)
plt.title("Regression Coefficients")
plt.show()

#print RMSE
rmse = np.sqrt(np.mean((Y - results.predict(X_transformed)) ** 2))
print(rmse)

# Plot residuals distribution
plt.hist(residuals)
plt.title("Residuals Histogram")
plt.show()