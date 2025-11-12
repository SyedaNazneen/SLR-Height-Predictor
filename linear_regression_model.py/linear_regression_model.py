import csv
import math

# --- 1. Load and Prepare Data ---

def load_data(filepath="Pearson.csv"):
    """
    The Kaggle dataset columns are: 'Father', 'Mother', 'MidParent', 'Child'.
    For SLR, let's use 'MidParent' (X) to predict 'Child' (Y) height.
    """
    # Mock Data for demonstration purposes (replace with actual data loading)
    # X (MidParent Height), Y (Child Height)
    data = [
        [69.0, 70.4], [72.5, 71.2], [65.5, 66.8], [67.5, 68.8], [71.0, 71.0], 
        [63.0, 65.5], [68.0, 69.2], [70.0, 70.8], [66.0, 67.5], [64.0, 66.2]
    ]
    
    X = [row[0] for row in data]
    Y = [row[1] for row in data]
    return X, Y

def train_test_split(X, Y, split_ratio=0.8):
    """Manually splits the data into training and testing sets."""
    split_index = int(len(X) * split_ratio)
    
    X_train = X[:split_index]
    Y_train = Y[:split_index]
    
    X_test = X[split_index:]
    Y_test = Y[split_index:]
    
    return X_train, Y_train, X_test, Y_test

# --- 2. Core Mathematical Functions ---

def mean(values):
    """Calculates the mean (average) of a list of numbers."""
    return sum(values) / len(values)

def variance(values, mu):
    """Calculates the variance of a list of numbers."""
    return sum([(x - mu)**2 for x in values])

def covariance(X, Y, mu_x, mu_y):
    """Calculates the covariance between two lists of numbers."""
    return sum([(X[i] - mu_x) * (Y[i] - mu_y) for i in range(len(X))])

def calculate_coefficients(X, Y):
    """
    Calculates the slope (beta1) and intercept (beta0) using the Normal Equation.
    
    beta1 = Cov(X, Y) / Var(X)
    beta0 = mean(Y) - beta1 * mean(X)
    """
    if len(X) == 0:
        return 0, 0
        
    mu_x = mean(X)
    mu_y = mean(Y)
    
    cov_xy = covariance(X, Y, mu_x, mu_y)
    var_x = variance(X, mu_x)
    
    # Check for zero variance (unlikely in real data, but good practice)
    if var_x == 0:
        beta1 = 0
    else:
        beta1 = cov_xy / var_x
        
    beta0 = mu_y - beta1 * mu_x
    
    return beta0, beta1

# --- 3. Prediction and Metric Calculation ---

def predict(x, beta0, beta1):
    """Makes a prediction using the SLR equation: y_hat = beta0 + beta1 * x."""
    return beta0 + beta1 * x

def calculate_mse_loss(Y_actual, Y_predicted):
    """
    Calculates the Mean Squared Error (MSE), which serves as the Loss metric.
    Loss = (1/N) * sum((Y_predicted - Y_actual)^2)
    """
    if len(Y_actual) == 0:
        return 0
    
    errors = [(Y_predicted[i] - Y_actual[i])**2 for i in range(len(Y_actual))]
    return sum(errors) / len(Y_actual)

def calculate_r_squared_accuracy(Y_actual, Y_predicted):
    """
    Calculates the R-squared (R^2) value, which serves as the Accuracy metric
    for regression (it explains the proportion of variance in Y predictable from X).
    R^2 = 1 - (SSR / SST)
    """
    if len(Y_actual) == 0:
        return 0
        
    mu_y = mean(Y_actual)
    
    # Sum of Squared Residuals (SSR) - Unexplained variance
    SSR = sum([(Y_predicted[i] - Y_actual[i])**2 for i in range(len(Y_actual))])
    
    # Total Sum of Squares (SST) - Total variance
    SST = sum([(y - mu_y)**2 for y in Y_actual])
    
    if SST == 0:
        return 0 # Perfect fit or no variance in Y
    
    return 1 - (SSR / SST)

# --- 4. Main Execution ---

if __name__ == "__main__":
    X, Y = load_data("pearsons_height_data.csv") # Use the actual file path
    
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, 0.8)
    
    print(f"--- Data Split ---")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("-" * 30)

    # Calculate Coefficients using Training Data
    beta0, beta1 = calculate_coefficients(X_train, Y_train)

    print(f"--- Model Coefficients ---")
    print(f"Intercept (beta0): {beta0:.4f}")
    print(f"Slope (beta1): {beta1:.4f}")
    print(f"Regression Equation: Y = {beta0:.4f} + {beta1:.4f} * X")
    print("-" * 30)
    
    # --- Evaluate on Training Set ---
    Y_train_pred = [predict(x, beta0, beta1) for x in X_train]
    train_loss = calculate_mse_loss(Y_train, Y_train_pred)
    train_r2 = calculate_r_squared_accuracy(Y_train, Y_train_pred)

    print(f"--- Training Metrics (Loss & Accuracy) ---")
    print(f"Train Loss (MSE): {train_loss:.4f}")
    print(f"Train Accuracy (R-squared): {train_r2:.4f}")
    print("-" * 30)

    # --- Evaluate on Testing Set ---
    Y_test_pred = [predict(x, beta0, beta1) for x in X_test]
    test_loss = calculate_mse_loss(Y_test, Y_test_pred)
    test_r2 = calculate_r_squared_accuracy(Y_test, Y_test_pred)

    print(f"--- Testing Metrics (Loss & Accuracy) ---")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test Accuracy (R-squared): {test_r2:.4f}")
    print("-" * 30)

    # Example Prediction
    example_x = 68.0 
    example_y_pred = predict(example_x, beta0, beta1)
    print(f"Prediction for X={example_x} inches: Y={example_y_pred:.2f} inches")

    # The final coefficients you will use in the frontend (after running on full data):
    # beta0 = -13.0800  # Placeholder values based on typical results for this data
    # beta1 = 1.1800