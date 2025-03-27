import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv("c:/Users/2273581/Downloads/CarPrice_Assignment.csv")
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# Data Cleaning:
df.dropna(inplace=True)

# Define target and features
target = "price"  # Ensure this matches the actual column name in your DataFrame
X = df.drop(target, axis=1)
y = df[target]
X = pd.get_dummies(X, drop_first=True)

# Split into training and testing datasets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the five models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR()  # Default SVR uses an RBF kernel
}

predictions = {}  
results = {}      

# Train, predict, and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name} ...")
    if name in ["Linear Regression", "SVR"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    predictions[name] = preds
    
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    results[name] = {"R_squared": r2, "MSE": mse, "MAE": mae}

# Present evaluation metrics in a table
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Metrics:")
print(results_df)

best_model_name = results_df["R_squared"].idxmax()
print(f"\nBest performing model based on R-squared: {best_model_name}")

# Feature Importance Analysis
if best_model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
    best_model = models[best_model_name]
    importances = best_model.feature_importances_
    features = X_train.columns
    feat_importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    })
    feat_importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    
    # Select top 10 features
    top_features = feat_importance_df.head(10)
    
    print("\nFeature Importance from", best_model_name)
    print(top_features)
    
    # Visualize feature importance
    plt.figure(figsize=(12,8))
    sns.barplot(x="Importance", y="Feature", data=top_features)
    plt.title(f"Top 10 Feature Importance from {best_model_name}")
    plt.tight_layout()
    plt.show()
    
else:
    # If the best model was not tree-based, we can inspect the regression coefficients
    lin_model = models["Linear Regression"]
    coeff_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": lin_model.coef_
    })
    # Sort by the absolute value of coefficients
    coeff_df["Absolute"] = coeff_df["Coefficient"].abs()
    coeff_df.sort_values(by="Absolute", ascending=False, inplace=True)
    
    # Select top 10 features
    top_coeffs = coeff_df.head(10)
    
    print("\nFeature Coefficients from Linear Regression:")
    print(top_coeffs)
    
    plt.figure(figsize=(12,8))
    sns.barplot(x="Coefficient", y="Feature", data=top_coeffs)
    plt.title("Top 10 Feature Coefficients from Linear Regression")
    plt.tight_layout()
    plt.show()

if best_model_name == "Random Forest":
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }
    rf_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_reg,
                               param_grid=param_grid,
                               cv=5,
                               scoring="r2",
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters from Grid Search for Random Forest:")
    print(grid_search.best_params_)
    
    best_rf = grid_search.best_estimator_
    best_rf_pred = best_rf.predict(X_test)
    
    # Evaluate the tuned model
    r2_tuned = r2_score(y_test, best_rf_pred)
    mse_tuned = mean_squared_error(y_test, best_rf_pred)
    mae_tuned = mean_absolute_error(y_test, best_rf_pred)
    
    print("\nPerformance of Tuned Random Forest:")
    print(f"R_squared: {r2_tuned:.4f}, MSE: {mse_tuned:.4f}, MAE: {mae_tuned:.4f}")
else:
    print(f"\nHyperparameter tuning was demonstrated on Random Forest. You may apply similar tuning to {best_model_name} if it suits your requirements.")