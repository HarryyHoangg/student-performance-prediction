
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load dataset
df = pd.read_csv("../data/student_performance_cleaned.csv")

# 2. Select features and target
X = df[["Study_Hours"]]
y = df["Test_Score"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 4. Build final model
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# 5. Train
final_model.fit(X_train, y_train)

# 6. Predict
y_pred = final_model.predict(X_test)

print("Final model trained successfully!")
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n===== FINAL MODEL EVALUATION =====")

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")



plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Test Score")
plt.ylabel("Predicted Test Score")
plt.title("Actual vs Predicted")

plt.grid(True)
plt.tight_layout()
plt.show()



results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Residual"] = (
    results["Actual"] - results["Predicted"]
)

results["Absolute_Error"] = (
    results["Residual"].abs()
)

print(results.head(10))



worst_predictions = results.sort_values(
    by="Absolute_Error",
    ascending=False
)

print("\n===== TOP 10 WORST PREDICTIONS =====")
print(worst_predictions.head(10))



# Create reports directory
report_dir = Path.cwd() / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

# Save prediction results
results.to_csv(
    report_dir / "day19_predictions.csv",
    index=False
)

# Save evaluation metrics
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Value": [mae, mse, rmse, r2]
})

metrics_df.to_csv(
    report_dir / "day19_evaluation_metrics.csv",
    index=False
)

# Save Actual vs Predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Test Score")
plt.ylabel("Predicted Test Score")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    report_dir / "day19_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Reports saved at:", report_dir.resolve())
