
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Đọc dữ liệu
df = pd.read_csv("../data/student_performance_cleaned.csv")
print(df.head())

# 2. Feature Engineering
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

# 3. Xác định X và y
X = df.drop(
    columns=["Test_Score", "Student_ID", "Grade"],
    errors="ignore"
)

y = df["Test_Score"]

# 4. Chia Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# 5. Xác định các nhóm cột
numeric_features = [
    "Age",
    "Study_Hours",
    "Attendance(%)",
    "Study_Attendance_Index"
]

categorical_features = ["Gender"]

# 6. Pipeline xử lý dữ liệu số
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Pipeline xử lý dữ liệu phân loại
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
))
])

# 8. Kết hợp bằng ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 9. Tạo Pipeline hoàn chỉnh
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 10. Huấn luyện mô hình
model.fit(X_train, y_train)

# 11. Dự đoán
y_pred = model.predict(X_test)

# 12. Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# 13. So sánh kết quả
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\n--- PREDICTION RESULTS ---")
print(results.head(10))


# Lấy tên các features sau preprocessing
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Lấy coefficients của Linear Regression
coefficients = model.named_steps[
    "regressor"
].coef_

# Hiển thị thành bảng
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

print("\n--- MODEL COEFFICIENTS ---")
print(coef_df)

print("\nIntercept:", model.named_steps["regressor"].intercept_)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(coef_df.to_string(index=False))
coef_df["Abs_Coefficient"] = (
    coef_df["Coefficient"].abs()
)

coef_df = coef_df.sort_values(
    by="Abs_Coefficient",
    ascending=False
)

print(coef_df.to_string(index=False))