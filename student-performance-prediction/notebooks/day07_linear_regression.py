import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Lấy thư mục gốc của project
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn đến file dữ liệu
DATA_PATH = BASE_DIR / "data" / "student_performance_cleaned.csv"

# Kiểm tra file tồn tại
print("File path:", DATA_PATH)
print("File exists:", DATA_PATH.exists())

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH)

print(df.head())


#Xác định X,y
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

df_encoded = pd.get_dummies(
    df,
    columns=["Gender"],
    dtype=int
)

X = df_encoded.drop(
    columns=["Test_Score", "Student_ID", "Grade"],
    errors="ignore"
)

y = df_encoded["Test_Score"]

#Chia train test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Xử lý giá trị khuyết
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Kiểm tra kết quả
import numpy as np

print("Mean:", X_train_scaled.mean(axis=0))
print("Std:", X_train_scaled.std(axis=0))

print("NaN trong train:", np.isnan(X_train_scaled).sum())
print("NaN trong test:", np.isnan(X_test_scaled).sum())



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Khởi tạo mô hình
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train_scaled, y_train)

# Dự đoán điểm thi
y_pred = model.predict(X_test_scaled)

# So sánh kết quả
print("Điểm thực tế:")
print(y_test.head(10).values)

print("Điểm dự đoán:")
print(y_pred[:10])

# Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

print("\nFeature names:")
print(X.columns.tolist())