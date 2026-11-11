import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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

#Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:5])

print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))