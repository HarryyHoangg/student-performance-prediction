<<<<<<< HEAD
import pandas as pd
import numpy as np

data = {
    "Study_Hours": [2, 3, 5, 6, 8],
    "Attendance": [60, 70, 75, 85, 95],
    "Previous_Score": [50, 55, 65, 75, 90],
    "Final_Score": [45, 50, 68, 78, 95]
}

df = pd.DataFrame(data)

# 1. Kich thuoc du lieu
print("Shape:", df.shape)

# 2. Ten cac cot
print("Columns:", df.columns)

# 3. Kieu du lieu
print(df.dtypes)

# 4. Thong tin tong quan
df.info()

# 5. Thong ke mo ta
print(df.describe())

# 6. Diem trung binh
print("Average:", np.mean(df["Final_Score"]))

# 7. Diem cao nhat
=======
import pandas as pd
import numpy as np

data = {
    "Study_Hours": [2, 3, 5, 6, 8],
    "Attendance": [60, 70, 75, 85, 95],
    "Previous_Score": [50, 55, 65, 75, 90],
    "Final_Score": [45, 50, 68, 78, 95]
}

df = pd.DataFrame(data)

# 1. Kich thuoc du lieu
print("Shape:", df.shape)

# 2. Ten cac cot
print("Columns:", df.columns)

# 3. Kieu du lieu
print(df.dtypes)

# 4. Thong tin tong quan
df.info()

# 5. Thong ke mo ta
print(df.describe())

# 6. Diem trung binh
print("Average:", np.mean(df["Final_Score"]))

# 7. Diem cao nhat
>>>>>>> 0aef3f0a3a3312cbf40bea6f7f9b1c61d5c99c77
print("Max:", df["Final_Score"].max())