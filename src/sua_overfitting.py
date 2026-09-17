import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


# 1. Đọc dữ liệu
data = pd.read_csv("data/train.csv")


# 2. Chọn dữ liệu
data = data[["OverallQual", "GrLivArea", "GarageCars", "SalePrice"]]

# Xóa dữ liệu bị thiếu
data = data.dropna()


# 3. X và y
X = data[["OverallQual", "GrLivArea", "GarageCars"]]

y = data["SalePrice"]


# 4. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Tạo Pipeline
model = Pipeline([
    ("poly", PolynomialFeatures(degree=10)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=10))
])


# 6. Huấn luyện
model.fit(X_train, y_train)


# 7. Dự đoán
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# 8. Tính R2
train_r2 = r2_score(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)


# 9. In kết quả
print("KẾT QUẢ SAU KHI GIẢM OVERFITTING")
print("----------------------------------")

print("R2 trên tập Train:", train_r2)

print("R2 trên tập Test :", test_r2)

print("----------------------------------")

print("Chênh lệch R2:", train_r2 - test_r2)