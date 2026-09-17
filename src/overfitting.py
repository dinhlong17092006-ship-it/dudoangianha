import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 1. Đọc dữ liệu
data = pd.read_csv("data/train.csv")


# 2. Chọn các cột cần sử dụng
data = data[["OverallQual", "GrLivArea", "GarageCars", "SalePrice"]]

# Xóa các dòng bị thiếu dữ liệu
data = data.dropna()


# 3. X là dữ liệu đầu vào
X = data[["OverallQual", "GrLivArea", "GarageCars"]]

# 4. y là giá nhà
y = data["SalePrice"]


# 5. Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Tạo đa thức bậc cao
poly = PolynomialFeatures(degree=10)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 7. Tạo mô hình Linear Regression
model = LinearRegression()


# 8. Huấn luyện
model.fit(X_train_poly, y_train)


# 9. Dự đoán trên tập train
y_train_pred = model.predict(X_train_poly)

# 10. Dự đoán trên tập test
y_test_pred = model.predict(X_test_poly)


# 11. Tính R2
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)


# 12. In kết quả
print("KẾT QUẢ MÔ HÌNH OVERFITTING")
print("-----------------------------")

print("R2 trên tập Train:", train_r2)
print("R2 trên tập Test :", test_r2)

print("-----------------------------")

if train_r2 - test_r2 > 0.1:
    print("Mô hình có dấu hiệu OVERFITTING")
else:
    print("Chưa thấy dấu hiệu overfitting rõ ràng")