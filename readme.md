# House Price Prediction - Overfitting

## 1. Giới thiệu

Project sử dụng dữ liệu giá nhà để xây dựng mô hình
Machine Learning dự đoán giá nhà.

Project tập trung vào hiện tượng Overfitting và
cách sử dụng Regularization để giảm Overfitting.

## 2. Dataset

Dataset House Prices được lấy từ Kaggle.

File dữ liệu:

- train.csv

Một số thuộc tính được sử dụng:

- OverallQual: chất lượng tổng thể
- GrLivArea: diện tích sinh hoạt
- GarageCars: sức chứa gara
- SalePrice: giá bán nhà

## 3. Công nghệ

- Python
- Pandas
- Scikit-learn
- Polynomial Regression
- Ridge Regression

## 4. Cấu trúc project

HousePricePrediction/

├── data/

│   └── train.csv

├── src/

│   ├── overfitting.py

│   └── solve_overfitting.py

├── README.md

└── requirements.txt

## 5. Cài đặt

Cài đặt các thư viện:

```bash
pip install -r requirements.txt