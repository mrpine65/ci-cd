import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error


# =========================
# 1. Load & preprocess data
# =========================
df = pd.read_csv("data.csv")

# Chuẩn hóa tên cột và giá trị dạng chuỗi
df.columns = df.columns.str.lower().str.replace(" ", "_")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

# Xử lý giá trị khuyết & chia tập
df_full = df.copy().fillna(0)
df_train, df_val = train_test_split(df_full, train_size=0.8, random_state=100)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)

# =========================
# 2. Tách biến đầu ra & biến đầu vào
# =========================
y_train = np.log1p(df_train["msrp"].values)
y_val = np.log1p(df_val["msrp"].values)

train_dicts = df_train.drop("msrp", axis=1).to_dict(orient="records")
val_dicts = df_val.drop("msrp", axis=1).to_dict(orient="records")

# Vector hóa dữ liệu
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# =========================
# 3. Huấn luyện mô hình
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 4. Dự đoán & đánh giá
# =========================
y_pred = model.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)
rmse = np.expm1(rmse)
print("RMSE:", rmse)
with open("metrics.txt", "w") as outfile:
    outfile.write(f"RMSE: {rmse}")

# Lưu model & DictVectorizer
# with open("model.b", "wb") as f_out:
#     pickle.dump((dv, model), f_out)


# =========================
# 5. Hàm trực quan hóa
# =========================
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Vẽ biểu đồ so sánh giá trị thực tế và dự đoán.
    """
    y_train_true = np.expm1(train_labels)
    y_test_true = np.expm1(test_labels)
    y_pred = np.expm1(predictions)

    plt.figure(figsize=(7, 6))

    # Dữ liệu huấn luyện
    plt.scatter(y_train_true, y_train_true, c="b", alpha=0.4, label="Training data")

    # Dữ liệu kiểm thử (thực tế)
    plt.scatter(y_test_true, y_test_true, c="g", alpha=0.4, label="Testing data (actual)")

    # Dự đoán
    plt.scatter(y_test_true, y_pred, c="r", alpha=0.6, label="Predictions")

    # Đường tham chiếu y = x
    plt.plot(
        [y_test_true.min(), y_test_true.max()],
        [y_test_true.min(), y_test_true.max()],
        'k--', lw=1
    )

    plt.legend(shadow=True)
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('Actual MSRP', family='Arial', fontsize=11)
    plt.ylabel('Predicted MSRP', family='Arial', fontsize=11)
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=120)
    plt.show()


# =========================
# 6. Gọi hàm vẽ
# =========================
# plot_predictions(X_train, y_train, X_val, y_val, y_pred)

