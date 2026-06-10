import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 加载 MNIST 数据集（首次运行会自动下载）
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 像素值 0~255 归一化到 0~1，并增加通道维度 (28, 28) -> (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
print(x_train.shape)
print(x_test.shape)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_test.shape)

print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")

# 2. 构建模型
model = keras.Sequential(
    [
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu", name="conv1"),
        keras.layers.MaxPooling2D(name="pool1"),
        keras.layers.Conv2D(64, 3, activation="relu", name="conv2"),
        keras.layers.MaxPooling2D(name="pool2"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 3. 训练
print("\n开始训练...")
history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)

# 4. 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n测试集准确率: {test_acc:.4f}")

model_path = os.path.join(os.path.dirname(__file__), "mnist_model.keras")
model.save(model_path)
print(f"模型已保存: {model_path}")

# 5. 绘制训练曲线
# print(history.history["loss"])
epochs_range = range(1, len(history.history["loss"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history.history["loss"], "b-o", label="训练 loss")
axes[0].plot(epochs_range, history.history["val_loss"], "r-o", label="验证 loss")
axes[0].set_title("Loss 变化")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history["accuracy"], "b-o", label="训练 accuracy")
axes[1].plot(epochs_range, history.history["val_accuracy"], "r-o", label="验证 accuracy")
axes[1].set_title("Accuracy 变化")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
pictures_dir = os.path.join(os.path.dirname(__file__), "pictures")
os.makedirs(pictures_dir, exist_ok=True)
plot_path = os.path.join(pictures_dir, "training_history.png")
plt.savefig(plot_path, dpi=150)
print(f"\n训练曲线已保存: {plot_path}")
plt.show()

# 6. 看几个预测结果
predictions = model.predict(x_test[:5], verbose=0)
for i in range(5):
    pred = int(tf.argmax(predictions[i]))
    print(f"样本 {i}: 真实={y_test[i]}, 预测={pred}")
