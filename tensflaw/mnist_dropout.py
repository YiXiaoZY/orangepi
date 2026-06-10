"""
第 3 课：Dropout 正则化 + EarlyStopping
对比「无 Dropout」与「有 Dropout」在长训练下的过拟合差异。
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(__file__)
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
os.makedirs(PICTURES_DIR, exist_ok=True)


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., tf.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., tf.newaxis]
    return x_train, y_train, x_test, y_test


def build_model(use_dropout=False):
    layers = [
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
    ]
    if use_dropout:
        layers.append(keras.layers.Dropout(0.5))
    layers.append(keras.layers.Dense(10, activation="softmax"))
    return keras.Sequential(layers)


def train_and_evaluate(name, use_dropout, x_train, y_train, x_test, y_test, epochs=12):
    print(f"\n{'=' * 50}\n训练模型: {name}\n{'=' * 50}")
    model = build_model(use_dropout=use_dropout)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} 测试集准确率: {test_acc:.4f}")
    return history


def plot_comparison(hist_baseline, hist_dropout):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_b = range(1, len(hist_baseline.history["loss"]) + 1)
    epochs_d = range(1, len(hist_dropout.history["loss"]) + 1)

    axes[0].plot(epochs_b, hist_baseline.history["loss"], "b-o", label="无 Dropout 训练")
    axes[0].plot(epochs_b, hist_baseline.history["val_loss"], "b--o", label="无 Dropout 验证")
    axes[0].plot(epochs_d, hist_dropout.history["loss"], "r-s", label="有 Dropout 训练")
    axes[0].plot(epochs_d, hist_dropout.history["val_loss"], "r--s", label="有 Dropout 验证")
    axes[0].set_title("Loss：过拟合时验证 loss 会回升")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_b, hist_baseline.history["accuracy"], "b-o", label="无 Dropout 训练")
    axes[1].plot(epochs_b, hist_baseline.history["val_accuracy"], "b--o", label="无 Dropout 验证")
    axes[1].plot(epochs_d, hist_dropout.history["accuracy"], "r-s", label="有 Dropout 训练")
    axes[1].plot(epochs_d, hist_dropout.history["val_accuracy"], "r--s", label="有 Dropout 验证")
    axes[1].set_title("Accuracy：训练与验证的差距")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PICTURES_DIR, "dropout_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"\n对比图已保存: {path}")
    plt.show()


def main():
    x_train, y_train, x_test, y_test = load_data()

    hist_base = train_and_evaluate(
        "基线（无 Dropout）", False, x_train, y_train, x_test, y_test
    )
    hist_drop = train_and_evaluate(
        "Dropout 版", True, x_train, y_train, x_test, y_test
    )
    plot_comparison(hist_base, hist_drop)

    print("\n观察要点:")
    print("  1. 无 Dropout：训练 acc 持续升，验证 acc 可能停滞或下降 → 过拟合")
    print("  2. Dropout：训练 acc 略低，但验证 acc 更稳、测试集往往更好")
    print("  3. EarlyStopping：验证 loss 不再下降时自动停训，避免白跑")


if __name__ == "__main__":
    main()
