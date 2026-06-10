"""
第 5 课：自定义训练循环（GradientTape）
不用 model.fit，手写 train_step，理解 model.fit 内部在做什么。
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

EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def load_datasets():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., tf.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., tf.newaxis]

    # 从训练集末尾切 10% 做验证集（与 validation_split=0.1 一致）
    val_size = int(len(x_train) * 0.1)
    print("val_size:  3",val_size)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
    return train_ds, val_ds, test_ds


def build_model():
    return keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def main():
    train_ds, val_ds, test_ds = load_datasets()
    model = build_model()

    # 优化器、损失、指标 — 等价于 model.compile(...) 里的配置
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    train_loss = keras.metrics.Mean(name="train_loss")
    train_acc = keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_loss = keras.metrics.Mean(name="val_loss")
    val_acc = keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    # ── 核心：一个 batch 的训练步骤 ──
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)          # 前向传播
            loss = loss_fn(y, preds)                 # 计算 loss

        grads = tape.gradient(loss, model.trainable_variables)  # 反向传播
        optimizer.apply_gradients(                  # 更新权重
            zip(grads, model.trainable_variables)
        )

        train_loss.update_state(loss)
        train_acc.update_state(y, preds)
        return loss

    # 验证步骤：只前向，不更新权重
    @tf.function
    def val_step(x, y):
        preds = model(x, training=False)
        loss = loss_fn(y, preds)
        val_loss.update_state(loss)
        val_acc.update_state(y, preds)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print("开始自定义训练循环...\n")
    for epoch in range(EPOCHS):
        # 每个 epoch 开始前重置指标
        train_loss.reset_state()
        train_acc.reset_state()
        val_loss.reset_state()
        val_acc.reset_state()

        # 训练阶段
        for batch_x, batch_y in train_ds:
            train_step(batch_x, batch_y)

        # 验证阶段
        for batch_x, batch_y in val_ds:
            val_step(batch_x, batch_y)

        tl, ta = float(train_loss.result()), float(train_acc.result())
        vl, va = float(val_loss.result()), float(val_acc.result())
        history["loss"].append(tl)
        history["accuracy"].append(ta)
        history["val_loss"].append(vl)
        history["val_accuracy"].append(va)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}  "
            f"loss={tl:.4f}  acc={ta:.4f}  "
            f"val_loss={vl:.4f}  val_acc={va:.4f}"
        )

    # 测试集评估（同样用手写循环）
    test_loss = keras.metrics.Mean(name="test_loss")
    test_acc = keras.metrics.SparseCategoricalAccuracy(name="test_acc")
    for batch_x, batch_y in test_ds:
        preds = model(batch_x, training=False)
        test_loss.update_state(loss_fn(batch_y, preds))
        test_acc.update_state(batch_y, preds)

    print(f"\n测试集  loss={float(test_loss.result()):.4f}  acc={float(test_acc.result()):.4f}")

    # 绘制曲线
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs_range, history["loss"], "b-o", label="训练")
    axes[0].plot(epochs_range, history["val_loss"], "r-o", label="验证")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["accuracy"], "b-o", label="训练")
    axes[1].plot(epochs_range, history["val_accuracy"], "r-o", label="验证")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PICTURES_DIR, "custom_train_history.png")
    plt.savefig(path, dpi=150)
    print(f"训练曲线已保存: {path}")
    plt.show()

    print("\n对照理解 model.fit 内部:")
    print("  model.fit(...)  ≈  for epoch: for batch: train_step(...)")
    print("  model.compile   ≈  定义 optimizer + loss_fn + metrics")
    print("  @tf.function    ≈  把 Python 循环编译成图，加速训练")


if __name__ == "__main__":
    main()
