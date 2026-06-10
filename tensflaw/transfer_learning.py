"""
第 6 课：迁移学习（Transfer Learning）
用 ImageNet 预训练的 MobileNetV2，在 CIFAR-10 上微调。
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(__file__)
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
os.makedirs(PICTURES_DIR, exist_ok=True)

IMG_SIZE = 96          # 比 224 小，CPU 更快
BATCH_SIZE = 64
EPOCHS_HEAD = 5        # 阶段 1：只训练分类头
EPOCHS_FINE = 3        # 阶段 2：微调预训练层
MAX_TRAIN = 10000      # 子集加速（全量 50000 太慢）
CLASS_NAMES = [
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "蛙", "马", "船", "卡车",
]


def load_cifar10_datasets():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # 取子集 + 划分验证集
    x_train, y_train = x_train[:MAX_TRAIN], y_train[:MAX_TRAIN]
    val_size = int(len(x_train) * 0.1)
    
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    def make_ds(x, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(2000)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return make_ds(x_train, y_train, True), make_ds(x_val, y_val), make_ds(x_test, y_test)


def preprocess(image, label):
    """resize + MobileNetV2 标准预处理"""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def build_model(trainable_base=False):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = trainable_base

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


def plot_history(hist_head, hist_fine, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for hist, label, color in [
        (hist_head, "阶段1-分类头", "b"),
        (hist_fine, "阶段2-微调", "r"),
    ]:
        offset = len(axes[0].lines) // 2
        epochs = range(offset + 1, offset + len(hist.history["accuracy"]) + 1)
        axes[0].plot(epochs, hist.history["loss"], f"{color}-o", label=f"{label} loss")
        axes[0].plot(
            epochs, hist.history["val_loss"], f"{color}--o", label=f"{label} val_loss"
        )
        axes[1].plot(epochs, hist.history["accuracy"], f"{color}-o", label=f"{label} acc")
        axes[1].plot(
            epochs, hist.history["val_accuracy"], f"{color}--o", label=f"{label} val_acc"
        )

    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"训练曲线已保存: {save_path}")
    plt.show()


def show_predictions(model, test_ds, save_path, n=9):
    images, labels = [], []
    for batch_x, batch_y in test_ds.take(1):
        images, labels = batch_x.numpy(), batch_y.numpy()
        break

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, i in zip(axes.flat, range(n)):
        # 反预处理以便显示
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        ax.imshow(img)
        true_name = CLASS_NAMES[labels[i]]
        pred_name = CLASS_NAMES[pred_labels[i]]
        color = "green" if labels[i] == pred_labels[i] else "red"
        ax.set_title(f"真:{true_name}\n预:{pred_name}", fontsize=9, color=color)
        ax.axis("off")
    fig.suptitle("CIFAR-10 迁移学习预测", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"预测样例已保存: {save_path}")
    plt.show()


def main():
    train_ds, val_ds, test_ds = load_cifar10_datasets()
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # ── 阶段 1：冻结预训练 backbone，只训练新加的分类头 ──
    print("=" * 50)
    print("阶段 1：冻结 MobileNetV2，训练分类头")
    print("=" * 50)
    model, base = build_model(trainable_base=False)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    hist_head = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=1
    )

    # ── 阶段 2：解冻 backbone 末尾若干层，低学习率微调 ──
    print("\n" + "=" * 50)
    print("阶段 2：解冻末尾 30 层，低学习率微调")
    print("=" * 50)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
    print(f"可训练参数: {trainable:,}")

    hist_fine = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, verbose=1
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n测试集准确率: {test_acc:.4f}")

    model_path = os.path.join(BASE_DIR, "cifar10_mobilenet.keras")
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    plot_history(hist_head, hist_fine, os.path.join(PICTURES_DIR, "transfer_learning.png"))
    show_predictions(model, test_ds, os.path.join(PICTURES_DIR, "transfer_predictions.png"))


if __name__ == "__main__":
    main()
