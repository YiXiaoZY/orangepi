"""
MNIST 深入学习：交叉熵梯度、卷积核、特征图、错分样本
先运行 mnist.py 生成 mnist_model.keras，或本脚本会自动训练。
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
MODEL_PATH = os.path.join(BASE_DIR, "mnist_model.keras")
PICTURES_DIR = os.path.join(BASE_DIR, "pictures")
os.makedirs(PICTURES_DIR, exist_ok=True)


def build_model():
    return keras.Sequential(
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


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"加载已训练模型: {MODEL_PATH}")
        return keras.models.load_model(MODEL_PATH)

    print("未找到模型，快速训练 3 个 epoch...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., tf.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., tf.newaxis]

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1, verbose=1)
    model.save(MODEL_PATH)
    return model


def demo_cross_entropy_gradient():
    """演示 softmax + 交叉熵 时，输出层梯度 = 预测概率 - 真实标签"""
    print("\n" + "=" * 60)
    print("【1】交叉熵 + Softmax 的梯度（输出层）")
    print("=" * 60)

    logits = tf.Variable([2.0, 1.0, 0.1, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    true_label = 3

    with tf.GradientTape() as tape:
        probs = tf.nn.softmax(logits)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.constant([true_label], dtype=tf.int32),
            probs[tf.newaxis, :],
            from_logits=False,
        )

    grad = tape.gradient(loss, logits)
    one_hot = tf.one_hot(true_label, 10)

    print(f"真实标签: {true_label}")
    print(f"Softmax 概率: {np.round(probs.numpy(), 4)}")
    print(f"Loss: {float(np.asarray(loss).reshape(-1)[0]):.4f}")
    print(f"梯度 d(loss)/d(logits) = 预测 - one_hot:")
    print(f"  手算: {np.round((probs - one_hot).numpy(), 4)}")
    print(f"  Tape: {np.round(grad.numpy(), 4)}")
    print("\n解读:")
    print("  - 真实类别(3)的梯度为负 → 增大 logit[3] 可降低 loss")
    print("  - 错误类别梯度为正 → 减小它们的 logit 可降低 loss")
    print("  - 这就是反向传播在最后一层的具体形态")


def plot_conv_kernels(model, save_path):
    """第一层 32 个 3x3 卷积核"""
    conv1 = model.get_layer("conv1")
    kernels = conv1.get_weights()[0]  # shape: (3, 3, 1, 32)

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(kernels[:, :, 0, i], cmap="viridis")
        ax.set_title(f"#{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("第一层卷积核 (3×3) — 训练后学到的局部检测器", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"卷积核图已保存: {save_path}")
    plt.close()


def get_intermediate_outputs(model, image, layer_names):
    """逐层前向，提取指定层的输出（兼容 Keras 3 Sequential）"""
    outputs = {}
    x = tf.constant(image[tf.newaxis, ...])
    for layer in model.layers:
        x = layer(x)
        if layer.name in layer_names:
            outputs[layer.name] = x[0].numpy()
    return outputs


def plot_feature_maps(model, image, save_path):
    """对一张图，展示 conv1 / conv2 各通道的激活图"""
    feats = get_intermediate_outputs(model, image, ["conv1", "pool1", "conv2"])
    f1 = feats["conv1"]    # (26, 26, 32)
    f1p = feats["pool1"]   # (13, 13, 32)
    f2 = feats["conv2"]    # (11, 11, 64)

    fig = plt.figure(figsize=(14, 8))
    ax0 = fig.add_subplot(2, 3, 1)
    ax0.imshow(image[:, :, 0], cmap="gray")
    ax0.set_title("原图")
    ax0.axis("off")

    def show_grid(maps, ax, title, n_show=16):
        n = min(n_show, maps.shape[-1])
        cols = 4
        rows = (n + cols - 1) // cols
        mosaic = np.zeros((rows * maps.shape[0], cols * maps.shape[1]))
        for i in range(n):
            r, c = divmod(i, cols)
            mosaic[
                r * maps.shape[0] : (r + 1) * maps.shape[0],
                c * maps.shape[1] : (c + 1) * maps.shape[1],
            ] = maps[:, :, i]
        ax.imshow(mosaic, cmap="hot")
        ax.set_title(title)
        ax.axis("off")

    show_grid(f1, fig.add_subplot(2, 3, 2), "Conv1 特征图 (前16通道)")
    show_grid(f1p, fig.add_subplot(2, 3, 3), "Pool1 后 (前16通道)")
    show_grid(f2, fig.add_subplot(2, 3, 5), "Conv2 特征图 (前16通道)")

    ax_txt = fig.add_subplot(2, 3, 4)
    ax_txt.axis("off")
    ax_txt.text(
        0.05,
        0.6,
        "亮色区域 = 该卷积核\n对图像有强响应\n\n越深层的特征图\n越抽象（笔画组合）",
        fontsize=11,
        va="top",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"特征图已保存: {save_path}")
    plt.close()


def plot_misclassified(model, x_test, y_test, save_path, max_show=12):
    """展示预测错误的样本"""
    preds = model.predict(x_test, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    wrong_idx = np.where(pred_labels != y_test)[0]

    print(f"\n测试集错分数量: {len(wrong_idx)} / {len(y_test)}")

    if len(wrong_idx) == 0:
        return

    show_n = min(max_show, len(wrong_idx))
    cols = 4
    rows = (show_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    for i, ax in enumerate(np.ravel(axes)):
        if i >= show_n:
            ax.axis("off")
            continue
        idx = wrong_idx[i]
        ax.imshow(x_test[idx, :, :, 0], cmap="gray")
        conf = preds[idx][pred_labels[idx]]
        ax.set_title(
            f"真:{y_test[idx]} 预:{pred_labels[idx]}\n置信:{conf:.2f}",
            fontsize=9,
            color="red",
        )
        ax.axis("off")
    fig.suptitle("错分样本 — 观察哪些数字容易混淆", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"错分样本图已保存: {save_path}")
    plt.close()


def main():
    demo_cross_entropy_gradient()

    model = load_or_train_model()

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., tf.newaxis]

    plot_conv_kernels(model, os.path.join(PICTURES_DIR, "conv_kernels.png"))
    plot_feature_maps(model, x_test[0], os.path.join(PICTURES_DIR, "feature_maps.png"))
    plot_misclassified(model, x_test, y_test, os.path.join(PICTURES_DIR, "misclassified.png"))

    print("\n全部完成。请打开以下图片:")
    print("  - conv_kernels.png   卷积核")
    print("  - feature_maps.png   特征图")
    print("  - misclassified.png  错分样本")


if __name__ == "__main__":
    main()
