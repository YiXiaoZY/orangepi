"""
第 4 课：模型推理 — 加载已训练模型，对单张/多张图片预测
用法:
  python predict_digit.py              # 预测测试集随机 6 张
  python predict_digit.py 图片路径.png  # 预测你自己的图
把图片放到 images/ 目录也可批量预测。
"""
import os
import sys

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
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(PICTURES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型 {MODEL_PATH}\n请先运行: python mnist.py"
        )
    return keras.models.load_model(MODEL_PATH)


def preprocess_for_mnist(img_array):
    """
    转为 MNIST 格式：28x28 灰度，白字黑底，值域 0~1。
    img_array: 2D numpy，任意尺寸。
    """
    img = tf.image.resize(img_array[..., np.newaxis], (28, 28))
    img = img.numpy().squeeze()

    # 归一化到 0~1
    if img.max() > 1.0:
        img = img / 255.0

    # 常见扫描图是黑字白底，MNIST 是白字黑底 → 自动反转
    if img.mean() > 0.5:
        img = 1.0 - img

    return img.astype("float32")


def load_image_file(path):
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("需要 Pillow: pip install pillow")

    raw = np.array(Image.open(path).convert("L"), dtype=np.float32)
    processed = preprocess_for_mnist(raw)
    return raw, processed


def predict_one(model, image_28x28):
    x = image_28x28[np.newaxis, ..., np.newaxis]
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs


def show_prediction(raw_or_proc, pred, probs, title="", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].imshow(raw_or_proc, cmap="gray")
    axes[0].set_title(title or "输入")
    axes[0].axis("off")

    axes[1].bar(range(10), probs, color="steelblue")
    axes[1].set_xticks(range(10))
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"预测: {pred}  (置信度 {probs[pred]:.2%})")
    axes[1].axvline(pred, color="red", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"结果图已保存: {save_path}")
    plt.show()


def predict_test_samples(model, n=6):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0

    indices = np.random.choice(len(x_test), n, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, idx in zip(axes.flat, indices):
        img = x_test[idx]
        pred, probs = predict_one(model, img)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"真:{y_test[idx]} 预:{pred} ({probs[pred]:.0%})")
        ax.axis("off")
    fig.suptitle("测试集随机样本预测", fontsize=12)
    plt.tight_layout()
    path = os.path.join(PICTURES_DIR, "predict_samples.png")
    plt.savefig(path, dpi=150)
    print(f"批量预测图已保存: {path}")
    plt.show()


def predict_path(model, path):
    raw, processed = load_image_file(path)
    pred, probs = predict_one(model, processed)
    print(f"文件: {path}")
    print(f"预测数字: {pred}")
    print("各类概率:", np.round(probs, 3))
    save_path = os.path.join(PICTURES_DIR, "predict_custom.png")
    show_prediction(processed, pred, probs, title=os.path.basename(path), save_path=save_path)


def predict_images_folder(model):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    files = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f.lower())[1] in exts
    ]
    if not files:
        print(f"images/ 目录为空，可放入手写数字图片: {IMAGES_DIR}")
        return
    for path in sorted(files):
        predict_path(model, path)


def main():
    model = load_model()
    print(f"已加载模型: {MODEL_PATH}\n")

    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            predict_path(model, path)
    elif any(
        os.path.splitext(f.lower())[1] in {".png", ".jpg", ".jpeg", ".bmp"}
        for f in os.listdir(IMAGES_DIR)
    ):
        predict_images_folder(model)
    else:
        predict_test_samples(model)
        print(f"\n提示: 把手写数字图放到 {IMAGES_DIR} 再运行本脚本")
        print("或: python predict_digit.py 你的图片.png")


if __name__ == "__main__":
    main()
