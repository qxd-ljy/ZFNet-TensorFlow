import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """
    加载并预处理 MNIST 数据集
    :return: 训练集和测试集的图像及标签
    """
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 数据预处理：标准化、调整大小、添加维度
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = tf.image.resize(x_train[..., tf.newaxis], (32, 32))
    x_test = tf.image.resize(x_test[..., tf.newaxis], (32, 32))

    # 类别标签 one-hot 编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def load_trained_model(model_path='zfnet_mnist.h5'):
    """
    加载训练好的模型
    :param model_path: 模型文件路径
    :return: 加载的模型
    """
    return load_model(model_path)


def evaluate_model(model, x_test, y_test):
    """
    评估模型在测试集上的表现
    :param model: 训练好的模型
    :param x_test: 测试集图像
    :param y_test: 测试集标签
    :return: 测试集上的损失和准确率
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")
    return test_loss, test_acc


def visualize_predictions(model, x_test, y_test, num_images=6):
    """
    可视化模型对多张测试图片的预测结果
    :param model: 训练好的模型
    :param x_test: 测试集图像
    :param y_test: 测试集标签
    :param num_images: 显示图像的数量
    """
    predictions = model.predict(x_test[:num_images])
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y_test[:num_images], axis=1)

    # 绘制结果
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()

    for i in range(num_images):
        ax = axes[i]
        # 将 Tensor 转换为 NumPy 数组，并使用 reshape
        img = x_test[i].numpy().reshape(32, 32)  # 这里调用 .numpy() 将 Tensor 转换为 NumPy 数组
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Pred: {predicted_labels[i]} | Actual: {actual_labels[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    主函数，组织数据加载、模型加载、评估和可视化步骤
    """
    # 加载并预处理数据
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # 加载训练好的模型
    model = load_trained_model()

    # 评估模型
    evaluate_model(model, x_test, y_test)

    # 可视化预测结果
    visualize_predictions(model, x_test, y_test, num_images=6)

    # 在整个测试集上进行预测并计算精度
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y_test, axis=1)

    # 计算整体准确率
    accuracy = np.sum(predicted_labels == actual_labels) / len(actual_labels)
    print(f"Accuracy on the entire test set: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
