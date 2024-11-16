import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from zfnet import create_zfnet_model  # 从 zfnet.py 导入模型创建函数


def prepare_data():
    """
    准备 MNIST 数据集并进行预处理
    :return: 训练集和测试集的图像及标签
    """
    # 加载 MNIST 数据集
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 数据预处理：标准化、调整大小、添加维度
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 调整图像大小并添加额外维度 (32x32, 1通道)
    x_train = tf.image.resize(x_train[..., tf.newaxis], (32, 32))
    x_test = tf.image.resize(x_test[..., tf.newaxis], (32, 32))

    # 确保数据类型是 float32
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    # 类别标签 one-hot 编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def initialize_device():
    """
    检查可用的计算设备（GPU/CPU）
    :return: 设备字符串
    """
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    print(f"Using device: {device}")
    return device


def initialize_model(input_shape=(32, 32, 1), num_classes=10):
    """
    创建并初始化 ZFNet 模型
    :param input_shape: 输入图像的形状
    :param num_classes: 分类类别数
    :return: 初始化的模型
    """
    model = create_zfnet_model(input_shape=input_shape, num_classes=num_classes)
    return model


def compile_model(model):
    """
    编译模型
    :param model: 待编译的模型
    :return: 已编译的模型
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, x_test, y_test, device, epochs=5, batch_size=128):
    """
    在指定设备上训练模型
    :param model: 训练的模型
    :param x_train: 训练集图像
    :param y_train: 训练集标签
    :param x_test: 测试集图像
    :param y_test: 测试集标签
    :param device: 设备
    :param epochs: 训练轮数
    :param batch_size: 批处理大小
    """
    with tf.device(device):
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))


def save_model(model, model_path='zfnet_mnist.keras'):
    """
    保存模型到文件
    :param model: 训练后的模型
    :param model_path: 保存模型的路径
    """
    model.save(model_path)
    print(f"Model saved as '{model_path}'")

if __name__ == '__main__':
    """
      主函数：组织所有步骤的执行
      """
    # 准备数据
    x_train, y_train, x_test, y_test = prepare_data()

    # 设备选择
    device = initialize_device()

    # 模型初始化
    model = initialize_model(input_shape=(32, 32, 1), num_classes=10)

    # 编译模型
    model = compile_model(model)

    # 训练模型
    train_model(model, x_train, y_train, x_test, y_test, device, epochs=5, batch_size=128)

    # 保存模型
    save_model(model, model_path='zfnet_mnist.keras')
