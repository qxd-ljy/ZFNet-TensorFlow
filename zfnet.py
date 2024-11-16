from tensorflow.keras import layers, models


def create_zfnet_model(input_shape=(32, 32, 1), num_classes=10):
    """
    创建 ZFNet 模型。

    参数：
    - input_shape: 输入图像的形状，默认 (32, 32, 1)。
    - num_classes: 类别数目，默认 10。

    返回：
    - 返回构建好的模型。
    """
    model = models.Sequential()

    # 使用 Input 层显式定义输入形状
    model.add(layers.Input(shape=input_shape))  # 显式指定输入形状

    # 特征提取部分
    model.add(layers.Conv2D(64, (7, 7), activation='relu', strides=2, padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))

    # 扁平化层
    model.add(layers.Flatten())

    # 全连接层
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))

    # 输出层
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
