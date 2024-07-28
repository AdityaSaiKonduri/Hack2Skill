import tensorflow as tf
import numpy as np
from keras import layers, Model

def build_yolov3(input_shape=(416, 416, 3), num_classes=15):
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv layers
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    x = layers.Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # First residual block
    residual = x
    x = layers.Conv2D(32, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Add()([residual, x])
    
    # Conv layers before second residual block
    x = layers.Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Second residual block
    for _ in range(2):
        residual = x
        x = layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Add()([residual, x])
    
    # Conv layers before third residual block
    x = layers.Conv2D(256, 3, strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Third residual block
    for _ in range(8):
        residual = x
        x = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Add()([residual, x])
    route1 = x  # Save for later
    
    # Conv layers before fourth residual block
    x = layers.Conv2D(512, 3, strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Fourth residual block
    for _ in range(8):
        residual = x
        x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Add()([residual, x])
    route2 = x  # Save for later
    
    # Conv layers before fifth residual block
    x = layers.Conv2D(1024, 3, strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # Fifth residual block
    for _ in range(4):
        residual = x
        x = layers.Conv2D(512, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Conv2D(1024, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.Add()([residual, x])
    
    # YOLOv3 head
    x = layers.Conv2D(512, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(1024, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(512, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(1024, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(512, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    # First scale prediction
    scale1 = layers.Conv2D(1024, 3, padding='same', use_bias=False)(x)
    scale1 = layers.BatchNormalization()(scale1)
    scale1 = layers.LeakyReLU(0.1)(scale1)
    scale1 = layers.Conv2D((num_classes + 9), 1, padding='same')(scale1)
    scale1 = layers.Reshape((scale1.shape[1], scale1.shape[2], 1, num_classes+9))(scale1)
    # output1 = layers.Reshape((1, -1, 3, num_classes + 5))(scale1)
    # output1 = layers.Permute((1, 3, 2, 4))(output1)
    
    # Upsample and concatenate with route2
    x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, route2])
    
    # Second scale prediction
    x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    scale2 = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    scale2 = layers.BatchNormalization()(scale2)
    scale2 = layers.LeakyReLU(0.1)(scale2)
    scale2 = layers.Conv2D((num_classes + 9), 1, padding='same')(scale2)
    scale2 = layers.Reshape((scale2.shape[1], scale2.shape[2], 1, num_classes+9))(scale2)
    # output2 = layers.Reshape((1, -1, 3, num_classes + 5))(scale2)
    # output2 = layers.Permute((1, 3, 2, 4))(output2)
    
    # Upsample and concatenate with route1
    x = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, route1])
    
    # Third scale prediction
    x = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Conv2D(128, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    
    scale3 = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    scale3 = layers.BatchNormalization()(scale3)
    scale3 = layers.LeakyReLU(0.1)(scale3)
    scale3 = layers.Conv2D((num_classes + 9), 1, padding='same')(scale3)
    scale3 = layers.Reshape((scale3.shape[1], scale3.shape[2], 1, num_classes+9))(scale3)
    # output3 = layers.Reshape((1, -1, 3, num_classes + 5))(scale3)
    # output3 = layers.Permute((1, 3, 2, 4))(output3)
    
    model = Model(inputs=inputs, outputs=[scale1, scale2, scale3])
    return model

# Example usage
# model = build_yolov3()
# model.summary()

# # Print output shapes
# dummy_input = tf.random.normal((1, 416, 416, 3))
# outputs = model(dummy_input)
# print("\nOutput shapes:")
# for i, output in enumerate(outputs):
#     print(f"Output {i + 1}: {output.shape}")

def test():
    num_classes = 15
    img_size = 416
    model = build_yolov3(num_classes=num_classes)
    
    # Generate random input data
    x = np.random.randn(16, img_size, img_size, 3).astype(np.float32)
    print("Inside Model")
    print(x.shape)
    
    # Forward pass through the model
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print("Outside model")
    # Check output shapes
    # assert out[0].shape == (2, img_size // 32, img_size // 32, 3 * (9 + num_classes)), f"Expected shape: {(2, img_size // 32, img_size // 32, 3 * (9 + num_classes))}, but got: {out[0].shape}"
    # assert out[1].shape == (2, img_size // 16, img_size // 16, 3 * (9 + num_classes)), f"Expected shape: {(2, img_size // 16, img_size // 16, 3 * (9 + num_classes))}, but got: {out[1].shape}"
    # assert out[2].shape == (2, img_size // 9, img_size // 9, 3 * (9 + num_classes)), f"Expected shape: {(2, img_size // 9, img_size // 9, 3 * (9 + num_classes))}, but got: {out[2].shape}"

# test()
