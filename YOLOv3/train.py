from tqdm import tqdm
import tensorflow as tf
import numpy as np
from model import build_yolov3
from loss import YoloLoss
from dataset import YOLODataset, load_data

input_folder = "DOTAv1\\images\\train"
output_folder = "DOTAv1\\preprocessed_images\\train"
input_labels_folder = "DOTAv1\\labels\\train"
output_labels_folder = "DOTAv1\\preprocessed_labels\\train"

anchors = [
    np.array([0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2], dtype=np.float32),
    np.array([0.3, 0.3, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6], dtype=np.float32),
    np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], dtype=np.float32),
    np.array([0.4, 0.1, 0.5, 0.1, 0.5, 0.7, 0.4, 0.7], dtype=np.float32),
    np.array([0.1, 0.4, 0.9, 0.4, 0.9, 0.5, 0.1, 0.5], dtype=np.float32),
    np.array([0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6], dtype=np.float32),
    np.array([0.3, 0.3, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5], dtype=np.float32),
    np.array([0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7], dtype=np.float32),
    np.array([0.1, 0.1, 0.7, 0.1, 0.7, 0.7, 0.1, 0.7], dtype=np.float32)
]

# Group the anchors for different scales
anchors = [anchors[:3], anchors[3:6], anchors[6:]]

def train_step(x, y0, y1, y2, model, optimizer, loss_fn, scaled_anchors):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        loss = (
            loss_fn(out, y0) * scaled_anchors[0]
            + loss_fn(out, y1) * scaled_anchors[1]
            + loss_fn(out, y2) * scaled_anchors[2]
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_fn(dataset, model, optimizer, loss_fn, policy, scaled_anchors):
    losses = []
    for batch in tqdm(dataset, leave=True):
        x, (y0, y1, y2) = batch
        loss = train_step(x, y0, y1, y2, model=model, optimizer=optimizer, loss_fn=loss_fn, scaled_anchors=scaled_anchors)
        losses.append(loss.numpy())

        # Update progress bar
        mean_loss = np.mean(losses)
        tqdm.write(f'Loss: {mean_loss:.4f}')

def main():
    model = build_yolov3(num_classes=15)

    anchors = [
        np.array([0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2], dtype=np.float32),
        np.array([0.3, 0.3, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6], dtype=np.float32),
        np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], dtype=np.float32),
        np.array([0.4, 0.1, 0.5, 0.1, 0.5, 0.7, 0.4, 0.7], dtype=np.float32),
        np.array([0.1, 0.4, 0.9, 0.4, 0.9, 0.5, 0.1, 0.5], dtype=np.float32),
        np.array([0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6], dtype=np.float32),
        np.array([0.3, 0.3, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5], dtype=np.float32),
        np.array([0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7], dtype=np.float32),
        np.array([0.1, 0.1, 0.7, 0.1, 0.7, 0.7, 0.1, 0.7], dtype=np.float32)
    ]

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-4)
    loss_function = YoloLoss()
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    S=[416//32, 416//16, 416//8] # scale array for the grids
    anchors = [anchors[:3], anchors[3:6], anchors[6:]]
    train_ds = load_data(output_folder, input_labels_folder, anchors=anchors, S=[416//32, 416//16, 416//8])
    # test_ds = load_data(, anchors, S=[416//32, 416//16, 416//8])

    # Convert ANCHORS and S to TensorFlow tensors
    anchors = tf.constant(anchors, dtype=tf.float32)
    scales = tf.constant(S, dtype=tf.float32)

    # Expand dimensions of scales and repeat to match the shape of anchors
    scales_expanded = tf.expand_dims(scales, axis=-1)
    # print(scales_expanded.shape)
    scales_expanded = tf.expand_dims(scales_expanded, axis=-1)
    # print(scales_expanded.shape)
    scales_tiled = tf.tile(scales_expanded, [1, 3, 8])
    # print(scales_tiled.shape)
    # Perform element-wise multiplication and move to the specified device
    scaled_anchors = anchors * scales_tiled

    # print(scaled_anchors)

    for epoch in range(100):
        train_fn(train_ds, model, optimizer, loss_function, policy, scaled_anchors)
        # if epoch > 0 and epoch % 3 == 0:
        #     class_accuracy

main()