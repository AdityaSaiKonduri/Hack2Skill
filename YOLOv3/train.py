from tqdm import tqdm
import tensorflow as tf
import numpy as np
from model import build_yolov3
import loss
from dataset import YOLODataset, load_data

input_folder = "Hack2Skill\\DOTAv1\\images\\train"
output_folder = "DOTAv1\\preprocessed_images\\train"
input_labels_folder = "DOTAv1\\labels\\train"
output_labels_folder = "DOTAv1\\preprocessed_labels\\train"



def train_step(x, y0, y1, y2, model, optimizer, loss_fn, scaled_anchors):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        print("Train step entered")
        # loss0 = loss_fn.call(out[0], y0)
        # loss1 = loss_fn.call(out[1], y1)
        # loss2 = loss_fn.call(out[2], y2)
        loss= loss_fn.call(out, [y0, y1, y2])
        # loss = loss0 + loss1 + loss2
        
        # print(f"Loss components: {loss0.numpy()}, {loss1.numpy()}, {loss2.numpy()}")
        print(f"Total loss: {loss.numpy()}")
        # print(f"Class Accuracy : {class_accuracy.numpy() :.4f}")
        # print(f"Box Accuracy : {box_accuracy.numpy() :.4f}")
        
        if tf.math.is_nan(loss):
            print("NaN detected in loss!")
            print(f"Model output shapes: {[o.shape for o in out]}")
            print(f"Target shapes: {y0.shape}, {y1.shape}, {y2.shape}")
    
    if not tf.math.is_nan(loss):
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss#,class_accuracy,box_accuracy

def train_fn(dataset, model, optimizer, loss_fn, policy, scaled_anchors):
    losses = []
    class_accuracies = []
    box_accuracies = []
    for batch in tqdm(dataset, leave=True):
        x, (y0, y1, y2) = batch
        print(y0.shape)
        print(y1.shape)
        print(y2.shape)
        print("loss function eval")
        loss = train_step(x, y0, y1, y2, model=model, optimizer=optimizer, loss_fn=loss_fn, scaled_anchors=scaled_anchors)
        losses.append(loss.numpy())
        # class_accuracies.append(accuracy.numpy())
        # box_accuracies.appned(box_accu.numpy())
        # Update progress bar
        mean_loss = np.mean(losses)
        # mean_class_acc = np.mean(class_accuracies)
        # mean_box_acc = np.mean(box_accuracies)
        tqdm.write(f'Loss: {mean_loss:.4f}')
        # tqdm.write(f'Class Accuracy: {mean_class_acc:.4f}')
        # tqdm.write(f'Box Accuracy: {mean_box_acc:.4f}')

def main():
    model = build_yolov3(num_classes=15)

    anchors = [
        [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
        [0.3, 0.3, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6],
        [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8],
        [0.4, 0.1, 0.5, 0.1, 0.5, 0.7, 0.4, 0.7],
        [0.1, 0.4, 0.9, 0.4, 0.9, 0.5, 0.1, 0.5],
        [0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6],
        [0.3, 0.3, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5],
        [0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7],
        [0.1, 0.1, 0.7, 0.1, 0.7, 0.7, 0.1, 0.7]
    ]

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-4)
    print("Loss Class called")
    loss_function = loss.YoloLoss()
    print("Loss class ended")
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    S=[416//32, 416//16, 416//8] # scale array for the grids
    anchors = anchors[:3]
    print(type(anchors))
    train_ds = load_data(output_folder, batch_size=8, label_dir=input_labels_folder, anchors=anchors, S=[416//32, 416//16, 416//8])
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
        print(f"Entered epoch {epoch}")
        train_fn(train_ds, model, optimizer, loss_function, policy, scaled_anchors)
        # if epoch > 0 and epoch % 3 == 0:
        #     class_accuracy

main()