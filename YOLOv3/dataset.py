import os
import numpy as np
import tensorflow as tf
import cv2
from shapely.geometry import Polygon

class YOLODataset:
    def __init__(self, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=15, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = np.array(anchors[0] + anchors[1] + anchors[2], dtype=np.float32)  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    def __len__(self):
        return len(self.image_files)

    def load_image_and_labels(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        # Load and preprocess the image using OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Load and preprocess the labels
        bboxes = np.loadtxt(label_path, delimiter=" ", ndmin=2).tolist()
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Build the targets
        targets = [np.zeros((self.num_anchors_per_scale, S, S, 10)) for S in self.S]
        for box in bboxes:
            iou_anchors = self.calculate_polygon_iou(np.array(box[1:9]), self.anchors)
            anchor_indices = np.argsort(iou_anchors)[::-1]
            class_label,x1, y1, x2, y2, x3, y3, x4, y4 = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y1), int(S * x1)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x1_cell, y1_cell = S * x1 - j, S * y1 - i
                    x2_cell, y2_cell = S * x2 - j, S * y2 - i
                    x3_cell, y3_cell = S * x3 - j, S * y3 - i
                    x4_cell, y4_cell = S * x4 - j, S * y4 - i
                    box_coordinates = [x1_cell, y1_cell, x2_cell, y2_cell, x3_cell, y3_cell, x4_cell, y4_cell]
                    targets[scale_idx][anchor_on_scale, i, j, 2:] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 1] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, targets

    def calculate_polygon_iou(self, box, anchors):
        def iou(box, anchors):
            box_polygon = Polygon([(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])])
            iou_scores = []
            for anchor in anchors:
                anchor_polygon = Polygon([(anchor[0], anchor[1]), (anchor[2], anchor[3]), 
                                        (anchor[4], anchor[5]), (anchor[6], anchor[7])])
                intersection_area = box_polygon.intersection(anchor_polygon).area
                union_area = box_polygon.union(anchor_polygon).area
                iou_score = intersection_area / union_area
                iou_scores.append(iou_score)
            return np.array(iou_scores)
        return iou(box, anchors)

def load_data(img_dir, label_dir, anchors, batch_size=16, image_size=416, S=[13, 26, 52], C=15, transform=None):
    dataset = YOLODataset(img_dir, label_dir, anchors, image_size, S, C, transform)

    def generator():
        for i in range(len(dataset)):
            image, targets = dataset.load_image_and_labels(i)
            yield image, tuple(targets)  # Ensure that targets is a tuple

    # Define output types and shapes
    output_types = (tf.float32, (tf.float32, tf.float32, tf.float32))
    output_shapes = (tf.TensorShape([image_size, image_size, 3]),
                     (tf.TensorShape([dataset.num_anchors_per_scale, S[0], S[0], 10]),
                      tf.TensorShape([dataset.num_anchors_per_scale, S[1], S[1], 10]),
                      tf.TensorShape([dataset.num_anchors_per_scale, S[2], S[2], 10])))

    # Create TensorFlow dataset from generator
    tf_dataset = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)
    
    # Apply batching and prefetching
    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return tf_dataset

# Example usage
img_dir = "DOTAv1/images/train"
label_dir = "DOTAv1/labels/train"
anchors = [
    np.array([0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2], dtype=np.float32),
    np.array([0.3, 0.3, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6], dtype=np.float32),
    np.array([0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], dtype=np.float32),
    np.array([0.4, 0.1, 0.5, 0.1, 0.5, 0.7, 0.4, 0.7], dtype=np.float32),
    np.array([0.1, 0.4, 0.9, 0.4, 0.9, 0.5, 0.1, 0.5], dtype=np.float32),
    np.array([0.5, 0.5, 0.6, 0.5, 0.6, 0.6, 0.5, 0.6], dtype=np.float32),
    np.array([0.3, 0.3, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5], dtype=np.float32),
    np.array([0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7], dtype=np.float32)
]
anchors = [anchors[:3], anchors[3:5], anchors[5:]]

train_ds = load_data(img_dir, label_dir, anchors)

# Inspect dataset
print(train_ds)

# print(train_ds)
