import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon

def iou(boxes, anchors):
    def calculate_iou(box, anchor):
        try:
            box_polygon = Polygon([(box[i].numpy(), box[i+1].numpy()) for i in range(0, 8, 2)])
            anchor_polygon = Polygon([(anchor[i].numpy(), anchor[i+1].numpy()) for i in range(0, 8, 2)])
            
            intersection_area = box_polygon.intersection(anchor_polygon).area
            union_area = box_polygon.union(anchor_polygon).area
            
            return intersection_area / (union_area + 1e-10)
        except Exception as e:
            tf.print("Error in IOU calculation:", e)
            return 0.0
    
    iou_matrix = [[calculate_iou(box, anchor) for anchor in anchors] for box in boxes]
    return tf.convert_to_tensor(iou_matrix, dtype=tf.float32)

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=15, num_boxes=1):
        super(YoloLoss, self).__init__(name='yolo_loss')
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.huber = tf.keras.losses.Huber(delta=1.0)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.lambda_class = 1.0
        self.lambda_noobj = 0.5
        self.lambda_obj = 1.0
        self.lambda_box = 5.0

    @tf.function
    def call(self, y_true, y_pred):
        total_loss = 0
        num_scales = len(y_true)
        
        for scale_idx in range(num_scales):
            scale_loss= self.compute_loss_per_scale(y_true[scale_idx], y_pred[scale_idx])
            total_loss += scale_loss
            # accuracy +=accuracy
            # box_accu +=box_accu
        
        return total_loss / num_scales#, accuracy/num_scales , box_accu/num_scales

    @tf.function
    def compute_loss_per_scale(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-4, 1.0)
        y_true = tf.clip_by_value(y_true, 1e-4, 1.0)
        # Ensure inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Reshape inputs
        batch_size = tf.shape(y_true)[0]
        # print("Batch size : ", batch_size)
        grid_size = tf.shape(y_true)[1]
        # print("Grid size : ", grid_size)
        y_true = tf.reshape(y_true, [batch_size, grid_size, grid_size, self.num_boxes, -1])
        y_pred = tf.reshape(y_pred, [batch_size, grid_size, grid_size, self.num_boxes, -1])

        # Split predictions
        pred_obj, pred_class, pred_box = tf.split(
            y_pred, 
            [1, self.num_classes, 8], 
            axis=-1
        )

        # print("Prediction split : ", pred_obj, pred_class, pred_box) 

        # Split ground truth
        true_obj, true_class, true_box = tf.split(
            y_true, 
            [1, self.num_classes, 8], 
            axis=-1
        )
        # print("True split : ", true_obj, true_class, true_box) 

        # Object mask
        obj_mask = tf.squeeze(true_obj, -1)
        
        # No-object loss
        no_obj_loss = self.bce(true_obj, pred_obj)
        no_obj_loss = self.lambda_noobj * tf.reduce_sum(no_obj_loss * (1 - obj_mask))

        # Object loss
        obj_loss = self.bce(true_obj, pred_obj)
        obj_loss = self.lambda_obj * tf.reduce_sum(obj_loss * obj_mask)

        # Box coordinates loss
        box_loss = self.huber(true_box, pred_box)
        box_loss = self.lambda_box * tf.reduce_sum(box_loss * tf.expand_dims(obj_mask, -1))

        # Class loss
        class_loss = self.categorical_crossentropy(true_class, pred_class)
        class_loss = self.lambda_class * tf.reduce_sum(class_loss * obj_mask)

        # Total loss
        total_loss = no_obj_loss + obj_loss + box_loss + class_loss

        # Normalize by number of objects
        num_objects = tf.maximum(tf.reduce_sum(obj_mask), 1)
        total_loss /= num_objects

        # Object mask
        # obj_mask = tf.squeeze(true_obj, -1)

        # # Calculate class accuracy
        # pred_class_label = tf.argmax(pred_class, axis=-1)
        # true_class_label = tf.argmax(true_class, axis=-1)
        # class_accuracy = tf.reduce_sum(tf.cast(tf.equal(pred_class_label, true_class_label), tf.float32) * obj_mask)
        
        # # Calculate IoU for boxes
        # batch_iou = []
        # for b in range(batch_size):
        #     boxes = tf.where(obj_mask[b], true_box[b], tf.zeros_like(true_box[b])).numpy().tolist()
        #     anchors = tf.where(obj_mask[b], pred_box[b], tf.zeros_like(pred_box[b])).numpy().tolist()
        #     if len(boxes) > 0 and len(anchors) > 0:
        #         batch_iou.append(tf.reduce_mean(iou(boxes, anchors)))
        #     else:
        #         batch_iou.append(0.0)
        
        # mean_iou = tf.reduce_mean(batch_iou)

        # # Normalize by number of objects
        # num_objects = tf.maximum(tf.reduce_sum(obj_mask), 1)
        # class_accuracy /= num_objects

        return total_loss#, class_accuracy, mean_iou

    @tf.function
    def calculate_iou(box, anchor):
        try:
            box_polygon = Polygon([(box[i].numpy(), box[i+1].numpy()) for i in range(0, 8, 2)])
            anchor_polygon = Polygon([(anchor[i].numpy(), anchor[i+1].numpy()) for i in range(0, 8, 2)])
            
            intersection_area = box_polygon.intersection(anchor_polygon).area
            union_area = box_polygon.union(anchor_polygon).area
            
            return intersection_area / (union_area + 1e-10)
        except Exception as e:
            tf.print("Error in IOU calculation:", e)
            return 0.0
        # return tf.reduce_mean(tf.abs(true_box - pred_box), axis=-1)

    @tf.function
    def debug_info(self, y_true, y_pred):
        """Provides debugging information for the loss calculation."""
        info = {}
        for i, (scale_true, scale_pred) in enumerate(zip(y_true, y_pred)):
            info[f'scale_{i}'] = {
                'y_true_shape': tf.shape(scale_true),
                'y_pred_shape': tf.shape(scale_pred),
                'y_true_range': (tf.reduce_min(scale_true), tf.reduce_max(scale_true)),
                'y_pred_range': (tf.reduce_min(scale_pred), tf.reduce_max(scale_pred)),
                'y_true_has_nan': tf.reduce_any(tf.math.is_nan(scale_true)),
                'y_pred_has_nan': tf.reduce_any(tf.math.is_nan(scale_pred)),
                'y_true_has_inf': tf.reduce_any(tf.math.is_inf(scale_true)),
                'y_pred_has_inf': tf.reduce_any(tf.math.is_inf(scale_pred)),
            }
        return info

# Example usage
def example_usage():
    loss_fn = YoloLoss(num_classes=15, num_boxes=1)
    
    # Example data (adjust shapes as needed)
    y_true = [
        tf.random.uniform((16, 13, 13, 1, 24), minval=0, maxval=1, dtype=tf.float32),
        tf.random.uniform((16, 26, 26, 1, 24), minval=0, maxval=1, dtype=tf.float32),
        tf.random.uniform((16, 52, 52, 1, 24), minval=0, maxval=1, dtype=tf.float32)
    ]
    y_pred = [
        tf.random.uniform((16, 13, 13, 1, 24), minval=0, maxval=1, dtype=tf.float32),
        tf.random.uniform((16, 26, 26, 1, 24), minval=0, maxval=1, dtype=tf.float32),
        tf.random.uniform((16, 52, 52, 1, 24), minval=0, maxval=1, dtype=tf.float32)
    ]

    # print(len(y_true[0]))
    # print(len(y_true[0][0]))
    # print(len(y_true[0][0][0]))
    # print(len(y_true[0][0][0][0]))
    # print(len(y_true[0][0][0][0][0]))
    # Calculate loss
    loss_value = loss_fn(y_true, y_pred)
    print("Loss value:", loss_value.numpy())

    # Get debug info
    debug_info = loss_fn.debug_info(y_true, y_pred)
    print("Debug info:", debug_info["scale_0"]["y_true_has_nan"])

if __name__ == "__main__":
    example_usage()