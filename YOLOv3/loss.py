import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon

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

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(YoloLoss, self).__init__(name='yolo_loss')
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def call(self, y_true, y_pred, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = tf.equal(y_true[..., 0], 1)  # in paper this is Iobj_i
        noobj = tf.equal(y_true[..., 0], 0)  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.bce(
            y_true[..., 0:1][noobj], y_pred[..., 0:1][noobj]
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        anchors = tf.reshape(anchors, (1, 3, 3, 1, 8))
        box_preds = self.sigmoid(y_pred[..., 2:10])
        ious = iou(box_preds[obj], y_true[..., 2:10][obj])
        object_loss = self.mse(self.sigmoid(y_pred[..., 0:1][obj]), ious * y_true[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        y_pred_boxes = self.sigmoid(y_pred[..., 2:10])
        y_true_boxes = y_true[..., 2:10]  # assuming y_true already has the correct format
        box_loss = self.mse(y_pred_boxes[obj], y_true_boxes[obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        class_loss = self.entropy(y_true[..., 1:2][obj], y_pred[..., 1:2][obj])

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
