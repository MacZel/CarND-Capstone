import cv2
import datetime
import numpy as np
import tensorflow as tf
from PIL import {
    Image,
    ImageDraw
}

from styx_msgs.msg import TrafficLight

def filter_boxes(min_score, target_class, boxes, scores, classes):
    n_classes = len(classes)
    ids = []
    for class_i in range(n_classes):
        if scores[class_i] >= min_score and classes[class_i] == target_class:
            ids.append(class_i)
    return boxes[ids, ...], scores[ids, ...], classes[ids, ...]

def normalize_coords(boxes, img_height, img_width):
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * img_height
    box_coords[:, 1] = boxes[:, 1] * img_width
    box_coords[:, 2] = boxes[:, 2] * img_height
    box_coords[:, 3] = boxes[:, 3] * img_width
    return box_coords

COLORS = ['red', 'yellow', 'green']
IS_OUTPUT_IMG = False

def draw_bounding_boxes(img, boxes, classes, scores, color_id, width=2):
    img_draw = ImageDraw.Draw(img)
    for box_i in range(len(boxes)):
        bottom, left, top, right = boxes[box_i, ...]
        class_i = int(classes[box_i])
        fill = COLORS[color_id]
        img_draw.line([
            (left, top), (left, bottom), (right, bottom), (right, top), (left, top) 
        ], width=width, fill=fill)
        img_draw.rectangle([(left, bottom-20), (right, bottom)], outline=fill, fill=fill)
        img_draw.text((left, bottom-15), str(scores[class_i]), "black")

def init_graph(graph_pb):
    g = tf.Graph()
    with g.as_default():
        g_def = tf.GraphDef()
        with tf.gfile.Open(graph_pb, "rb") as infile:
            g_binary = infile.read()
            g_def.ParseFromString(g_binary)
            tf.import_graph_def(g_def, name="")
    return g


class TLClassifier(object):
    def __init__(self):
        self.detection_graph = init_graph("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb")
        self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
        self.detection_boxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")
        self.detection_scores = self.detection_graph.get_tensor_by_name("detection_scores:0")
        self.detection_classes = self.detection_graph.get_tensor_by_name("detection_classes:0")

        with tf.Session(graph=self.detection_graph) as session:
            self.session = session

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        CONFIDENCE_CUTOFF = 0.2
        TARGET_CLASS = 10

        COLOR_THRESHOLDS = [
            ([0, 100, 80], [10, 255, 255]), # R
            ([18, 0, 195], [35, 255, 255]), # Y
            ([35, 200, 60], [70, 255, 255]) # G
        ]

        color = TrafficLight.UNKNOWN
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        np_img = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        (boxes, scores, classes) = self.session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: np_img}
        )
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes, scores, classes = filter_boxes(CONFIDENCE_CUTOFF, TARGET_CLASS, boxes, scores, classes)
        if len(boxes) > 0:
            width, height = image.shape[-2::-1]
            box_coords = normalize_coords(boxes, img_height, img_width)           
            red_yellow_green = [0, 0, 0]
            for coord_i in range(len(box_coords)):
                bottom, left, top, right = box_coords[coord_i, ...]
                box_img = image[int(bottom):int(top), int(left):int(right), :]
                box_img = cv2.GaussianBlur(box_img, (3, 3), 0)
                hue_saturation_value = cv2.cvtColor(box_img, cv2.COLOR_RGB2HSV)
                mask = [0, 0, 0]
                box_height = hue_saturation_value.shape[0]
                box_width = hue_saturation_value.shape[1]
                if box_height < box_width:
                    for threshold_i, (lower_boundary, upper_boundary) in enumerate(COLOR_THRESHOLDS):
                        lower_boundary = np.array(lower_boundary, dtype="uint8")
                        upper_boundary = np.array(upper_boundary, dtype="uint8")
                        mask[threshold_i] = sum(np.hstack(cv2.inRange(hue_saturation_value, lower_boundary, upper_boundary)))
                else:
                    value = hue_saturation_value[:,:,2]

                    top_value = np.sum(value[:int(box_height/3), :])
                    middle_value = np.sum(value[int(box_height/3):int(box_height*2/3), :])
                    bottom_value = np.sum(value[int(box_height*2/3):, :])
                    max_value = max(top_value, middle_value, bottom_value)
 
                    if max_value != 0:
                        for i, item in enumerate([top_value, middle_value, bottom_value]):
                            if item / max_value == 1:
                                mask[i] = 1
                                break
                    else:
                        mask = [1, 0, 0]
                red_yellow_green[mask.index(max(mask))] += 1
            color = red_yellow_green.index(max(red_yellow_green))

            if IS_OUTPUT_IMG:
                img_filepath = "../../../imgs/" + str(datetime.datetime.now()) + ".png"
                pil_img = Image.fromarray(image)
                draw_bounding_boxes(pil_img, box_coords, classes, scores, color)
                pil_img.save(img_filepath)
        return color
