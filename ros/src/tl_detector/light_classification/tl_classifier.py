import tensorflow as tf

from styx_msgs.msg import TrafficLight


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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
