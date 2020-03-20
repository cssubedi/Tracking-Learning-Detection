#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from vehicle_detection_tracking.Detector import Detector
from vehicle_detection_tracking.Tracker import Tracker
from vehicle_detection_tracking.Introducer import Introducer
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import VehicleDetected, VehicleBoundingBox
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32
from mutex import mutex
import threading
from duckietown_utils.jpg import image_cv_from_jpg
from time import time


class TLD():
    def __init__(self):
        self.introduce = False
        try:
            pos_dist = np.load(
                "/home/ubuntu/duckietown/catkin_ws/src/vehicle_detection_tracking/distribution/posDist.npy").tolist()
            neg_dist = np.load(
                "/home/ubuntu/duckietown/catkin_ws/src/vehicle_detection_tracking/distribution/negDist.npy").tolist()
        except IOError:
            print "Object to be detected is not introduced"
            self.introduce = True
        self.active = True
        self.bridge = CvBridge()
        self.Detector = Detector()
        self.Detector.set_posterior(pos_dist, neg_dist)
        self.Tracker = Tracker()
        self.Introducer = Introducer()
        self.tracking = False
        self.sub_image = rospy.Subscriber("/autopilot/camera_node/image/compressed", CompressedImage, self.cbImage,
                                          queue_size=1)
        self.pub_image = rospy.Publisher("~image_with_detection", Image, queue_size=1)
        self.pub_vehicle_detected = rospy.Publisher("~vehicle_detected", VehicleDetected, queue_size=1)
        self.pub_vehicle_bbox = rospy.Publisher("~vehicle_bounding_box", VehicleBoundingBox, queue_size=1)
        self.lock = mutex()
        self.margin = 3
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

    def cbImage(self, image_msg):
        try:
            image_cv = image_cv_from_jpg(image_msg.data)
        except ValueError as e:
            print 'Could not decode image: %s' % (e)
            return
        if not self.active:
            return
        thread = threading.Thread(target=self.run, args=(image_cv,))
        thread.setDaemon(True)
        thread.start()

    def run(self, image_cv):
        if self.lock.testandset():
            vehicle_bounding_box_msg = VehicleBoundingBox()
            vehicle_detected_msg = VehicleDetected()
            image_cv = image_cv[80:200, 0:640]
            if not self.tracking:
                veh = self.Detector.run(image_cv)
                if veh == None:
                    vehicle_detected_msg = False
                    self.pub_vehicle_detected.publish(vehicle_detected_msg)
                    cv2.imshow("Image", image_cv)
                    cv2.waitKey(10)
                else:
                    vehicle_detected_msg = True
                    vehicle_bounding_box_msg.data = veh
                    cv2.rectangle(image_cv, (veh[0], veh[1]), (veh[2], veh[3]), (0, 255, 0), 2)
                    image_msg = self.bridge.cv2_to_imgmsg(image_cv, "bgr8")

                    self.pub_vehicle_bbox.publish(vehicle_bounding_box_msg)
                    self.pub_vehicle_detected.publish(vehicle_detected_msg)
                    self.pub_image.publish(image_msg)

                    veh = [veh[0] + self.margin, veh[1] + self.margin,
                           veh[2] - self.margin, veh[3] - self.margin]
                    self.Tracker.initialize(veh, image_cv)
                    self.tracking = True
                    cv2.imshow("Image", image_cv)
                    cv2.waitKey(10)
            else:
                veh = self.Tracker.run(image_cv)
                if veh == None:
                    rospy.loginfo("Tracking Failed")
                    self.tracking = False
                    vehicle_detected_msg = False
                    self.pub_vehicle_detected.publish(vehicle_detected_msg)
                    cv2.imshow("Image", image_cv)
                    cv2.waitKey(10)
                else:
                    veh = [veh[0] - self.margin, veh[1] - self.margin,
                           veh[2] + self.margin, veh[3] + self.margin]
                    vehicle_detected_msg = True
                    vehicle_bounding_box_msg.data = veh
                    print self.cal_distance(veh)
                    cv2.rectangle(image_cv, (veh[0], veh[1]), (veh[2], veh[3]), (255, 0, 0), 2)
                    image_msg = self.bridge.cv2_to_imgmsg(image_cv, "bgr8")

                    self.pub_vehicle_bbox.publish(vehicle_bounding_box_msg)
                    self.pub_vehicle_detected.publish(vehicle_detected_msg)
                    self.pub_image.publish(image_msg)
                    cv2.imshow("Image", image_cv)
                    cv2.waitKey(10)

            self.lock.unlock()

    def cal_distance(self, bbox):
        d = 14
        h = 6.5
        p = 120
        focal_length = (d * p) / h
        height = bbox[3] - bbox[1]
        distance = (h * focal_length) / height
        return distance


if __name__ == "__main__":
    rospy.init_node("vehicle_detection_tracking_node")
    vehicle_detection_tracking_node = TLD()
    rospy.spin()
