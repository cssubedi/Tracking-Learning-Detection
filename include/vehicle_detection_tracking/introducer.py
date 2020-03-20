#!/usr/bin/env python
import cv2
import sys
import numpy as np
from detector import Detector
import matplotlib.pyplot as plt


class Introducer():
    def __init__(self):
        try:
            self.template = cv2.imread("/home/ubuntu/duckietown/catkin_ws/src/"
                                       "vehicle_detection_tracking/ObjectModel/template001.jpg", 0)
        except IOError:
            print "Object to detect is unknown."
        self.detector = Detector()
        self.detector.init_distribution()
        self.detector.sliding_windows()
        self.num_samples = 0

    def patch_matching(self, image):
        correlation = 0
        vehicle = None
        ind = 0
        while ind < len(self.init_windows):
            win = self.init_windows[ind:ind + 4]
            ind += 4
            patch = image[win[1]:win[3], win[0]:win[2]]
            template = cv2.resize(self.template, (win[2] - win[0], win[3] - win[1]),
                                  interpolation=cv2.INTER_LINEAR)
            matching = cv2.matchTemplate(patch, template, method=cv2.cv.CV_TM_CCOEFF_NORMED)[0][0]
            if matching > correlation:
                correlation = matching
                vehicle = win
        return vehicle, correlation

    def run(self, frame):
        if self.num_samples < 100:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vehicle, correlation = self.patch_matching(image)
            if correlation > 0.7:
                self.Detector.train_positive(vehicle, frame, 1, 0)
                self.Detector.train_negative(frame, vehicle, 10)
                self.num_samples += 1
            else:
                pass

    def show_image(self, image, veh):
        cv2.rectangle(image, (veh[0], veh[1]), (veh[2], veh[3]), (0, 255, 0), 1)
        cv2.imshow("Introducer", image)
        cv2.waitKey(10)

    def show_template(self):
        cv2.imshow("Template", self.template)
        cv2.waitKey(10)

    def save_distribution(self):
        self.detector.normalise_hist()
        pos, neg = self.Detector.get_posterior()
        np.save("~/duckietown/catkin_ws/src/vehicle_detection_tracking/distribution/posDist", pos)
        np.save("~/duckietown/catkin_ws/src/vehicle_detection_tracking/distribution/negDist", neg)

    def get_num_samples(self):
        return self.num_samples
