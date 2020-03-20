#!/usr/bin/env python

import cv2
import os
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import itertools
from time import time
from time import clock
from scipy import stats


class Detector():
    def __init__(self):
        self.img = None
        self.viz = None
        self.nccImg = None
        self.integral_img = None
        self.fig = 0
        self.vehicle_variance = 0
        self.init_windows = []
        self.windows = []
        self.numWindows = 0
        self.es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.init_fern()
        self.sliding_windows()

    def sliding_windows(self):
        scale_factor = 1.2
        min_window_width = 30
        min_window_height = 36
        img_height = 120
        img_width = 640
        win_size_w = min_window_width
        win_size_h = min_window_height
        shift_w = int(round(0.1 * win_size_w))
        shift_h = int(round(0.1 * win_size_h))
        index = 0

        while win_size_w < img_width / 4 and win_size_h < img_height:
            for x in xrange(0, img_width - win_size_w, shift_w):
                for y in xrange(0, img_height - win_size_h, shift_h):
                    self.init_windows[index:index + 4] = [x, y, x + win_size_w, y + win_size_h]
                    index += 4
                    self.numWindows += 1
                self.init_windows.append("switch")

            win_size_h = int(round(scale_factor * win_size_h))
            win_size_w = int(round(scale_factor * win_size_w))
            shift_w = int(round(0.1 * win_size_w))
            shift_h = int(round(0.1 * win_size_h))
        return self.init_windows

    def init_fern(self):
        self.posteriors_pos = [1] * (2 ** 10) * 10
        self.posteriors_neg = [1] * (2 ** 10) * 10

    def train(self, win, image, label, step):
        self.img = self.image_transformation(image)
        featureVector = self.fern(win, label, step)
        self.update_fern(featureVector)

    def mean_filter(self, win):
        _sum = self.integral_img[win[1], win[0]] + self.integral_img[win[3] + 1, win[2] + 1] - \
               self.integral_img[win[3] + 1, win[0]] - self.integral_img[win[1], win[2] + 1]
        numPixels = (win[2] - win[0] + 1) * (win[3] - win[1] + 1)
        avg = _sum / numPixels

        win = [win[0], win[1], win[2], (win[3] + win[1]) / 2]

        _sum = self.integral_img[win[1], win[0]] + self.integral_img[win[3] + 1, win[2] + 1] - \
               self.integral_img[win[3] + 1, win[0]] - self.integral_img[win[1], win[2] + 1]
        numPixels = (win[2] - win[0] + 1) * (win[3] - win[1] + 1)
        avg_upper = _sum / numPixels

        if 10 <= avg <= 30:
            if avg_upper <= 5:
                return True
        else:
            return False

    def bool2binary(self, boolean):
        return ''.join(['1' if x else '0' for x in boolean])

    def int2binary(self, integer):
        return '{0:08b}'.format(integer)

    def fern(self, win, label, step):
        if not step:
            height = win[3] - win[1]
            width = win[2] - win[0]
            step_w = width / 10.0
            step_h = height / 10.0
            featureVector = [0] * 10

            for i in range(5, 10):
                boolean = []
                for j in range(10):
                    sec = [int(round(win[0] + j * step_w)), int(round(win[1] + i * step_h)),
                           int(round(win[0] + (j + 1) * step_w)), int(round(win[1] + (i + 1) * step_h))]
                    _sum = np.sum(self.img[sec[1]:sec[3], sec[0]:sec[2]])
                    avg = int(round(float(_sum) / (step_h * step_w), 2))
                    if avg < 127:
                        boolean.append(False)
                    else:
                        boolean.append(True)
                featureVector[i] = self.bool2binary(boolean)

            featureVector.append(label)
            self.prev_featureVector = featureVector
            return featureVector
        else:
            height = win[3] - win[1]
            width = win[2] - win[0]
            step_w = width / 10.0
            step_h = height / 10.0
            featureVector = [0] * 10

            for i in range(5, 10):
                if i < 10 - step:
                    featureVector[i] = self.prev_featureVector[i + step]
                    continue
                boolean = []
                for j in range(10):
                    sec = [int(round(win[0] + j * step_w)), int(round(win[1] + i * step_h))
                        , int(round(win[0] + (j + 1) * step_w)), int(round(win[1] + (i + 1) * step_h))]
                    _sum = np.sum(self.img[sec[1]:sec[3], sec[0]:sec[2]])
                    avg = int(round(float(_sum) / (step_h * step_w), 2))
                    if avg < 127:
                        boolean.append(False)
                    else:
                        boolean.append(True)
                featureVector[i] = self.bool2binary(boolean)

            featureVector.append(label)
            self.prev_featureVector = featureVector
            return featureVector

    def update_fern(self, featureVector):
        if featureVector[-1] == 1:
            for classifier in range(5, 10):
                ind = classifier * 2 ** 10
                self.posteriors_pos[ind + int(featureVector[classifier], 2)] += 1

        if featureVector[-1] == 0:
            for classifier in range(5, 10):
                ind = classifier * 2 ** 10
                self.posteriors_neg[ind + int(featureVector[classifier], 2)] += 1

    def cal_prob(self, featureVector):
        list_prob = []
        for classifier in range(5, 10):
            ind = classifier * 2 ** 10
            pos = self.posteriors_pos[ind + int(featureVector[classifier], 2)]
            neg = self.posteriors_neg[ind + int(featureVector[classifier], 2)]

            if pos == 0.0 and neg == 0.0:
                prob = 0.1
                list_prob.append(prob)
            else:
                prob = round(float(pos) / (pos + neg), 3)
                list_prob.append(prob)
        return list_prob

    def isVehicle(self, prob):
        sum_prob = sum(prob)
        average_prob = float(sum_prob) / len(prob)
        if average_prob > 0.5:
            return True
        return False

    def nearest_neighbor(self, list_vehicle, method):
        _, _, files = os.walk("/home/ubuntu/TLD/ObjectModel").next()
        numTemplate = len(files)
        correlations = []

        for veh in list_vehicle:
            image = self.nccImg[veh[1]:veh[3], veh[0]:veh[2]]
            shape = image.shape
            ncc = []

            for i in range(1, numTemplate + 1):
                template = cv2.imread("/home/ubuntu/temp/tld_bag/template{:>03}.jpg".format(i), 0)
                template = cv2.resize(template, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                matching = cv2.matchTemplate(image, template, method)[0][0]
                ncc.append(matching)

            max_ncc = max(ncc)
            correlations.append(max_ncc)

        max_correlation = max(correlations)
        if max_correlation > 0.5:
            return list_vehicle[correlations.index(max_correlation)]
        return None

    def generate_negative(self, image, bbox, num):
        img = self.image_transformation(image.copy())
        self.integral_img = cv2.integral(img)
        i = 0
        ind = 4000

        while ind < len(self.init_windows) and i < num:
            if self.init_windows[ind] != "switch":
                box = self.init_windows[ind:ind + 4]
                ind += 4
                if self.mean_filter(box):
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    bbox_new = [bbox[0] - width, bbox[1] - height, bbox[2], bbox[3]]
                    outside = box[0] > bbox_new[2] or box[0] < bbox_new[0] \
                              or box[1] > bbox_new[3] or box[1] < bbox_new[1]
                    inside = box[2] < 640 and box[3] < 120
                    if outside and inside:
                        i += 1
                        self.train(box, image, 0, 0)
                        cv2.rectangle(self.viz, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.imshow("Image", self.viz)
                        cv2.waitKey(1)
            else:
                ind += 1
                continue
        print "Number of Negatives: %i" % (num), "windows"

    def train_negative(self, image, bbox, num):
        img = self.image_transformation(image)
        self.integral_img = cv2.integral(img)
        i = 0
        while i < num:
            ind = random.choice(range(self.numWindows))
            box = self.init_windows[4 * ind:4 * (ind + 1)]
            if self.mean_filter(box):
                width = box[2] - box[0]
                height = box[3] - box[1]
                bbox_new = [bbox[0] - width, bbox[1] - height, bbox[2], bbox[3]]
                outside = box[0] > bbox_new[2] or box[0] < bbox_new[0] \
                          or box[1] > bbox_new[3] or box[1] < bbox_new[1]
                inside = box[2] < 640 and box[3] < 120
                if outside and inside:
                    i += 1
                    self.train(box, image, 0, 0)
                cv2.rectangle(self.viz, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.imshow("Image", self.viz)
                cv2.waitKey(10)
        print "Number of Negatives: %i" % (num), "windows"

    def plot_hist(self, histogram, numHistogram):
        for i in range(numHistogram):
            fig = plt.figure(self.fig)
            ind = 2 ** 10
            y = histogram[i * ind:(i + 1) * ind]
            pos = np.arange(len(y))
            width = 1.0
            ax = plt.axes()
            ax.set_xticks(pos + (width / 2))
            ax.set_xticklabels(range(len(y)))
            plt.bar(pos, y, width, color='r')
            self.fig += 1

    def normalise_hist(self):
        for i in range(10):
            ind = 2 ** 10
            _sum = sum(self.posteriors_pos[i * ind:(i + 1) * ind]) - 2 ** 10
            print _sum
            for j in range(2 ** 10):
                if _sum != 0:
                    self.posteriors_pos[i * ind + j] = float(self.posteriors_pos[i * ind + j] - 1) / _sum
                else:
                    self.posteriors_pos[i * ind + j] = float(self.posteriors_pos[i * ind + j] - 1)

        for i in range(10):
            ind = 2 ** 10
            _sum = sum(self.posteriors_neg[i * ind:(i + 1) * ind]) - 2 ** 10
            print _sum
            for j in range(2 ** 10):
                if _sum != 0:
                    self.posteriors_neg[i * ind + j] = float(self.posteriors_neg[i * ind + j] - 1) / _sum
                else:
                    self.posteriors_neg[i * ind + j] = float(self.posteriors_neg[i * ind + j] - 1)

    def get_posterior(self):
        return self.posteriors_pos, self.posteriors_neg

    def set_posterior(self, pos_dist, neg_dist):
        self.posteriors_pos = pos_dist
        self.posteriors_neg = neg_dist

    def image_transformation(self, image):
        self.nccImg = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
		self.viz = self.nccImg.copy()
		img1 = cv2.GaussianBlur(self.nccImg.copy(), (3,3), 3.0)
		img2 = cv2.equalizeHist(self.nccImg.copy())
		img3 = cv2.threshold(self.nccImg.copy(), 240, 255, cv2.THRESH_BINARY)[1]
		img4 = cv2.dilate(self.nccImg.copy(), self.es, iterations=1)
		cv2.imshow("Threshold", img3)
		cv2.waitKey(10)
		cv2.imshow("GaussianBlur", img1)
		cv2.waitKey(10)
		cv2.imshow("EqualizeHist", img2)
		cv2.waitKey(10)
		cv2.imshow("Dilate", img4)
		cv2.waitKey(10)
		return img4

    def set_video(self, video):
        self.video = cv2.VideoCapture(video)
    # self.video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,32)

    def run(self, frame):
        param = dict(winSize_match=10, method=cv2.cv.CV_TM_CCOEFF_NORMED)
        self.img = self.image_transformation(frame)
        self.integral_img = cv2.integral(self.img)

        image = self.viz.copy()
        list_vehicle = []
        index = 0
        prev_ind = 0
        switch = False
        start = True

        while index < len(self.init_windows):
            if self.init_windows[index] != "switch":
                win = self.init_windows[index:index + 4]

                if self.mean_filter(win):
                    if start:
                        step = 0
                        start = False
                    else:
                        step = (index - prev_ind) / 4
                    if step < 10 and not switch:
                        featureVector = self.fern(win, -1, step)
                        prob = self.cal_prob(featureVector)
                        veh = self.isVehicle(prob)
                        if veh:
                            list_vehicle.append(win)
                        cv2.rectangle(image, (win[0],win[1]), (win[2],win[3]),(0,255,0),2)
                        cv2.imshow("Image", image)
                        cv2.waitKey(1)
                    else:
                        if switch:
                            switch = False
                        featureVector = self.fern(win, -1, 0)
                        prob = self.cal_prob(featureVector)
                        veh = self.isVehicle(prob)
                        if veh:
                            list_vehicle.append(win)
                        cv2.rectangle(image, (win[0],win[1]), (win[2],win[3]),(0,255,0),2)
                        cv2.imshow("Image", image)
                        cv2.waitKey(1)
                    prev_ind = index

            if self.init_windows[index] == "switch":
                index += 1
                switch = True
                continue
            index += 4

        if len(list_vehicle) > 0:
            vehicle = self.nearest_neighbor(list_vehicle, **param)
            for win in list_vehicle:
            img = self.img.copy()
            detection = img[win[1]:win[3], win[0]:win[2]]
            cv2.rectangle(img, (win[0],win[1]), (win[2],win[3]),(255,0,0),1)
            cv2.imshow("Detected Window", detection)
            cv2.waitKey(100)

            if vehicle:
                cv2.rectangle(frame, (vehicle[0],vehicle[1]), (vehicle[2],vehicle[3]),(255,0,0),1)
                cv2.imshow("Detected Window", frame)
                cv2.waitKey(1)
                print vehicle
                return vehicle
            else:
                cv2.imshow("Detected Window", frame)
                cv2.waitKey(1)
                print "No good correlation"
                return None
        else:
            cv2.imshow("Detected Window", frame)
            cv2.waitKey(1)
            print "No vehicle detected"
            return None
