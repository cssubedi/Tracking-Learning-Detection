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
        t0 = time()
        scale_factor = 1.2
        min_window_width = 36
        min_window_height = 43
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
        feature_vector = self.fern(win, label, step)
        self.update_fern(feature_vector)

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
            feature_vector = [0] * 10
            for i in range(5, 10):
                boolean = []
                for j in range(10):
                    sec = [int(round(win[0] + j * step_w)), int(round(win[1] + i * step_h))
                        , int(round(win[0] + (j + 1) * step_w)), int(round(win[1] + (i + 1) * step_h))]
                    _sum = np.sum(self.img[sec[1]:sec[3], sec[0]:sec[2]])
                    avg = int(round(float(_sum) / (step_h * step_w), 2))
                    if avg < 127:
                        # temp_img[sec[1]:sec[3], sec[0]:sec[2]] = 0
                        boolean.append(False)
                    else:
                        # temp_img[sec[1]:sec[3], sec[0]:sec[2]] = 255
                        boolean.append(True)
                feature_vector[i] = self.bool2binary(boolean)
            # cv2.imshow("Features", temp_img[win[1]:win[3], win[0]:win[2]])
            # cv2.waitKey(100)
            feature_vector.append(label)
            # print feature_vector
            self.prev_feature_vector = feature_vector
            # print "New feature", feature_vector
            # cv2.imshow("Threshold", self.img)
            # cv2.waitKey(100)
            return feature_vector
        else:
            height = win[3] - win[1]
            width = win[2] - win[0]
            step_w = width / 10.0
            step_h = height / 10.0
            feature_vector = [0] * 10
            for i in range(5, 10):
                if i < 10 - step:
                    # print "Fast feature_vector", step
                    feature_vector[i] = self.prev_feature_vector[i + step]
                    continue
                # print "Cal"
                boolean = []
                for j in range(10):
                    sec = [int(round(win[0] + j * step_w)), int(round(win[1] + i * step_h)) \
                        , int(round(win[0] + (j + 1) * step_w)), int(round(win[1] + (i + 1) * step_h))]
                    _sum = np.sum(self.img[sec[1]:sec[3], sec[0]:sec[2]])
                    avg = int(round(float(_sum) / (step_h * step_w), 2))
                    if avg < 127:
                        # temp_img[sec[1]:sec[3], sec[0]:sec[2]] = 0
                        boolean.append(False)
                    else:
                        # temp_img[sec[1]:sec[3], sec[0]:sec[2]] = 255
                        boolean.append(True)
                feature_vector[i] = self.bool2binary(boolean)
            # cv2.imshow("Features", temp_img[win[1]:win[3], win[0]:win[2]])
            # cv2.waitKey(100)
            feature_vector.append(label)
            self.prev_feature_vector = feature_vector
            # print feature_vector
            # cv2.imshow("Threshold", self.img)
            # cv2.waitKey(100)
            return feature_vector

    def update_fern(self, feature_vector):
        if feature_vector[-1] == 1:
            for classifier in range(5, 10):
                ind = classifier * 2 ** 10
                self.posteriors_pos[ind + int(featureVector[classifier], 2)] += 1
            # print int(featureVector[classifier],2)
            # print self.posteriors_pos[ind+int(featureVector[classifier],2)]
        if feature_vector[-1] == 0:
            for classifier in range(5, 10):
                ind = classifier * 2 ** 10
                self.posteriors_neg[ind + int(feature_vector[classifier], 2)] += 1

    def cal_prob(self, feature_vector):
        list_prob = []
        for classifier in range(5, 10):
            ind = classifier * 2 ** 10
            pos = self.posteriors_pos[ind + int(feature_vector[classifier], 2)]
            neg = self.posteriors_neg[ind + int(feature_vector[classifier], 2)]
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
        else:
            return False

    def nearest_neighbor(self, list_vehicle, method):
        _, _, files = os.walk("/home/ubuntu/TLD/ObjectModel").next()
        num_template = len(files)
        correlations = []
        for veh in list_vehicle:
            image = self.nccImg[veh[1]:veh[3], veh[0]:veh[2]]
            shape = image.shape
            ncc = []
            for i in range(1, num_template + 1):
                template = cv2.imread("/home/ubuntu/temp/tld_bag/template{:>03}.jpg".format(i), 0)
                template = cv2.resize(template, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                matching = cv2.matchTemplate(image, template, method)[0][0]
                ncc.append(matching)
            max_ncc = max(ncc)
            correlations.append(max_ncc)
        max_correlation = max(correlations)
        print max_correlation
        if max_correlation > 0.6:
            return list_vehicle[correlations.index(max_correlation)]
        else:
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
                    outside = box[0] > bbox_new[2] or box[0] < bbox_new[0] or box[1] \
                              > bbox_new[3] or box[1] < bbox_new[1]
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
                outside = box[0] > bbox_new[2] or box[0] < bbox_new[0] or \
                          box[1] > bbox_new[3] or box[1] < bbox_new[1]
                inside = box[2] < 640 and box[3] < 120
                if outside and inside:
                    i += 1
                    self.train(box, image, 0, 0)
        print "Number of Negatives: %i" % (num), "windows"

    def plot_hist(self, histogram, numHistogram):
        for i in range(numHistogram):
            ind = 2 ** 10
            y = histogram[i * ind:(i + 1) * ind]
            print len(y)
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
        # img = cv2.GaussianBlur(img, (3,3), 3.0)
        # img = cv2.equalizeHist(self.nccImg.copy())
        img = cv2.threshold(self.nccImg.copy(), 230, 255, cv2.THRESH_BINARY)[1]
        # img = cv2.dilate(img, self.es, iterations=1)
        # cv2.imshow("Threshold", img)
        # cv2.waitKey(1)
        return img

    def set_video(self, video):
        self.video = cv2.VideoCapture(video)

    def run(self, frame):
        param = dict(winSize_match=10, method=cv2.cv.CV_TM_CCOEFF_NORMED)
        self.img = self.image_transformation(frame)
        self.integral_img = cv2.integral(self.img)
        list_vehicle = []
        index = 0
        prev_ind = 0
        switch = False
        while index < len(self.init_windows):
            if self.init_windows[index] != "switch":
                win = self.init_windows[index:index + 4]
                if self.mean_filter(win):
                    step = (index - prev_ind) / 4
                    if step < 10 and not switch:
                        featureVector = self.fern(win, -1, step)
                        prob = self.cal_prob(featureVector)
                        veh = self.isVehicle(prob)
                        if veh:
                            list_vehicle.append(win)
                        # cv2.rectangle(image, (win[0],win[1]), (win[2],win[3]),(0,255,0),2)
                        # cv2.imshow("Image", image)
                        # cv2.waitKey(1)
                    else:
                        if switch:
                            switch = False
                        featureVector = self.fern(win, -1, 0)
                        prob = self.cal_prob(featureVector)
                        veh = self.isVehicle(prob)
                        if veh:
                            list_vehicle.append(win)
                        # cv2.rectangle(image, (win[0],win[1]), (win[2],win[3]),(0,255,0),2)
                        # cv2.imshow("Image", image)
                        # cv2.waitKey(1)
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
                cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
                cv2.imshow("Detected Window", detection)
                cv2.waitKey(100)
            if vehicle:
                cv2.rectangle(frame, (vehicle[0], vehicle[1]), (vehicle[2], vehicle[3]), (255, 0, 0), 1)
                cv2.imshow("Detected Window", frame)
                cv2.waitKey(1)
                return vehicle
            else:
                cv2.waitKey(1)
                print "No good correlation"
                return None
        else:
            cv2.imshow("Detected Window", frame)
            cv2.waitKey(1)
            print "No vehicle detected"
            return None


# Initial Testing
detector = Detector()
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0068.jpg")
img = img[80:250, 0:640]
win = [168, 38, 200, 78]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 1 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0069.jpg")
img = img[80:250, 0:640]
win = [200, 35, 234, 77]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 2 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0070.jpg")
img = img[80:250, 0:640]
win = [210, 34, 246, 77]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 3 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0071.jpg")
img = img[80:250, 0:640]
win = [231, 30, 268, 76]
img_temp = img.copy()
img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
template = img_temp[win[1]:win[3], win[0]:win[2]]
cv2.imwrite("/home/ubuntu/temp/tld_bag/template001.jpg", template)
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 4 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0075.jpg")
img = img[80:250, 0:640]
win = [229, 31, 269, 80]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 5 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0080.jpg")
img = img[80:250, 0:640]
win = [220, 24, 266, 80]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 6 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0085.jpg")
img = img[80:250, 0:640]
win = [210, 24, 264, 90]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 7 dataset"
img = cv2.imread("/home/ubuntu/temp/tld_bag/record1/frame0089.jpg")
img = img[80:250, 0:640]
win = [221, 23, 273, 89]
detector.train(win, img.copy(), 1, 0)
cv2.rectangle(img, (win[0], win[1]), (win[2], win[3]), (255, 0, 0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
print "Trained with 8 dataset"
detector.generate_negative(img, win)
detector.normalise_hist()
pos, neg = detector.get_posterior()

detector.plot_hist(pos, 5)
detector.plot_hist(neg, 5)
plt.show('hold')

video = cv2.VideoCapture("/home/ubuntu/temp/tld_bag/record1/testing1.mpg")
video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 60)
while True:
    ret, frame = video.read()
    frame = frame[80:250, 0:640]
    detector.run(frame)
