#!/usr/bin/env python
import cv2
import numpy as np
from time import time
import scipy
from scipy import stats

feature_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=1, blockSize=1)


class Tracker():
    def __init__(self):
        self.bounding_box = None
        self.init_pts_density = 4
        self.required_pts_density = 100
        self.start_img = None
        self.target_img = None
        self.flag = True

    def initialize(self, bbox, image):
        self.start_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.start_img = cv2.equalizeHist(self.start_img)
        self.bounding_box = bbox
        self.start_pts = self.gen_point_cloud(self.bounding_box)

    def run(self, image):
        required_pts_density = float((self.bounding_box[3] - self.bounding_box[1]) *
                                     (self.bounding_box[2] - self.bounding_box[0])) / (len(self.start_pts))
        if required_pts_density > 100:
            self.start_pts = self.gen_point_cloud(self.bounding_box)
            self.flag = True
        self.target_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        # self.target_img = cv2.equalizeHist(self.target_img)
        corr, dist, valid_target_pts, valid_start_pts = self.cal_target_pts(self.start_pts)
        # for point in valid_start_pts:
        # 	cv2.circle(self.viz, (int(point[0]),int(point[1])),2,(0,255,0),-1)
        if self.flag:
            good_target_pts, good_start_pts = self.filter_pts(corr, dist, valid_target_pts, valid_start_pts)
            self.flag = False
        else:
            good_target_pts, good_start_pts = self.filter_pts2(corr, dist, valid_target_pts, valid_start_pts)
        if good_target_pts is not None:
            for point in good_target_pts:
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
            bbox = self.target_bounding_box(self.bounding_box, good_start_pts, good_target_pts)
            self.bounding_box = bbox
            cv2.rectangle(image, (self.bounding_box[0], self.bounding_box[1]),
                          (self.bounding_box[2], self.bounding_box[3]), (0, 255, 0), 1)
        else:
            self.bounding_box = None
            print "Unable to track object"
        self.start_img = self.target_img
        self.start_pts = good_target_pts
        cv2.imshow("Tracking", image)
        cv2.waitKey(100)
        return self.bounding_box, image

    def gen_point_cloud(self, box):
        pts = []  # [(x1,y1),(x2,y2)...]
        numY = int(((box[3] - box[1]) / self.init_pts_density)) + 1
        numX = int((box[2] - box[0]) / self.init_pts_density) + 1
        for i in range(numX):
            for j in range(numY):
                pts_x = box[0] + i * self.init_pts_density
                pts_y = box[1] + j * self.init_pts_density
                pts.append((pts_x, pts_y))
        return pts

    def goodFeature2Track(self):
        pts = []
        mask = np.zeros_like(self.start_img)
        mask[self.bounding_box[1]:self.bounding_box[3], self.bounding_box[0]:self.bounding_box[2]] = 255
        goodFeatures = cv2.goodFeaturesToTrack(self.start_img, mask=mask, **feature_params)
        if goodFeatures is not None:
            for x, y in np.float32(goodFeatures).reshape(-1, 2):
                pts.append((x, y))
        return pts

    def cal_target_pts(self, pts0):
        valid_target_pts = []  # initialize the target points with equal length to source
        valid_start_pts = []
        start_pts = np.asarray(pts0, dtype="float32")
        target_pts = np.asarray(pts0, dtype="float32")
        back_pts = np.asarray(pts0, dtype="float32")
        lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | \
                                                               cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                         flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        matching_param = dict(winSize_match=5, method=cv2.cv.CV_TM_CCOEFF_NORMED)

        target_pts, status_forward, _ = cv2.calcOpticalFlowPyrLK(self.start_img, self.target_img, start_pts, target_pts,
                                                                 **lk_params)

        back_pts, status_backward, _ = cv2.calcOpticalFlowPyrLK(self.target_img, self.start_img, target_pts, back_pts,
                                                                **lk_params)
        status = status_forward & status_backward
        dist_all = self.euclidean_distance(start_pts, target_pts)
        valid_corr = self.patch_matching(start_pts, target_pts, status, **matching_param)
        valid_dist = []

        for i in np.argwhere(status):
            i = i[0]
            valid_target_pts.append(tuple(target_pts[i].tolist()))
            valid_start_pts.append(tuple(start_pts[i].tolist()))
            valid_dist.append(dist_all[i])

        test = len(valid_start_pts) == len(valid_target_pts) == len(valid_dist) == len(valid_corr)
        return valid_corr, valid_dist, valid_target_pts, valid_start_pts

    def patch_matching(self, start_pts, target_pts, status, winSize_match, method):
        match_patches = []
        for i in np.argwhere(status):
            i = i[0]
            patch_start = cv2.getRectSubPix(self.start_img, (winSize_match, winSize_match),
                                            tuple(start_pts[i]))  # Use numpy array image extraction 12 times faster
            patch_target = cv2.getRectSubPix(self.target_img, (winSize_match, winSize_match), tuple(target_pts[i]))
            match_patches.append(cv2.matchTemplate(patch_start, patch_target, method)[0][0])
        return match_patches

    def euclidean_distance(self, start_pts, target_pts):
        dist = ((target_pts[:, 0] - start_pts[:, 0]) ** 2 + (target_pts[:, 1] - start_pts[:, 1]) ** 2) ** 0.5
        return np.round(dist, 1)

    def filter_pts2(self, valid_corr, valid_dist, valid_target_pts, valid_start_pts):
        good_target_points = []
        good_start_points = []
        medDist = self.median(valid_dist)
        medCorr = self.median(valid_corr)
        modDist = stats.mode(valid_dist)[0][0]
        print modDist, len(valid_dist), max(valid_dist)
        quarDist = np.percentile(valid_dist, 99)
        quarCorr = np.percentile(valid_corr, 60)
        valid_disp = []
        corr = []
        for i in range(len(valid_dist)):
            valid_disp.append(abs(valid_dist[i] - medDist))
        print "Median: ", self.median(valid_disp)

        for i in range(len(valid_corr)):
            corr.append(abs(valid_corr[i] - medCorr))
        print "Correlation: ", self.median(corr)

        # if self.median(valid_disp) > 5:
        # 	print "Median displacement Failure"
        # 	return None, None
        # if self.median(corr) > 0.01:
        # 	print "Correlation very bad. Failure"
        # 	return None, None
        median_failure = self.median(valid_disp) > 5
        correlation_failure = self.median(corr) > 0.01
        tracking_failure = median_failure and correlation_failure
        if tracking_failure:
            print "tracking failure"
            return None, None

        # for i in range(len(valid_dist)):
        # 	if abs(valid_dist[i] - modDist) <= 2:
        # 		good_target_points.append(valid_target_pts[i])
        # 		good_start_points.append(valid_start_pts[i])

        for i in range(len(valid_dist)):
            if valid_dist[i] <= quarDist:
                good_target_points.append(valid_target_pts[i])
                good_start_points.append(valid_start_pts[i])

        if len(good_target_points) <= 5:
            print 'Not enough target points'
            return None, None
        else:
            return good_target_points, good_start_points

    # def adaptive_mode_filter(self,):

    def filter_pts(self, valid_corr, valid_dist, valid_target_pts, valid_start_pts):
        good_target_points = []
        good_start_points = []
        medDist = self.median(valid_dist)
        medCorr = self.median(valid_corr)
        quarDist = np.percentile(valid_dist, 50)
        quarCorr = np.percentile(valid_corr, 60)
        valid_disp = []
        for i in range(len(valid_dist)):
            valid_disp.append(abs(valid_dist[i] - medDist))
        print "Median: ", self.median(valid_disp)
        # if self.median(valid_disp) > 20:
        # 	print "Median displacement Failure"
        # 	return None, None
        # for i in range(len(valid_dist)):
        # 	if valid_dist[i] <= medDist and valid_corr[i] >= medCorr:
        # 		good_target_points.append(valid_target_pts[i])
        # 		good_start_points.append(valid_start_pts[i])
        # return good_target_points, good_start_points
        for i in range(len(valid_dist)):
            if valid_dist[i] <= quarDist and valid_corr[i] >= quarCorr:
                good_target_points.append(valid_target_pts[i])
                good_start_points.append(valid_start_pts[i])

        if len(good_target_points) <= 5:
            print 'Not enough target points'
            return None, None
        else:
            return good_target_points, good_start_points

    def target_bounding_box(self, start_box, good_start_points, good_target_points):
        num_target_pts = len(good_target_points)
        # print num_target_pts
        width_start = start_box[2] - start_box[0]
        height_start = start_box[3] - start_box[1]
        diff_x = []
        diff_y = []
        for i in range(num_target_pts):
            diff_x.append(good_target_points[i][0] - good_start_points[i][0])
            diff_y.append(good_target_points[i][1] - good_start_points[i][1])
        dx = self.median(diff_x)
        dy = self.median(diff_y)
        # dx = np.percentile(diff_x, 40)
        # dy = np.percentile(diff_y, 40)
        # dx = self.mean(diff_x)
        # dy = self.mean(diff_y)
        diff_y = diff_x = 0

        # print dx, dy, "The shift from distance"

        scale_factor = []
        for i in range(num_target_pts):
            for j in range(i + 1, num_target_pts):
                start_img = ((good_start_points[i][0] - good_start_points[j][0]) ** 2 +
                             (good_start_points[i][1] - good_start_points[j][1]) ** 2) ** 0.5
                target_img = ((good_target_points[i][0] - good_target_points[j][0]) ** 2
                              + (good_target_points[i][1] - good_target_points[j][1]) ** 2) ** 0.5
                scale_factor.append(float(target_img) / start_img)

        scale = self.median(scale_factor)
        # scale = np.percentile(scale_factor,40)
        # scale = self.mean(scale_factor)
        # print scale, "The scale change"
        # print width_start, height_start
        scale_x = ((scale - 1) / 2) * width_start
        scale_y = ((scale - 1) / 2) * height_start

        x1_new = start_box[0] + dx - scale_x
        x2_new = start_box[2] + dx + scale_x
        y1_new = start_box[1] + dy - scale_y
        y2_new = start_box[3] + dy + scale_y

        target_box = [int(round(x1_new)), int(round(y1_new)), int(round(x2_new)), int(round(y2_new))]
        dimension = float(target_box[3] - target_box[1]) / (target_box[2] - target_box[0])
        if dimension < 1.0:
            return None
        else:
            return target_box

    def median(self, data):
        new_data = list(data)
        new_data.sort()
        if len(new_data) < 1:
            print "No Data point to calculate median"
            return None
        else:
            return new_data[len(new_data) / 2]

    def mean(self, data):
        return sum(data) / len(data)

tracker = Tracker()
bbox = [219,24,275,88]
video = cv2.VideoCapture("/home/ubuntu/temp/tld_bag/record1/testing1.mpg")
video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,88)
initialize = False
while True:
    ret,frame = video.read()
    frame = frame[80:250,0:640]
    if not initialize:
        tracker.initialize(bbox,frame)
        initialize = True
    else:
        tracker.run(frame)
