#!/usr/bin/env python
import cv2
import numpy as np 
from detector_tester import Detector 
from tracker_tester import Tracker
import matplotlib.pyplot as plt 

class Introducer():
	def __init__(self,template):
		self.template = cv2.imread(template,0)
		self.numWindows = 0
		self.init_windows = []
		self.sliding_windows()
		self.Tracker = Tracker()
		self.Detector = Detector()

	def sliding_windows(self):
		scale_factor = 1.05
		min_window_width = 30
		min_window_height = 36
		img_height = 120
		img_width = 640
		win_size_w = min_window_width
		win_size_h = min_window_height
		shift_w = int(round(0.1*win_size_w))
		shift_h = int(round(0.1*win_size_h))
		index = 0
		while win_size_w < img_width/4 and win_size_h < img_height:
			for x in xrange(0, img_width-win_size_w, shift_w):
				for y in xrange(0, img_height-win_size_h, shift_h):
					self.init_windows[index:index+4] = [x,y,x+win_size_w, y+win_size_h]
					index +=4
					self.numWindows += 1
			win_size_h=int(round(scale_factor*win_size_h))
			win_size_w=int(round(scale_factor*win_size_w))
			shift_w = int(round(0.1*win_size_w))
			shift_h = int(round(0.1*win_size_h))
		print self.numWindows
		return self.init_windows

	def patch_matching(self, image):
		correlation = 0
		vehicle = None
		ind = 0
		while ind < len(self.init_windows):
			win = self.init_windows[ind:ind+4]
			# print "Window: ", win
			ind += 4
			patch = image[win[1]:win[3],win[0]:win[2]]
			template = cv2.resize(self.template, (win[2]-win[0],win[3]-win[1]),interpolation=cv2.INTER_LINEAR)
			# print patch.shape, template.shape
			matching = cv2.matchTemplate(patch, template, method=cv2.cv.CV_TM_CCOEFF_NORMED)[0][0]
			# print "NCC: ", matching
			if matching > correlation:
				# print "Updated corr"
				correlation = matching
				vehicle = win 
		return vehicle, correlation

	def run(self,frame):
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# self.show_template()
		vehicle,correlation = self.patch_matching(image)
		self.show_image(image,vehicle)
		print correlation
		if correlation > 0.7:
			self.Detector.train(vehicle,frame,1,0)
			self.Detector.generate_negative(frame,vehicle,10)
		else:
			pass

	def show_image(self,image,veh):
		cv2.rectangle(image,(veh[0],veh[1]),(veh[2],veh[3]),(0,255,0),1)
		cv2.imshow("Introducer", image)
		cv2.waitKey(10)

	def show_template(self):
		cv2.imshow("Template", self.template)
		cv2.waitKey(10)

	def save_distribution(self):
		self.Detector.normalise_hist()
		pos,neg = self.Detector.get_posterior()
		np.save("/home/ubuntu/temp/tld_bag/posDist",pos)
		np.save("/home/ubuntu/temp/tld_bag/negDist",neg)
		self.Detector.plot_hist(pos,10)
		self.Detector.plot_hist(neg,10)
		plt.show("hold")


img = cv2.imread("/home/ubuntu/tld_bag/template001.jpg")
cv2.imshow("Template", img)
cv2.waitKey(100)

introducer = Introducer("/home/ubuntu/temp/tld_bag/template001.jpg")

video = cv2.VideoCapture("/home/ubuntu/temp/tld_bag/record1/testing1.mpg")
video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,50)
while True:
	ret, frame = video.read()
	if ret:
		frame = frame[80:250,0:640]
		introducer.run(frame)
	else:
		introducer.save_distribution()

pos_dist = np.load("/home/ubuntu/temp/tld_bag/posDist.npy").tolist()
neg_dist = np.load("/home/ubuntu/temp/tld_bag/negDist.npy").tolist()
