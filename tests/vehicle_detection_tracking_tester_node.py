#!/usr/bin/env python

import rospy
import cv2
import numpy as np 
from detector_tester import Detector
from tracker_tester import Tracker
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.msg import VehicleDetected, VehicleBoundingBox
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int32
from mutex import mutex
import threading
from duckietown_utils.jpg import image_cv_from_jpg


class TLD():
	def __init__(self):
		pos_dist = np.load("/home/ubuntu/TLD/posDist.npy").tolist()
		neg_dist = np.load("/home/ubuntu/TLD/negDist.npy").tolist()
		self.active = True
		self.bridge = CvBridge()
		self.Detector = Detector()
		self.Detector.set_posterior(pos_dist,neg_dist)
		self.Tracker = Tracker()
		self.tracking = False
		self.sub_image = rospy.Subscriber("/autopilot/camera_node/image/compressed", CompressedImage, self.cbImage, queue_size=1)
		self.pub_image = rospy.Publisher("~image_with_detection", Image, queue_size=1)
		self.pub_vehicle_detected = rospy.Publisher("~vehicle_detected", VehicleDetected, queue_size=1)
		self.pub_vehicle_bbox = rospy.Publisher("~vehicle_bounding_box", VehicleBoundingBox, queue_size=1)
		self.lock = mutex()
		bbox = []
		self.tracking = True

	def cbImage(self, image_msg):
		try:
			image_cv = image_cv_from_jpg(image_msg.data)
		except ValueError as e:
			print 'Could not decode image: %s' %(e)
			return
		if not self.active:
			return
		thread = threading.Thread(target=self.run_tracking,args=(image_cv,bbox))
		thread.setDaemon(True)
		thread.start()

	def run(self, image_cv):
		if self.lock.testandset():
			vehicle_bounding_box_msg = VehicleBoundingBox()
			vehicle_detected_msg = VehicleDetected()
			image_cv = image_cv[80:200,0:640]
			if not self.tracking:
				veh = self.Detector.run(image_cv)
				if veh == None:
					vehicle_detected_msg = False
					self.pub_vehicle_detected.publish(vehicle_detected_msg)
				else:
					vehicle_detected_msg = True
					vehicle_bounding_box_msg.data = veh
					cv2.rectangle(image_cv, (veh[0],veh[1]), (veh[2],veh[3]),(255,0,0),1)
					image_msg = self.bridge.cv2_to_imgmsg(image_cv,"bgr8")
					self.pub_vehicle_bbox.publish(vehicle_bounding_box_msg)
					self.pub_vehicle_detected.publish(vehicle_detected_msg)
					self.pub_image.publish(image_msg)
					self.Tracker.initialize(veh,image_cv)
					self.tracking = True
			else:
				veh = self.Tracker.run(image_cv)
				if veh == None:
					rospy.loginfo("Tracking Failed")
					self.tracking = False
					vehicle_detected_msg = False
					self.pub_vehicle_detected.publish(vehicle_detected_msg)
				else:
					vehicle_detected_msg = True
					vehicle_bounding_box_msg.data = veh
					cv2.rectangle(image_cv, (veh[0],veh[1]), (veh[2],veh[3]),(255,0,0),1)
					image_msg = self.bridge.cv2_to_imgmsg(image_cv,"bgr8")
					self.pub_vehicle_bbox.publish(vehicle_bounding_box_msg)
					self.pub_vehicle_detected.publish(vehicle_detected_msg)
					self.pub_image.publish(image_msg)

			self.lock.unlock()

	def run_tracking(self,image_cv, bbox):
		if self.lock.testandset():
			vehicle_bounding_box_msg = VehicleBoundingBox()
			vehicle_detected_msg = VehicleDetected()
			image_cv = image_cv[80:200,0:640]
			if not self.tracking:
				self.Tracker.initialize(bbox,image_cv)
				self.tracking = True
			else:
				veh, image = self.Tracker.run(image_cv)
				if veh == None:
					rospy.loginfo("Tracking Failed")
					self.tracking = False
					vehicle_detected_msg = False
					image_msg = self.bridge.cv2_to_imgmsg(image,"bgr8")
					self.pub_vehicle_detected.publish(vehicle_detected_msg)
					self.pub_image.publish(image_msg)
				else:
					vehicle_detected_msg = True
					vehicle_bounding_box_msg.data = veh
					# cv2.rectangle(image_cv, (veh[0],veh[1]), (veh[2],veh[3]),(255,0,0),1)
					image_msg = self.bridge.cv2_to_imgmsg(image,"bgr8")
					self.pub_vehicle_bbox.publish(vehicle_bounding_box_msg)
					self.pub_vehicle_detected.publish(vehicle_detected_msg)
					self.pub_image.publish(image_msg)
			self.lock.unlock()

if __name__ == "__main__":
	rospy.init_node("vehicle_detection_tracking_node")
	vehicle_detection_tracking_node = TLD()
	rospy.spin()