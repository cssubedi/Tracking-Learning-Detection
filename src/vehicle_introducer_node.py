import rospy
import cv2
from vehicle_detection_tracking.introducer import Introducer
from duckietown_utils.jpg import image_cv_from_jpg

class vehicle_introducer_node():
	def __init__(self):
		try:
			pos_dist = np.load("~/duckietown/catkin_ws/src/vehicle_detection_tracking/distribution/posDist.npy").tolist()
			neg_dist = np.load("~/duckietown/catkin_ws/src/vehicle_detection_tracking/distribution/negDist.npy").tolist()
			self.introduce = False
			print "Object is introduced"
		except IOError:
			print "Object is not introduced"
			self.introduce = True
		self.introducer = Introducer()
		self.sub_image = rospy.Subscriber("/autopilot/camera_node/image/compressed", CompressedImage, self.cbImage, queue_size=1)
		self.lock = mutex()

	def cbImage(self, image_msg):
		try:
			image_cv = image_cv_from_jpg(image_msg.data)
		except ValueError as e:
			print 'Could not decode image: %s' %(e)
			return
		if not self.introduce:
			return
		thread = threading.Thread(target=self.run,args=(image_cv,))
		thread.setDaemon(True)
		thread.start()

	def run(self, image_cv):
		if self.lock.testandset():
			self.introducer.run(image_cv)
			if self.introduce.getnumSamples() > 100:
				self.introduce.save_distribution()
				print "Distribution trained with 100 samples. Terminating Introducer..."
				self.lock.unlock()
				sys.exit()
			self.lock.unlock()

if __name__ == "__main__":
	rospy.init_node("vehicle_introducer_node")
	vehicle_introducer_node = vehicle_introducer_node()
	rospy.spin()