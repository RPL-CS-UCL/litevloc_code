import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import argparse

def publish_image(image_path):
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('/goal_image', Image, queue_size=10)

    bridge = CvBridge()
    image = cv2.imread(image_path)
    image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")

    r = rospy.Rate(1)
    while not rospy.is_shutdown():
        image_msg.header.stamp = rospy.Time.now()
        image_pub.publish(image_msg)
        print('Publish image to /goal_image')
        r.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='Path to the image file')
    args = parser.parse_args()

    image_path = args.image_path
    publish_image(image_path)