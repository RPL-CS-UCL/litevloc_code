#! /usr/bin/env python

import os
import argparse

import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PublishGoalImage():
    def __init__(self, args):
        self.args = args

        self.goal_img_start_idx = 0
        self.global_planner_status = 0

        # ROS Subscriber
        self.planner_status_sub = rospy.Subscriber('/global_planner/status', Int16, self.planner_status_callback, queue_size=1)

        # ROS Publisher
        self.image_pub = rospy.Publisher('/goal_image', Image, queue_size=10)

    def planner_status_callback(self, msg):
        # if msg.data == 0: # not start planning, need to repeatedly publish the same goal image
        #     pass
        # if msg.data == 1: # in planning, no need to publish goal image
        #     return
        if msg.data == 2:  # Reach the original goal, need to publish a new goal
            self.goal_img_start_idx += 1

        bridge = CvBridge()
        goal_img_path = f'{self.args.map_path}/goal_images/goal_img_{self.goal_img_start_idx}.jpg'
        if os.path.exists(goal_img_path) == False:
            rospy.loginfo(f'{goal_img_path} does not exist')
            rospy.loginfo(f'Switch to goal_img_0')
            self.goal_img_start_idx = 0

            return
        
        goal_img = cv2.imread(goal_img_path)
        image_msg = bridge.cv2_to_imgmsg(goal_img, encoding="bgr8")
        image_msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(image_msg)
        rospy.loginfo(f'Publish {goal_img_path} to /goal_image')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', help='Path to the map file')
    args, unknown = parser.parse_known_args()

    rospy.init_node('image_publisher', anonymous=True)
    publish_goal_image = PublishGoalImage(args)
    rospy.loginfo('Start to publish goal image, waiting for the global planner status.')
    rospy.spin()