#! /opt/conda/envs/topo_loc/bin/python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf
import numpy as np
import open3d as o3d
from message_filters import ApproximateTimeSynchronizer, Subscriber
import time
import copy

class DepthRegistration:
    def __init__(self):
        self.last_depth_cloud = None
        self.T_w_cam = np.eye(4)
        self.radius = 0.1
        
    def initialize_ros(self):
        self.depth_sub = Subscriber("/depth/image", Image)
        self.info_sub = Subscriber("/depth/camera_info", CameraInfo)
        ats = ApproximateTimeSynchronizer([self.depth_sub, self.info_sub], queue_size=100, slop=0.1)
        ats.registerCallback(self.depth_image_callback)

        self.odom_pub = rospy.Publisher("/depth_reg/odometry", Odometry, queue_size=10)
        self.path_pub = rospy.Publisher("/depth_reg/path", Path, queue_size=10)
        self.path = Path()

        self.frame_id_map, self.frame_id_sensor = 'map', 'camera'

    def depth_image_callback(self, depth_msg, info_msg):
        bridge = CvBridge()
        depth_img = bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
        if depth_msg.encoding == "mono16":
            depth_img *= 0.001
        if depth_msg.encoding == "mono8":
            depth_img *= 0.039
        K = np.array(info_msg.K).reshape((3, 3))
        image_shape = (info_msg.width, info_msg.height)

        depth_points = self.depth_image_to_point_cloud(depth_img, K, image_shape)
        depth_cloud_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth_points))
        depth_cloud = depth_cloud_raw.voxel_down_sample(self.radius)

        # Register the current depth points with the last depth points
        start_time = time.time()
        if self.last_depth_cloud is not None:
            self.T_last_curr = self.estimate_pose_icp(depth_cloud, self.last_depth_cloud, np.eye(4))
            # self.draw_registration_result(depth_cloud, self.last_depth_cloud, self.T_last_curr)
        else:
            self.T_last_curr = np.eye(4)
        print(f"Time taken for ICP: {time.time() - start_time:.3f}s")

        self.T_w_cam = self.T_w_cam @ self.T_last_curr
        self.last_depth_cloud = depth_cloud

        # Publish the current transformation as odometry
        self.frame_id_sensor = depth_msg.header.frame_id
        header = Header(stamp=depth_msg.header.stamp, frame_id=self.frame_id_map)
        self.publish_odometry(self.T_w_cam, header)

        ##### DEBUG(gogojjh):
        print(self.T_last_curr[:3, 3].reshape(1, 3), self.T_w_cam[:3, 3].reshape(1, 3))

    def draw_registration_result(self, source, target, transformation):
        radius = 0.2
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        source_down.paint_uniform_color([0.8, 0, 0])
        target_down.paint_uniform_color([0, 0, 0])
        source_down.transform(transformation)
        o3d.visualization.draw_geometries([source_down, target_down], 
                            zoom=1.0,
                            front=[-0.2458, -0.8088, 0.5342],
                            lookat=[1.7745, 2.2305, 0.9787],
                            up=[0.3109, -0.5878, -0.7468])

    def depth_image_to_point_cloud(self, depth_image, intrinsics, image_shape):
        """
        Convert a depth image to a point cloud.

        Parameters:
        depth_image (numpy.ndarray): The depth image.
        intrinsics (numpy.ndarray): The camera intrinsic matrix.

        Returns:
        numpy.ndarray: The point cloud as an (N, 3) array.
        """
        w, h = image_shape
        i, j = np.indices((h, w))
        z = depth_image
        x = (j - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (i - intrinsics[1, 2]) * z / intrinsics[1, 1]
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        return points

    def estimate_pose_icp(self, source, target, current_transformation):
        # Point-to-plane ICP
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))       
        threshold = 0.5
        loss = o3d.pipelines.registration.TukeyLoss(k=0.05)
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, threshold, current_transformation, 
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                    relative_rmse=1e-6,
                                    max_iteration=100))

        # Point-to-point ICP
        # threshold = 0.5
        # result_icp = o3d.pipelines.registration.registration_icp(
        #     source, target, threshold, current_transformation, 
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-2,
        #                             relative_rmse=1e-2,
        #                             max_iteration=20))
        # print('inlier_rmse: {}m'.format(result_icp.inlier_rmse))
        # print('current_transformation:\n{}'.format(current_transformation))
        return result_icp.transformation

    def publish_odometry(self, transformation, header):
        # Create Odometry message
        odom = Odometry()
        odom.header = header
        odom.child_frame_id = self.frame_id_sensor
        odom.pose.pose.position.x = transformation[0, 3]
        odom.pose.pose.position.y = transformation[1, 3]
        odom.pose.pose.position.z = transformation[2, 3]
        quat = tf.transformations.quaternion_from_matrix(transformation)
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        self.odom_pub.publish(odom)

        self.path.header = header
        pose_stamped = PoseStamped()
        pose_stamped.header
        pose_stamped.pose = odom.pose.pose
        self.path.poses.append(pose_stamped)
        self.path_pub.publish(self.path)

if __name__ == '__main__':
    rospy.init_node('depth_registration', anonymous=True)
    depth_registration = DepthRegistration()
    depth_registration.initialize_ros()
    depth_registration.frame_id_map = rospy.get_param('~frame_id_map', 'map')
    rospy.spin()