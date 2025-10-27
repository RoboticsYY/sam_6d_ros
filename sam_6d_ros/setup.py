
import sys
from setuptools import find_packages, setup

python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

package_name = 'sam_6d_ros'

setup(
    name=package_name,
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['sam_6d_ros/params.yaml']),
        # Install all yaml files under instance_segmentation_model/configs and subfolders
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs', [
            'sam_6d_ros/instance_segmentation_model/configs/download.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/run_inference.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/user', [
            'sam_6d_ros/instance_segmentation_model/configs/user/default.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/callback', [
            'sam_6d_ros/instance_segmentation_model/configs/callback/base.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/callback/checkpoint', [
            'sam_6d_ros/instance_segmentation_model/configs/callback/checkpoint/base.yaml',
        ]),        
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/callback/lr', [
            'sam_6d_ros/instance_segmentation_model/configs/callback/lr/base.yaml',
        ]),        
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/machine', [
            'sam_6d_ros/instance_segmentation_model/configs/machine/local.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/machine/slurm.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/machine/trainer', [
            'sam_6d_ros/instance_segmentation_model/configs/machine/trainer/local.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/machine/trainer/slurm.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/data', [
            'sam_6d_ros/instance_segmentation_model/configs/data/bop.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/model', [
            'sam_6d_ros/instance_segmentation_model/configs/model/ISM_fastsam.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/model/ISM_sam.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/model/descriptor_model', [
            'sam_6d_ros/instance_segmentation_model/configs/model/descriptor_model/dinov2.yaml',
        ]),
        ('lib/' + python_version + '/site-packages/' + package_name + '/instance_segmentation_model/configs/model/segmentor_model', [
            'sam_6d_ros/instance_segmentation_model/configs/model/segmentor_model/fast_sam.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/model/segmentor_model/sam.yaml',
        ]),
        ('share/' + package_name + '/pose_estimation_model/config', [
            'sam_6d_ros/pose_estimation_model/config/ov_gpu_base.yaml',
        ]),
        ('share/' + package_name + '/pose_estimation_model/config', [
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/ov_gpu_custom_op.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/ball_query_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/ball_query.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_debug_node_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_debug_node.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_det_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_det.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_svd_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_svd.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_svd_u.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/custom_svd_v.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/furthest_point_sampling_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/furthest_point_sampling.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/gather_operation_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/gather_operation.cl',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/grouping_operation_cl.xml',
            'sam_6d_ros/pose_estimation_model/model/ov_pointnet2_op/grouping_operation.cl',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    include_package_data=True,
    maintainer='dev',
    maintainer_email='yu.yan@intel.com',
    description='ROS2 package for 6D pose estimation with SAM-6D',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation = sam_6d_ros.pose_estimation:main',
        ],
    },
)
