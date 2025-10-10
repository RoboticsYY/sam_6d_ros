from setuptools import find_packages, setup

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
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs', [
            'sam_6d_ros/instance_segmentation_model/configs/download.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/run_inference.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/user', [
            'sam_6d_ros/instance_segmentation_model/configs/user/default.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/callback', [
            'sam_6d_ros/instance_segmentation_model/configs/callback/base.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/callback/checkpoint', [
            'sam_6d_ros/instance_segmentation_model/configs/callback/checkpoint/base.yaml',
        ]),        
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/callback/lr', [
            'sam_6d_ros/instance_segmentation_model/configs/callback/lr/base.yaml',
        ]),        
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/machine', [
            'sam_6d_ros/instance_segmentation_model/configs/machine/local.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/machine/slurm.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/machine/trainer', [
            'sam_6d_ros/instance_segmentation_model/configs/machine/trainer/local.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/machine/trainer/slurm.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/data', [
            'sam_6d_ros/instance_segmentation_model/configs/data/bop.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/model', [
            'sam_6d_ros/instance_segmentation_model/configs/model/ISM_fastsam.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/model/ISM_sam.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/model/descriptor_model', [
            'sam_6d_ros/instance_segmentation_model/configs/model/descriptor_model/dinov2.yaml',
        ]),
        ('lib/' + 'python3.10/site-packages/' + package_name + '/instance_segmentation_model/configs/model/segmentor_model', [
            'sam_6d_ros/instance_segmentation_model/configs/model/segmentor_model/fast_sam.yaml',
            'sam_6d_ros/instance_segmentation_model/configs/model/segmentor_model/sam.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
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
