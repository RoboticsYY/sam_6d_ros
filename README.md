# sam_6d_ros
ROS2 package for the pose estimation using SAM-6D

## 1. Install OpenVINO from source on Linux

> **Note:** For instructions on building OpenVINO from source on Linux, see the [OpenVINO build guide](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md).

```bash
cd ~

git clone -b ov_sam6d_mix https://github.com/18582088138/xkd-openvino

cd xkd-openvino

git submodule update --init --recursive

# You must create a Python 3.10 virtual environment for building SAM-6D, as ROS2 Humble only supports Python 3.10.
python3.10 -m venv ov_build 

source ov_build/bin/activate

mkdir -p build && cd build

# Fix for dependencies install
pip install -r ./src/bindings/python/wheel/requirements-dev.txt
sudo apt install patchelf

cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON -DCMAKE_INSTALL_PREFIX=../ov_dist_mix  -DENABLE_WHEEL=ON -DENABLE_SYSTEM_TBB=OFF  -DENABLE_DEBUG_CAPS=ON -DENABLE_GPU_DEBUG_CAPS=ON  ..

make -j$(nproc)
make install

# Exit from ov_build virtual environment
deactivate 

# Setup ov environment variables anytine when OpenVINO is needed
source ../ov_dist_mix/setupvars.sh
```

## 2. Create and build SAM-6D ROS2 workspace
### 2.1 Install ROS2 Humble

Install ROS2 Humble: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

### 2.2 Create ROS2 workspace
```bash
mkdir -p ~/ws_sam6d/src

cd ~/ws_sam6d

mkdir checkpoints && touch checkpoints/COLCON_IGNORE

cd ~/ws_sam6d/src

git clone https://github.com/intel-sandbox/sam_6d_ros.git
```

Move the converted OpenVINO models to the 'checkpoints' directory, the final directory structure should like this:
```bash
# /home/intel/ws_sam6d/checkpoints/
├── COLCON_IGNORE
├── descriptor_dinov2_model_infer.bin
├── descriptor_dinov2_model_infer.onnx
├── descriptor_dinov2_model_infer.xml
├── det
│   ├── yolo.bin
│   └── yolo.xml
├── fastsam_yolo_v8_predictor.bin
├── fastsam_yolo_v8_predictor.onnx
├── fastsam_yolo_v8_predictor.xml
├── onnx_fe_model.onnx
├── onnx_pem_sub1_model.onnx
├── onnx_pem_sub2_model.onnx
├── onnx_pem_sub3_model.onnx
├── onnx_pem_sub4_model.onnx
├── ov_fe_model.bin
├── ov_fe_model.xml
├── ov_pem_sub1_model.bin
├── ov_pem_sub1_model.xml
├── ov_pem_sub2_model.bin
├── ov_pem_sub2_model.xml
├── ov_pem_sub3_model.bin
├── ov_pem_sub3_model.xml
├── ov_pem_sub4_model.bin
├── ov_pem_sub4_model.xml
├── sam_image_encoder.bin
├── sam_image_encoder.xml
├── sam_mask_predictor.bin
└── sam_mask_predictor.xml
```

Move pointnet2 custom OpenVINO operators into the system path like this:
```bash
# /usr/local/lib/libopenvino_operation_extension.so
```

> **Note:** For OpenVINO model convert and PointNet2 operators generation, see the [SAM-6D OpenVINO guide](https://github.com/intel-sandbox/frameworks.industrial.motion-control.sam_6d_openvino).

### 2.3 Create Python virtual environment
```bash
cd ~/ws_sam6d

python3.10 -m venv ov_sam6d && touch ov_sam6d/COLCON_IGNORE

source ov_sam6d/bin/activate

pip install -r src/sam_6d_ros/sam_6d_ros/requirements.txt
```

### 2.4 Fix Python virtual environment files

#### 2.4.1 Replace ultralytics predictor by OpenVINO FastSAM version 
```bash
cd ~/ws_sam6d

source ov_sam6d/bin/activate

cd ./src/sam_6d_ros/sam_6d_ros/sam_6d_ros/instance_segmentation_model/model

MODEL_PATH="/home/intel/ws_sam6d/checkpoints/fastsam_yolo_v8_predictor.xml"

sed "s|fastsam_yolo_v8_predictor_model_path = \"\"|fastsam_yolo_v8_predictor_model_path = \"$MODEL_PATH\"|" ov_predictor.py > tmp.py

ULTRALYTICS_PATH=$(python -c "import ultralytics, os; print(os.path.dirname(ultralytics.__file__))")

PREDICTOR_FILE_PATH="$ULTRALYTICS_PATH/yolo/engine/predictor.py"

cp tmp.py "$PREDICTOR_FILE_PATH"

rm tmp.py

echo "[Logging] ultralytics predictor.py replace success $PREDICTOR_FILE_PATH"
```

#### 2.4.2 Install PointNet2
```bash
cd ~/ws_sam6d

source ov_sam6d/bin/activate

source ~/xkd-openvino/ov_dist_mix/setupvars.sh

src/sam_6d_ros/sam_6d_ros/sam_6d_ros/pose_estimation_model/model/pointnet2/

python setup.py install
```

#### 2.4.3 Fix Pytorch safe load issue
```bash
cd ~/ws_sam6d

source ov_sam6d/bin/activate

bash src/sam_6d_ros/sam_6d_ros/fix_torch_safe_load.sh
```

### 2.5 Build ROS2 workspace
```bash
cd ~/ws_sam6d

source ov_sam6d/bin/activate

source ~/xkd-openvino/ov_dist_mix/setupvars.sh

source /opt/ros/humble/setup.bash

colcon build

source install/setup.bash
```

### 2.6 Run SAM-6D ROS2 service

#### 2.6.1 Install and launch RealSense ROS2 according to: https://github.com/IntelRealSense/realsense-ros 

#### 2.6.2 Prepare SAM-6D template data
```bash
mkdir ~/tmp

# Generate templates data according to: https://github.com/intel-sandbox/frameworks.industrial.motion-control.sam_6d_openvino/blob/5fca9ae04c5984cc38d2679bd4bd9c3ec6cd60d8/SAM-6D/ov_demo.sh#L2-L14

# Move CAD and templates data to the '~/tmp' folder like this:
└── sam_6d_ros
    ├── cad
    │   └── logistics-box.ply
    └── templates
        ├── mask_0.png
        ├── mask_10.png
        ├── mask_11.png
        ├── mask_12.png
        ├── mask_13.png
        ├── mask_14.png
        ├── mask_15.png
        ├── mask_16.png
        ├── mask_17.png
        ├── mask_18.png
        ├── mask_19.png
        ├── mask_1.png
        ├── mask_20.png
        ├── mask_21.png
        ├── mask_22.png
        ├── mask_23.png
        ├── mask_24.png
        ├── mask_25.png
        ├── mask_26.png
        ├── mask_27.png
        ├── mask_28.png
        ├── mask_29.png
        ├── mask_2.png
        ├── mask_30.png
        ├── mask_31.png
        ├── mask_32.png
        ├── mask_33.png
        ├── mask_34.png
        ├── mask_35.png
        ├── mask_36.png
        ├── mask_37.png
        ├── mask_38.png
        ├── mask_39.png
        ├── mask_3.png
        ├── mask_40.png
        ├── mask_41.png
        ├── mask_4.png
        ├── mask_5.png
        ├── mask_6.png
        ├── mask_7.png
        ├── mask_8.png
        ├── mask_9.png
        ├── rgb_0.png
        ├── rgb_10.png
        ├── rgb_11.png
        ├── rgb_12.png
        ├── rgb_13.png
        ├── rgb_14.png
        ├── rgb_15.png
        ├── rgb_16.png
        ├── rgb_17.png
        ├── rgb_18.png
        ├── rgb_19.png
        ├── rgb_1.png
        ├── rgb_20.png
        ├── rgb_21.png
        ├── rgb_22.png
        ├── rgb_23.png
        ├── rgb_24.png
        ├── rgb_25.png
        ├── rgb_26.png
        ├── rgb_27.png
        ├── rgb_28.png
        ├── rgb_29.png
        ├── rgb_2.png
        ├── rgb_30.png
        ├── rgb_31.png
        ├── rgb_32.png
        ├── rgb_33.png
        ├── rgb_34.png
        ├── rgb_35.png
        ├── rgb_36.png
        ├── rgb_37.png
        ├── rgb_38.png
        ├── rgb_39.png
        ├── rgb_3.png
        ├── rgb_40.png
        ├── rgb_41.png
        ├── rgb_4.png
        ├── rgb_5.png
        ├── rgb_6.png
        ├── rgb_7.png
        ├── rgb_8.png
        ├── rgb_9.png
        ├── xyz_0.npy
        ├── xyz_10.npy
        ├── xyz_11.npy
        ├── xyz_12.npy
        ├── xyz_13.npy
        ├── xyz_14.npy
        ├── xyz_15.npy
        ├── xyz_16.npy
        ├── xyz_17.npy
        ├── xyz_18.npy
        ├── xyz_19.npy
        ├── xyz_1.npy
        ├── xyz_20.npy
        ├── xyz_21.npy
        ├── xyz_22.npy
        ├── xyz_23.npy
        ├── xyz_24.npy
        ├── xyz_25.npy
        ├── xyz_26.npy
        ├── xyz_27.npy
        ├── xyz_28.npy
        ├── xyz_29.npy
        ├── xyz_2.npy
        ├── xyz_30.npy
        ├── xyz_31.npy
        ├── xyz_32.npy
        ├── xyz_33.npy
        ├── xyz_34.npy
        ├── xyz_35.npy
        ├── xyz_36.npy
        ├── xyz_37.npy
        ├── xyz_38.npy
        ├── xyz_39.npy
        ├── xyz_3.npy
        ├── xyz_40.npy
        ├── xyz_41.npy
        ├── xyz_4.npy
        ├── xyz_5.npy
        ├── xyz_6.npy
        ├── xyz_7.npy
        ├── xyz_8.npy
        └── xyz_9.npy
```

#### 2.6.3 Run SAM-6D ROS2 node in a terminal
```bash
cd ~/ws_sam6d

source ov_sam6d/bin/activate

source ~/xkd-openvino/ov_dist_mix/setupvars.sh

source install/setup.bash

ros2 run sam_6d_ros pose_estimation --ros-args --params-file install/sam_6d_ros/share/sam_6d_ros/params.yaml

# Note: before running the sam_6d_ros node, make sure the parameters in 'params.yaml' are correct:
pose_estimation:
    ros__parameters:
        segmentor_model: "fastsam"
        output_dir : "/home/intel/tmp/sam_6d_ros"
        cad_path : "/home/intel/tmp/sam_6d_ros/cad/logistics-box.ply"
        stability_score_thresh : 0.97
        depth_scale : 1.0
        visualize : True
        use_detection : True
        model_dir : "/home/intel/ros2_ws/checkpoints"
        ov_extension_lib_path : "/usr/local/lib/libopenvino_operation_extension.so"
        rgb_topic : "/camera/camera/color/image_raw"
        depth_topic : "/camera/camera/aligned_depth_to_color/image_raw"
        camera_info_topic : "/camera/camera/aligned_depth_to_color/camera_info"
```

#### 2.6.4 Call SAM-6D service in a terminal
```bash
cd ~/ws_sam6d

source ov_sam6d/bin/activate

source ~/xkd-openvino/ov_dist_mix/setupvars.sh

source install/setup.bash

ros2 service call /get_pose/sam6d pose_interfaces/srv/GetPose "{command_id: 1}"

# Note:
#     command_id: 1 for YOLO + MobileSAM + PEM pipeline
#     command_id: 2 for ISM + PEM pipeline
```