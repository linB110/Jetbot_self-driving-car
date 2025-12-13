<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/21551bd1-d1c1-434d-9974-ad2f3f5008da" /># 🚗 Jetbot Self-Driving Car

![My image](./asset/jetbot.png)

---

## ⚙️ system architecture

![System](./asset/systen_structure.png)

---

The video is recorded via SSH connection, so it might seems laggy (but inference is runnung at 30+FPS basis on Jetson nano)
<video controls width="600" src="https://github.com/user-attachments/assets/828a5b6e-d7d7-46b5-a1ad-34fc8aba68cd"></video>

---

## 🖥️ Environment

* **Ubuntu** : 18.04.6 LTS
* **ROS** : melodic
* **CUDA** : 10.2
* **Python** : 3.6.9 (default version on Jetson Nano)

---

## 🧰 Hardware

* **Nvidia Jetson Nano 4GB**
* **Arduino Uno**
* **Logitech USB 2D Camera**
* **RPi LiDAR A1**
* **DC Motors ×2**
* **L298N Motor Driver**
* **DC Power Bank**

---

# 🧪 Training Environment Setup (ScaledYOLOv4 + CUDA)

> Training is done on a desktop GPU (RTX series). Jetson Nano is used for inference only.

> 👉 [Training environment](./Env/Training_env.txt)

## 🚀 1. Create Conda Environment

```bash
conda create -n yolov4 python=3.10 -y
conda activate yolov4
```

## ⚡ 2. Install PyTorch (CUDA 12.8+)

```bash
pip install torch torchvision torchaudio
```

Test GPU:

```bash
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF
```

## 📥 3. Clone ScaledYOLOv4 + Mish-CUDA

```bash
mkdir -p ~/yolov4
cd ~/yolov4

git clone https://github.com/WongKinYiu/ScaledYOLOv4.git
git clone https://github.com/JunnYu/mish-cuda.git
```

Install mish-cuda:

```bash
cd ~/yolov4/mish-cuda
python setup.py build install
```

## 📦 4. Install Python Dependencies

```bash
pip install numpy==1.24.4 opencv-python pillow tqdm matplotlib pyyaml scipy seaborn cython tensorboard onnx thop
```

## 🛠️ 5. Apply Required Code Fixes

👉 [Fixes](./Env/readme.md)

---

# 🗂️ Training Folder Structure

```text
yolov4/
├── ScaledYOLOv4/                 # Main training code
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── test.py
│   └── ...
│
├── mish-cuda/                    # CUDA Mish activation
│   ├── setup.py
│   └── ...
│
├── data/                         # YOLO-format dataset
│   ├── train/
│   │   ├── images/               # Training images
│   │   └── labels/               # YOLO txt labels
│   ├── valid/
│   ├── test/
│   └── roboflow-raw/
│
├── data.yaml                     # Dataset configuration
│
├── runs/                         # Training logs + weights
│
├── model_transform/              # ONNX / TensorRT export tools
│   ├── export_onnx.py
│   ├── export_trt.py
│
└── README.md
```

---

# 📘 Example `data.yaml`

```yaml
train: ./data/train/images
val: ./data/valid/images
nc: 15
names: ['100km', '120km', '20km', '30km', '50km', '60km', '70km', '80km',
        'Ahead-only', 'General-caution', 'No-entry', 'Pedestrians',
        'Stop', 'Turn-left-ahead', 'Turn-right-ahead']
```

---

# 🚀 Start Training

```bash
cd ~/yolov4/ScaledYOLOv4

python train.py \
  --img 320 \
  --batch 16 \
  --epochs 50 \
  --data ../data.yaml \
  --cfg ./models/yolov4-csp.yaml \
  --weights '' \
  --name traffics_detection
```

---

# 🗂️ Project Folder Structure (Jetson + ROS)

```text
Arduino/
├── libraries/ros_lib                # ROS dependencies
│   ├── ros/
│   ├── ros.h/
│   └── ...
│
├── motor_motion/                    # Motor control logic
│   ├── motor_motion.h
│   ├── motor_motion.cpp
│   └── motor_motion.ino

catkin_ws/src                        # ROS workspace
│── rplidar_ros/                     # lidar detection and sensing
│   ├── launch/
│   ├── src/
│   └── ...
│
│── lidar/                          # lidar projection
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── config/
│   ├── scripts/lidar_projection.py
│   ├── scripts/lidar_yolo_fusion.py
│   └── launch/lidar_camera_fusion.launch
│
│── yolo_detection/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── msg/detection.msg
│   ├── launch/yolo_inference.launch
│   ├── config/yolo_param.yaml
│   ├                          
│   ├── include/                     # yolo inference
│   │   ├── utils.h
│   │   ├── trt_yolo.h
│   │   └──  yolo_ros_node.h
│   │
│   └── src/
│   │   ├── utils.cpp
│   │   ├── trt_yolo.cpp
│   │   ├── yolo_ros_node.cpp
│   │   └── main.cpp

```

---

# 🧠 Jetson Nano – Inference Pipeline

Final autonomous flow:

```
YOLOv4 image detection
        ↓
LiDAR distance detection
        ↓
Camera–LiDAR fusion
        ↓
Decision logic (stop / slow / turn)
        ↓
Arduino motor control
```

---

🔍 YOLO Inference on Jetson Nano

1️⃣ Load YOLO Parameters

Edit : yolo_detection/config/yolo_param.yaml

2️⃣ Launch YOLO Inference

`roslaunch yolo_detection yolo_inference.launch use_gui:=true`

* `use_gui:=true` → display camera with bounding boxes

---

# ▶️ Running the Full System on Jetbot

Below are the actual commands used to run the entire pipeline.

## **1️⃣ Start ROS Master**

```bash
roscore
```

## **2️⃣ Launch LiDAR + Camera + YOLO Fusion Node**

```bash
roslaunch lidar lidar_camera_fusion.launch use_gui:=true
```

* `use_gui:=true` → display camera with bounding boxes and lidar points projected
* Disable GUI:

```bash
use_gui:=false
```

## **3️⃣ Start Arduino Motor Control (ROSserial)**

```bash
rosrun rosserial_python serial_node.py _port:=/dev/ttyACM0
```

Jetbot begins autonomous driving based on fused perception + decision-making.

---
