## Jetson Nano Environment Setup

### 1. Jetson Nano initial setup

- Device: **NVIDIA Jetson Nano A01**
- OS: **Ubuntu 18.04.6 LTS**
- JetPack: **4.6 (L4T 32.7.6)**
- Default Python version: **Python 3.6.9**

I followed NVIDIA’s official Jetson Nano setup guide to:
- Flash the SD card image
- Complete the initial configuration (user account, network, etc.)
- Enable the performance mode / power configuration

Reference:  
- [NVIDIA Jetson Nano – Get Started](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro)

---

### 2. OpenCV

The default OpenCV version on Jetson Nano (installed via JetPack) is **4.1.1**, and CUDA support is **not** enabled in that build.

For this project, I manually built **OpenCV 4.5.1** from source with:
- CUDA enabled
- cuDNN support
- OpenCV DNN CUDA backend

Some very helpful resources I used:

- [Build OpenCV from source on Jetson Nano – Video 1](https://www.youtube.com/watch?v=mmyxWBOo1kg&t=1566s)  
- [Build OpenCV from source on Jetson Nano – Video 2](https://www.youtube.com/watch?v=P-EZr0zy53g)

---

### 3. PyTorch and Torchvision

Because JetPack 4.6 uses:
- CUDA 10.2
- Python 3.6.9

I installed a **Jetson-compatible build** of:
- **PyTorch 1.8.0 (aarch64, Python 3.6)**
- The matching **torchvision** version is '0.9.0'

These are installed inside my Python virtual environment `yolo-env`.

Reference tutorial:  
- [Build Torch and Torchvision on Jetson Nano](https://www.youtube.com/watch?v=o8QuRm-is_I)

---

## YOLO Model

Due to the constraints of JetPack 4.6 and Python 3.6.9, the latest YOLOv5 implementation is **not fully compatible** with this environment (it officially requires Python ≥ 3.8 and newer PyTorch versions).  
Therefore, in this project I use **YOLOv4** instead of YOLOv5.

- Device: Jetson Nano A01
- JetPack: 4.6 (L4T 32.7.6)
- Python: 3.6.9
- Deep learning framework: PyTorch + CUDA (Jetson build)
- Detection model: **YOLOv4**

### Model weights

The model file (weights / configuration) used in this project is stored under:

- `Reference/model/`

You can find the model download link inside the **cover folder** under `Reference/model`.

inference stage is divide into 2-stage (coarse and fine) which contains 4 and 43 classes respectively
model 1 : Yolov4.weights
model 2 : traffic.h5 => compile to onnx (on PC) => build to engine (tensorRT on jetson nano)
