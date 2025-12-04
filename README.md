# 🚗 Jetbot Self-Driving Car

## 🖥️ Environment

* **Ubuntu** : 18.04.6 LTS
* **ROS** : melodic
* **CUDA** : 10.2
* **Python** : 3.6.9（default version on Jetson nano）

## 🧰 Hardware

*  **Nvidia Jetson Nano 4GB**
*  **Arduino Uno**
*  **Logitech USB 2D Camera**
*  **RPi Lidar A1**
*  **DC Motor x2**
*  **L298N Motor Driver**
*  **DC Power Bank**

---

# 🧪 Training Environment Setup (ScaledYOLOv4 + CUDA)

> Training is done on Desktop GPU (RTX series). Jetson Nano is used for inference only.

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

### models/yolo.py — Fix initialize_biases

```python
b = mi.bias.view(m.na, -1).clone()
```

### numpy dtype fixes

```bash
find . -name "*.py" -exec sed -i 's/np.int/int/g' {} +
find . -name "*.py" -exec sed -i 's/dtype=int16/dtype=np.int16/g' {} +
sed -i 's/astype(int64)/astype(np.int64)/g' utils/general.py
sed -i 's/astype(int64)/astype(np.int64)/g' test.py
```

### general.py — Fix build_targets()

```python
gj = gj.clamp(0, int(gain[3].item() - 1e-3))
gi = gi.clamp(0, int(gain[2].item() - 1e-3))
```

### datasets.py — Cache bug fix

```bash
sed -i "s/torch.load(cache_path)/torch.load(cache_path, weights_only=False)/" utils/datasets.py
rm -f ../train/labels.cache ../valid/labels.cache
```

### test.py — Disable plotting

```python
#plot_images(...)
```

---

# 🗂️ Project Folder Structure

```text
yolov4/
├── ScaledYOLOv4/                 📁 Main training code
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── test.py
│   └── ...
│
├── mish-cuda/                    ⚙️ CUDA Mish activation
│   ├── setup.py
│   └── ...
│
├── data/                         🗂️ YOLO-format dataset
│   ├── train/
│   │   ├── images/               🖼️ Training images
│   │   └── labels/               📝 YOLO txt labels
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/ (optional)
│   │   ├── images/
│   │   └── labels/
│   └── roboflow-raw/ (optional)
│
├── data.yaml                     📘 Dataset configuration
│
├── runs/                         📊 Training logs + weights
│   ├── exp/
│   ├── exp2/
│   └── ...
│
├── scripts/                      🛠️ Helper tools
│   ├── train.sh
│   ├── export_onnx.py
│   └── visualize_data.py
│
├── requirements.txt
└── README.md
```

---

# 📘 Example `data.yaml`

```yaml
train: ./data/train/images
val: ./data/valid/images
nc: 3
names: ["left", "right", "stop"]
```

---

# 🚀 Start Training

```bash
cd ~/yolov4/ScaledYOLOv4

python train.py \
  --img 416 \
  --batch 16 \
  --epochs 50 \
  --data ../data.yaml \
  --cfg ./models/yolov4-csp.yaml \
  --weights '' \
  --name traffics_detection
```
