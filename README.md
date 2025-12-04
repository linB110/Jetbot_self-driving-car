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

🔧 models/yolo.py — Fix _initialize_biases()

File: models/yolo.py
Function: _initialize_biases(self, cf=None)
Typical location: around line 430–470

# avoid leaf in-place operation on torch variable view
b = mi.bias.view(m.na, -1).clone()   # (na, no)
🔧 numpy dtype fixes (global)

Files affected: all python files under ScaledYOLOv4

find . -name "*.py" -exec sed -i 's/np.int/int/g' {} +
find . -name "*.py" -exec sed -i 's/dtype=int16/dtype=np.int16/g' {} +
🔧 utils/general.py — Fix build_targets()

File: utils/general.py
Function: build_targets()
Location: around line 550–570

Original:

indices.append((b, a, gj.clamp_(0, gain[3]), gi.clamp_(0, gain[2])))

Modified:

gj = gj.clamp(0, int(gain[3].item() - 1e-3))
gi = gi.clamp(0, int(gain[2].item() - 1e-3))
indices.append((b, a, gj, gi))
🔧 utils/general.py — Fix output_to_target()

Function: output_to_target()
Location: around line 840–880

if isinstance(targets, list):
    if len(targets):
        targets = torch.stack(targets, 0)
    else:
        return np.zeros((0, 6), dtype=np.float32)
elif isinstance(targets, torch.Tensor):
    if targets.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
else:
    targets = torch.as_tensor(targets)
return targets.detach().cpu().numpy()
🔧 utils/datasets.py — Fix cache loading (PyTorch 2.x)

File: utils/datasets.py
Location: around line 300–330

sed -i "s/torch.load(cache_path)/torch.load(cache_path, weights_only=False)/" utils/datasets.py

Also clear old cache:

rm -f ../train/labels.cache ../valid/labels.cache
🔧 train.py — Fix interp → np.interp

File: train.py
Location: search keyword interp(

sed -i 's/interp(/np.interp(/g' train.py
🔧 Disable plotting in test.py

File: test.py
Location: around line 185–200

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
