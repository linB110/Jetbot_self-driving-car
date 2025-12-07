# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import time

# ================== setting ==================
ENGINE_PATH = "/home/lab605/yolo/yolov4.trt"

# should be same with training image size
INPUT_SIZE = (320, 320)

# classes
CLASS_NAMES = [
    'Ahead-only', 'General-caution', 'No-entry', 'Pedestrians',
    'Speed Limit 100', 'Speed Limit 120', 'Speed Limit 20',
    'Speed Limit 30', 'Speed Limit 50', 'Speed Limit 60',
    'Speed Limit 70', 'Speed Limit 80', 'Stop',
    'Turn-left-ahead', 'Turn-right-ahead'
]

NUM_CLASSES = len(CLASS_NAMES)

# thresholds
CONF_THRESHOLD = 0.8
IOU_THRESHOLD = 0.3

# how many frames to inference
DETECT_EVERY_N_FRAMES = 5

def xywh2xyxy(x):
    """
    x: (N,4) -> [cx,cy,w,h] 轉成 [x1,y1,x2,y2]
    """
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def bbox_iou(box1, box2):
    """
    Compute IoU of two bbox
    box: [x1, y1, x2, y2, ...]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def nms(bboxes, iou_threshold=0.5):
    """
    bboxes: [[x1, y1, x2, y2, conf, cls_id], ...]
    NMS bu classes
    """
    if len(bboxes) == 0:
        return []

    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    final_boxes = []

    while bboxes:
        best = bboxes.pop(0)
        final_boxes.append(best)

        remain = []
        for box in bboxes:
            # not suppress by different classes
            if box[5] != best[5]:
                remain.append(box)
                continue

            iou = bbox_iou(best, box)
            if iou < iou_threshold:
                remain.append(box)

        bboxes = remain

    return final_boxes


# ================== TensorRT inference ==================

class TRTInference:
    def __init__(self, engine_path, input_shape):
        """
        input_shape: (C, H, W) 例如 (3, 320, 320)
        """
        self.input_shape = input_shape
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        print(f"[TRT] importing TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"can't import TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            binding_shape = self.engine.get_binding_shape(binding)

            # resize to assigned shape
            if -1 in binding_shape and self.engine.binding_is_input(binding):
                binding_shape = (1, *self.input_shape)
                self.context.set_binding_shape(binding_idx, binding_shape)

            # azquire real shape
            shape = self.context.get_binding_shape(binding_idx)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
                #print(f"[TRT]  input binding: {binding}, shape={shape}")
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})
                #print(f"[TRT]  output binding: {binding}, shape={shape}")

        print(f"[TRT] finished initialization：")

    def infer(self, input_data: np.ndarray):
        """
        input_data: numpy array (1, C, H, W), float32
        returm: list[np.ndarray]
        """
      
        np.copyto(self.inputs[0]["host"], input_data.ravel())

        # Host -> Device
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        # inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Device -> Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()

        # reshape or numpy array
        results = []
        for out in self.outputs:
            arr = np.array(out["host"]).reshape(out["shape"])
            results.append(arr)

        return results


# ================== post processing：from Nx(5+num_classes) => bbox ==================

def postprocess_trt(outputs, orig_shape, input_size):
    """
    orig_shape: (orig_h, orig_w)
    input_size: (H, W) 例如 (320, 320)
    return: [[x1, y1, x2, y2, score, cls_id], ...]（in pixels）
    """
    orig_h, orig_w = orig_shape
    in_h, in_w = input_size

    preds = None

    target_last_dim = 5 + NUM_CLASSES
    for idx, out in enumerate(outputs):
        # print(f"[DEBUG] output[{idx}] shape: {out.shape}")
        if out.ndim == 3 and out.shape[-1] == target_last_dim:
            preds = out
            # print(f"[DEBUG] 使用 output[{idx}] 作為 Nx(5+num_classes) 輸出")
            break

    if preds is None:
        preds = outputs[-1]

    if preds.ndim == 3:
        preds = preds[0]  # -> (N, 5+NUM_CLASSES)

    if preds.shape[1] != target_last_dim:
        raise ValueError(
            f"output size is unexpected: {preds.shape[1]}，"
            f"should be 5+NUM_CLASSES = {target_last_dim}"
        )

    # separate box / obj / class conf
    boxes = preds[:, 0:4]       # cx, cy, w, h
    obj_conf = preds[:, 4:5]    # (N,1)
    cls_conf = preds[:, 5:]     # (N,num_classes)

    # get max class
    cls_ids = np.argmax(cls_conf, axis=1)
    cls_scores = cls_conf[np.arange(cls_conf.shape[0]), cls_ids]
    scores = cls_scores * obj_conf.squeeze(-1)

    # confidence filter
    mask = scores >= CONF_THRESHOLD
    if not np.any(mask):
        return []

    boxes = boxes[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    # [cx,cy,w,h] -> [x1,y1,x2,y2] 
    boxes_xyxy = xywh2xyxy(boxes)

    gain_w = orig_w / float(in_w)
    gain_h = orig_h / float(in_h)
    boxes_xyxy[:, [0, 2]] *= gain_w
    boxes_xyxy[:, [1, 3]] *= gain_h

    # compose [x1,y1,x2,y2,score,cls_id]
    dets = []
    for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
        dets.append([float(x1), float(y1), float(x2), float(y2), float(sc), int(cid)])

    # NMS
    dets = nms(dets, IOU_THRESHOLD)
    return dets


# ================== YOLO + TensorRT detector ==================

class YoloTRTDetector:
    def __init__(self, engine_path, input_size=(320, 320)):
        """
        input_size: (H, W) 
        """
        self.input_size = input_size
        self.infer_engine = TRTInference(
            engine_path,
            input_shape=(3, input_size[0], input_size[1])
        )

    def preprocess(self, frame):
        """
        -  resize to (H,W)
        - BGR -> RGB
        - /255
        """
        orig_h, orig_w = frame.shape[:2]

        H, W = self.input_size
        img = cv2.resize(frame, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # (1,3,H,W)

        meta = {"orig_h": orig_h, "orig_w": orig_w}
        return img, meta

    def detect(self, frame):
        img_input, meta = self.preprocess(frame)
        outputs = self.infer_engine.infer(img_input)

        orig_h = meta["orig_h"]
        orig_w = meta["orig_w"]

        bboxes = postprocess_trt(outputs, (orig_h, orig_w), self.input_size)
        return bboxes, frame


# ================== Camera real-time detection ==================

def main():
    print("=" * 60)
    print("YOLOv4 TensorRT detection（every 5 frame）")
    print("=" * 60)

    detector = YoloTRTDetector(ENGINE_PATH, input_size=INPUT_SIZE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("can't open camera")

    print("press 'q' to exit")

    frame_idx = 0
    last_bboxes = []          # last detection
    last_infer_time = 0.0     # last inference time cost
    prev_disp_time = time.time()
    disp_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("read frame fail")
            break

        frame_idx += 1

        # compute FPS
        now = time.time()
        disp_fps = 1.0 / (now - prev_disp_time + 1e-6)
        prev_disp_time = now

        # how many frames to run inference
        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            t1 = time.time()
            last_bboxes, img_show = detector.detect(frame)
            t2 = time.time()
            last_infer_time = t2 - t1
        else:
            img_show = frame.copy()

        # draw last detection
        for box in last_bboxes:
            x1, y1, x2, y2, conf, cls_id = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            if 0 <= cls_id < len(CLASS_NAMES):
                label = CLASS_NAMES[cls_id]
            else:
                label = f"id_{cls_id}"

            cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_show,
                f"{label} {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        #  FPS：display FPS + inference FPS
        infer_fps = (1.0 / last_infer_time) if last_infer_time > 0 else 0.0
        cv2.putText(
            img_show,
            f"Disp FPS: {disp_fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        cv2.putText(
            img_show,
            f"Infer FPS: {infer_fps:.2f} (every {DETECT_EVERY_N_FRAMES} frames)",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLOv4 TRT Detection", img_show)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
