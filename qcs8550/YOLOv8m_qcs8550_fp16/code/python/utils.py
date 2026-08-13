from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


COCO_CLASSES: Tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
YOLOV5_ANCHORS = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)
YOLOV5_STRIDES = (8, 16, 32)


def preprocess_image(image: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    length = max(height, width)
    scale = length / input_size
    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[:height, :width] = image
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas = cv2.resize(canvas, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return (canvas.astype(np.float32) / 255.0)[None, :], scale


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def clip_boxes(boxes: np.ndarray, shape: Sequence[int]) -> None:
    boxes[:, 0].clip(0, shape[1], out=boxes[:, 0])
    boxes[:, 1].clip(0, shape[0], out=boxes[:, 1])
    boxes[:, 2].clip(0, shape[1], out=boxes[:, 2])
    boxes[:, 3].clip(0, shape[0], out=boxes[:, 3])


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-12)
        order = rest[iou <= iou_thres]
    return keep


def normalize_prediction(prediction: np.ndarray, class_num: int = 80) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    channels = class_num + 5
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {prediction.shape}")
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D or 3D prediction, got {prediction.shape}")
    if prediction.shape[0] == channels:
        prediction = prediction.T
    if prediction.shape[1] != channels:
        raise ValueError(f"Expected prediction with {channels} channels, got {prediction.shape}")
    return prediction


def merge_split_outputs(outputs: Sequence[np.ndarray], class_num: int = 80) -> np.ndarray:
    boxes = None
    scores = None
    extras = []
    for output in outputs:
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            extras.append(arr)
            continue
        if 4 in arr.shape and boxes is None:
            boxes = arr if arr.shape[0] == 4 else arr.T
        elif class_num in arr.shape and scores is None:
            scores = arr if arr.shape[0] == class_num else arr.T
        else:
            extras.append(arr)
    if boxes is None or scores is None:
        raise ValueError("Could not find box and class outputs")
    if boxes.shape[1] != scores.shape[1]:
        raise ValueError(f"Output anchor counts differ: {boxes.shape} vs {scores.shape}")
    obj = np.ones((1, boxes.shape[1]), dtype=np.float32)
    return np.concatenate([boxes, obj, scores], axis=0).T


def decode_yolov5_heads(outputs: Sequence[np.ndarray], class_num: int = 80) -> np.ndarray:
    heads = []
    for output in outputs:
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] % 3 == 0:
            heads.append(arr)
    heads = sorted(heads, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
    decoded = []
    channels = class_num + 5
    for pred, layer_anchors, stride in zip(heads, YOLOV5_ANCHORS, YOLOV5_STRIDES):
        height, width = pred.shape[:2]
        values_per_anchor = pred.shape[-1] // len(layer_anchors)
        pred = pred.reshape(height, width, len(layer_anchors), values_per_anchor)
        pred = sigmoid(pred)
        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        grid = np.stack((grid_x, grid_y), axis=-1)[:, :, None, :].astype(np.float32)
        anchor_grid = np.asarray(layer_anchors, dtype=np.float32)[None, None, :, :]
        xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (pred[..., 2:4] * 2.0) ** 2 * anchor_grid
        obj = pred[..., 4:5]
        cls = pred[..., 5:5 + class_num]
        decoded.append(np.concatenate((xy, wh, obj, cls), axis=-1).reshape(-1, channels))
    if not decoded:
        raise ValueError("No YOLOv5-style raw heads found")
    return np.concatenate(decoded, axis=0)


def detect_postprocess(
    prediction: np.ndarray,
    original_shape: Sequence[int],
    scale: float,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    class_num: int = 80,
    max_det: int = 300,
) -> np.ndarray:
    prediction = normalize_prediction(prediction, class_num)
    return postprocess_decoded(prediction, original_shape, scale, conf_thres, iou_thres, class_num, max_det)


def detect_postprocess_outputs(
    outputs: Sequence[np.ndarray],
    output_layout: str,
    original_shape: Sequence[int],
    scale: float,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    class_num: int = 80,
    max_det: int = 300,
) -> np.ndarray:
    if output_layout == "raw_yolov5":
        prediction = decode_yolov5_heads(outputs, class_num)
    elif output_layout in ("split", "seg", "pose", "obb"):
        if output_layout == "pose":
            class_num = 1
        elif output_layout == "obb":
            class_num = 15
        prediction = merge_split_outputs(outputs, class_num)
    elif output_layout == "single":
        prediction = normalize_prediction(outputs[0], class_num)
    else:
        raise ValueError(f"Unsupported output_layout: {output_layout}")
    return postprocess_decoded(prediction, original_shape, scale, conf_thres, iou_thres, class_num, max_det)


def postprocess_decoded(
    prediction: np.ndarray,
    original_shape: Sequence[int],
    scale: float,
    conf_thres: float,
    iou_thres: float,
    class_num: int,
    max_det: int,
) -> np.ndarray:
    obj_conf = prediction[:, 4]
    class_conf = prediction[:, 5:5 + class_num]
    class_ids = class_conf.argmax(axis=1)
    class_scores = class_conf[np.arange(class_conf.shape[0]), class_ids]
    scores = obj_conf * class_scores
    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0, 6), dtype=np.float32)
    boxes = xywh2xyxy(prediction[mask, :4])
    boxes *= scale
    clip_boxes(boxes, original_shape)
    scores = scores[mask]
    class_ids = class_ids[mask]
    detections: List[np.ndarray] = []
    for class_id in np.unique(class_ids):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores_for_id = scores[class_mask]
        keep = nms(class_boxes, class_scores_for_id, iou_thres)[:max_det]
        if keep:
            rows = np.column_stack(
                (class_boxes[keep], class_scores_for_id[keep], np.full(len(keep), class_id, dtype=np.float32))
            )
            detections.append(rows.astype(np.float32))
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    output = np.concatenate(detections, axis=0)
    output = output[output[:, 4].argsort()[::-1]]
    return output[:max_det]


def draw_detect_res(image: np.ndarray, detections: Iterable[Sequence[float]]) -> np.ndarray:
    result = image.astype(np.uint8).copy()
    overlay = result.copy()
    color_step = max(1, int(255 / len(COCO_CLASSES)))
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2, score, class_id = det
        class_id = int(class_id)
        color = (0, int(class_id * color_step) % 255, int(255 - class_id * color_step) % 255)
        p1 = int(round(x1)), int(round(y1))
        p2 = int(round(x2)), int(round(y2))
        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
        print(i, [p1[0], p1[1], p2[0], p2[1]], float(score), label)
        cv2.rectangle(result, p1, p2, color, thickness=2)
        cv2.putText(result, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.addWeighted(overlay, 0.3, result, 0.7, 0)
