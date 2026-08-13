from __future__ import annotations

import numpy as np

from utils import decode_yolov5_heads, detect_postprocess, detect_postprocess_outputs, preprocess_image


def test_preprocess_image_pads_to_square_and_scales():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, scale = preprocess_image(image, 640)
    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.float32
    assert scale == 1.0


def test_detect_postprocess_decodes_single_output():
    prediction = np.zeros((1, 25200, 85), dtype=np.float32)
    prediction[0, 0, :4] = [320, 240, 100, 80]
    prediction[0, 0, 4] = 0.8
    prediction[0, 0, 10] = 0.9
    detections = detect_postprocess(prediction, (480, 640, 3), 1.0, conf_thres=0.25)
    assert detections.shape == (1, 6)
    np.testing.assert_allclose(detections[0, :4], [270, 200, 370, 280], atol=1e-5)
    np.testing.assert_allclose(detections[0, 4], 0.72, atol=1e-6)
    assert detections[0, 5] == 5


def test_detect_postprocess_decodes_split_outputs():
    boxes = np.zeros((1, 4, 8400), dtype=np.float32)
    scores = np.zeros((1, 80, 8400), dtype=np.float32)
    boxes[0, :, 0] = [320, 240, 100, 80]
    scores[0, 5, 0] = 0.9
    detections = detect_postprocess_outputs([boxes, scores], "split", (480, 640, 3), 1.0, conf_thres=0.25)
    assert detections.shape == (1, 6)
    np.testing.assert_allclose(detections[0, :4], [270, 200, 370, 280], atol=1e-5)
    assert detections[0, 5] == 5


def test_decode_yolov5_cutoff_heads():
    heads = [
        np.full((1, 80, 80, 255), -20.0, dtype=np.float32),
        np.full((1, 40, 40, 255), -20.0, dtype=np.float32),
        np.full((1, 20, 20, 255), -20.0, dtype=np.float32),
    ]
    y, x, anchor = 10, 20, 0
    start = anchor * 85
    heads[0][0, y, x, start:start + 4] = 0.0
    heads[0][0, y, x, start + 4] = 20.0
    heads[0][0, y, x, start + 10] = 20.0
    decoded = decode_yolov5_heads(heads)
    row = decoded[(y * 80 + x) * 3 + anchor]
    np.testing.assert_allclose(row[:4], [164, 84, 10, 13], atol=1e-5)
    assert row[4] > 0.99
    assert row[10] > 0.99
