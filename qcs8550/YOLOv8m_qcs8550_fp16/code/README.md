## Model Information

### Source Model

- Model: YOLOv8m
- Input shape: 640x640
- Restored int8 model: cutoff W8A8
- Output layout: `split`
- Output shapes: `[[1, 80, 8400], [1, 4, 8400]]`
- Target SoC: QCS8550
- QNN SDK: 2.36.0.250627
- AIMO task id: 188875a941f74aae92d2e2bbcdc31776

## Inference with AidLite SDK

```bash
cd YOLOv8m
python3 code/python/run_test.py   --target_model ./models/QCS8550/W8A8/cutoff_yolov8m_qcs8550_w8a8.qnn236.ctx.bin   --imgs ./code/python/bus.jpg   --invoke_nums 10
```
