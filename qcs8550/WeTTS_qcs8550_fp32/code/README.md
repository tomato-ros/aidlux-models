## Model Information

### Source model
- Input shape: Dynamic input
- Number of parameters: --,--
- Model size: 410M,65.9M
- Output shape: Dynamic output

Source model repository: [wetts](https://github.com/wenet-e2e/wetts)

## Inference 

### Installation

```bash
pip install onnxruntime
```

### Run Demo
```bash
cd model_farm_wetts_qcs6490_onnx_fp32_aidlite
python3 python/run_test.py --text_path "窗前明月光，疑是地上霜，举头望明月，低头思故乡" --wav_path ./python/results
python3 python/run_test.py --text_path python/chinese.txt --wav_path ./python/results
```