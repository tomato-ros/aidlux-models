# aidlux-models

model example code

- [aidlite-sdk 介绍](https://docs.aidlux.com/guide/software/sdk/aidlite/aidlite-sdk)

## 安装SDK

```shell
sudo aid-pkg update
sudo aid-pkg install aidlite-sdk
sudo aid-pkg install aidlite-qnn248
aid-pkg list
```

## 下载模型

- 模型广场：https://aiot.aidlux.com/zh/models

### 英文翻译中文模型示例

- 模型地址：https://aiot.aidlux.com/zh/models/detail/35?modelType=14&soc=4

- 登录

```shell
mms login
```

- 模型查询

```shell
mms list opus-mt-en-zh
```

返回结果：

```text
Model          Precision  Chipset  Backend  SOC
-----          ---------  -------  -------  -------
opus-mt-en-zh  FP16       qcs8550  QNN2.36  Qualcomm QCS8550
```

- 模型下载

```shell
# mms get -m [model name] -p [precision] -c [soc] -b [backend] -d [file path]

mms get -m opus-mt-en-zh -p FP16 -c qcs8550 -b QNN2.36 -d .
```

返回结果:

```text
Downloading model from https://aiot.aidlux.com to directory: .
Downloading [opus-mt-en-zh_qcs8550_fp16.zip] ... done! [270.40MB in 25.782s; 10.42MB/s]
Download complete!
```

- 解压模型

```shell
unzip ./opus-mt-en-zh_qcs8550_fp16.zip
```

返回结果:

```text
Archive:  ./opus-mt-en-zh_qcs8550_fp16.zip
  inflating: code/README.md
  inflating: code/aidlite_tool.py
  inflating: code/main.py
  inflating: code/opus-mt-en-zh/.gitattributes
  inflating: code/opus-mt-en-zh/README.md
  inflating: code/opus-mt-en-zh/config.json
  inflating: code/opus-mt-en-zh/generation_config.json
  inflating: code/opus-mt-en-zh/metadata.json
  inflating: code/opus-mt-en-zh/source.spm
  inflating: code/opus-mt-en-zh/target.spm
  inflating: code/opus-mt-en-zh/tokenizer_config.json
  inflating: code/opus-mt-en-zh/vocab.json
  inflating: code/requirements.txt
  inflating: models/QCS8550/FP16/opus_mt_decoder_htp.bin
  inflating: models/QCS8550/FP16/opus_mt_encoder_htp.bin
  inflating: models/QCS8550/FP16/source.spm
  inflating: models/QCS8550/FP16/target.spm
  inflating: models/QCS8550/FP16/vocab.json
```

- run example code

修改 `aidlite_tool.py` 代码QNN版本后端库

```python
config.framework_type = aidlite.FrameworkType.TYPE_QNN236
```

修改为

```python
config.framework_type = aidlite.FrameworkType.TYPE_QNN248
```

运行代码:

```shell
python main.py
```
