import numpy as np
import os

# 配置
out_dir = "run_data/example/calib_data"
os.makedirs(out_dir, exist_ok=True)
calib_list_path = "run_data/example/calib.txt"
sample_count = 20   # 一般10~50条足够，越多统计越准

# 模型输入shape [1,16] float32
shape = (1, 16)
paths = []

for i in range(sample_count):
    # 建议使用真实业务分布数据；临时测试可用随机样本
    data = np.random.randn(*shape).astype(np.float32)
    fname = os.path.join(out_dir, f"sample_{i:03d}.raw")
    data.tofile(fname)
    paths.append(fname)

# 写入calib.txt
with open(calib_list_path, "w", encoding="utf-8") as f:
    for p in paths:
        f.write(p + "\n")

print(f"生成 {sample_count} 条校准样本，清单写入 {calib_list_path}")