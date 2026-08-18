import torch
import torch.nn as nn
import onnx
import onnxsim


# --------------------------
# 1. 定义简单多层神经网络
# --------------------------
class SimpleNet(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=4):
        super().__init__()
        # 多层全连接
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # forward 内部只写张量运算，不要原生python循环/分支（方便导出ONNX）
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        out = self.fc3(x)
        return out


# --------------------------
# 2. 初始化模型、切推理模式
# --------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleNet(in_dim=16, hidden_dim=32, out_dim=4).to(device)
model.eval()  # 导出onnx必须eval
torch.set_grad_enabled(False)  # 关闭梯度

# --------------------------
# 3. 构造虚拟输入（和真实推理shape一致）
# --------------------------
# batch_size=2, feature_dim=16
dummy_input = torch.randn(2, 16, dtype=torch.float32).to(device)

# --------------------------
# 4. ONNX导出
# --------------------------
onnx_path = "run_data/example/simple_net.onnx"
with torch.no_grad():
    torch.onnx.export(model,
                      (dummy_input,),
                      f=onnx_path,
                      opset_version=17,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"]
                      )

print(f"原始onnx导出完成：{onnx_path}")

# --------------------------
# 5. onnxsim 模型化简（消除冗余节点）
# --------------------------
model_onnx = onnx.load(onnx_path)
sim_model, check_ok = onnxsim.simplify(model_onnx)
if check_ok:
    onnx.save(sim_model, onnx_path)
    print("onnxsim 化简成功！")
else:
    print("onnxsim 化简警告，请手动检查模型")

# --------------------------
# 6. 验证onnx合法性
# --------------------------
onnx.checker.check_model(sim_model)
print("✅ ONNX模型校验通过")

# --------------------------
# 7. 简单推理验证
# --------------------------
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("run_data/example/simple_net.onnx")
inp = np.random.randn(2, 16).astype(np.float32)
out = sess.run(["output"], {"input": inp})
print("onnx 推理输入shape:{0} 输出shape:{1}".format(inp.shape, out[0].shape))
