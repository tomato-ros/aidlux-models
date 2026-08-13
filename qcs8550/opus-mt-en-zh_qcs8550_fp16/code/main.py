from transformers import MarianTokenizer
import onnxruntime as ort
import numpy as np
import os
from aidlite_tool import AidliteTool
import pdb

def get_acc(onnx_out,other_out):
    cosine_similarity=np.dot(np.array(onnx_out),np.array(other_out))/(np.linalg.norm(np.array(onnx_out)) * np.linalg.norm(np.array(other_out)))
    return cosine_similarity

aid_tool=AidliteTool()

decoder_model_path="models/QCS8550/FP16/opus_mt_decoder_htp.bin"
encoder_model_path="models/QCS8550/FP16/opus_mt_encoder_htp.bin"

aid_tool.init(encoder_model_path,decoder_model_path)

MAX_SEQ_LEN_ENC = 256
MAX_SEQ_LEN_DEC = 256
TRANSPOSE_KEY = False

NUM_LAYERS = 6
#NUM_HEADS = 8
#HEAD_DIM = 64

class OnnxOpusMtEncoder:
    def __init__(self):
        # self.session = ort.InferenceSession(model_path)
        pass

    def forward(self, input_ids, encoder_attention_mask):
        inputs = {
            'input_ids': input_ids,
            'encoder_attention_mask': encoder_attention_mask,
        }

        # input_list = ""
        # os.makedirs(f"encoder_inputs", exist_ok=True)
        # for key, value in inputs.items():
        #     value.astype(np.float32).tofile(f"encoder_inputs/{key}.bin")
        #     input_list += f"{key}:=encoder_inputs/{key}.bin "
        # with open("encoder_input_list.txt","w") as f:
        #     f.write(input_list)

        output_names = []
        for layer_idx in range(NUM_LAYERS):
            output_names.append(f"block_{layer_idx}_cross_key_states")
            output_names.append(f"block_{layer_idx}_cross_value_states")
        # outputs = self.session.run(output_names, inputs)
        # for i in range(len(outputs)):
            # print(outputs[i][0,0,0,0])
        aid_res=aid_tool.encoder_infer(inputs)
        # print("/"*10)
        # for i in range(len(aid_res)):
            # print(aid_res[i][0,0,0,0])
        # for i in range(len(aid_res)):
            # acc=get_acc(outputs[i].flatten(),aid_res[i].flatten())
            # print("acc:",acc,end=" ")
        # pdb.set_trace()
        # print("")
        return aid_res

class OnnxOpusMtDecoder:
    def __init__(self):
        # self.session = ort.InferenceSession(model_path)
        pass

    def forward(self, input_ids, encoder_attention_mask, position, *past_key_values):
        inputs = {
            'input_ids': input_ids,
            'encoder_attention_mask': encoder_attention_mask,
            'position': position,
        }
        for layer_idx in range(NUM_LAYERS):
            inputs[f"block_{layer_idx}_past_self_key_states"] = past_key_values[4*layer_idx]
            inputs[f"block_{layer_idx}_past_self_value_states"] = past_key_values[4*layer_idx+1]
            inputs[f"block_{layer_idx}_cross_key_states"] = past_key_values[4*layer_idx+2]
            inputs[f"block_{layer_idx}_cross_value_states"] = past_key_values[4*layer_idx+3]

        # input_list = ""
        # os.makedirs(f"decoder_inputs", exist_ok=True)
        # for key, value in inputs.items():
        #     value.astype(np.float32).tofile(f"decoder_inputs/{key}.bin")
        #     input_list += f"{key}:=decoder_inputs/{key}.bin "
        # with open("decoder_input_list.txt","w") as f:
        #     f.write(input_list)

        output_names = ["logits"]
        for layer_idx in range(NUM_LAYERS):
            output_names.append(f"block_{layer_idx}_present_self_key_states")
            output_names.append(f"block_{layer_idx}_present_self_value_states")

        # outputs = self.session.run(output_names, inputs)
        aid_res=aid_tool.decoder_infer(inputs)
        
        # print("/"*10)
        # for i in range(len(aid_res)):
            # acc=get_acc(outputs[i].flatten(),aid_res[i].flatten())
            # print(i," acc:",acc," ",outputs[i].shape,aid_res[i].shape)

        return aid_res

intermediates_path="./model_en2zh"
model_name = './code/opus-mt-en-zh'

encoder  = OnnxOpusMtEncoder()
decoder  = OnnxOpusMtDecoder()

tokenizer = MarianTokenizer.from_pretrained(model_name)

# Example translation
# src_texts = ["床前明月光， 疑是地上霜。 举头望明月， 低头思故乡。"]
# src_texts = ["今天的天气怎么样？"]
src_texts = ["How is the weather today?"]
inputs = tokenizer(src_texts, return_tensors="pt", padding=True)
print(inputs)

encoder_input_ids = np.zeros([1,MAX_SEQ_LEN_ENC], dtype=np.int32)
encoder_attention_mask = np.zeros([1,MAX_SEQ_LEN_ENC], dtype=np.int32)
for idx in range(inputs.input_ids.shape[1]):
    encoder_input_ids[0,idx] = inputs.input_ids[0,idx]
    encoder_attention_mask[0,idx] = 1

enc_out = encoder.forward(input_ids=encoder_input_ids, encoder_attention_mask=encoder_attention_mask)

# Initialize kv cache
past_key_values = []
for layer_idx in range(NUM_LAYERS):
    if TRANSPOSE_KEY:
        past_key_values.append(np.zeros([enc_out[0].shape[0], enc_out[0].shape[1], enc_out[0].shape[2], MAX_SEQ_LEN_DEC-1], dtype=np.float32))
    else:
        past_key_values.append(np.zeros([enc_out[0].shape[0], enc_out[0].shape[1], MAX_SEQ_LEN_DEC-1, enc_out[0].shape[3]], dtype=np.float32))
    past_key_values.append(np.zeros([enc_out[1].shape[0], enc_out[1].shape[1], MAX_SEQ_LEN_DEC-1, enc_out[1].shape[3]], dtype=np.float32))
    past_key_values.append(enc_out[2*layer_idx])
    past_key_values.append(enc_out[2*layer_idx+1])

decoder_input_ids = np.zeros([1,1], dtype=np.int32)

token = 65000
decoder_input_ids[0,0] = token

tokens = []

for idx in range(MAX_SEQ_LEN_DEC-1):
    logits, *present_key_values = decoder.forward(decoder_input_ids, encoder_attention_mask, np.array(idx).astype(np.int32), *past_key_values)

    token = np.argmax(logits, -1)[0,0]
    # check for end of sequence
    if (token == 0):
        break

    tokens.append(token)

    # set up input for next time
    decoder_input_ids[0,0] = token

    for layer_idx in range(NUM_LAYERS):
        if TRANSPOSE_KEY:
            past_key_values[4*layer_idx][:,:,:,idx:idx+1] = present_key_values[2*layer_idx]
        else:
            past_key_values[4*layer_idx][:,:,idx:idx+1,:] = present_key_values[2*layer_idx]
        past_key_values[4*layer_idx+1][:,:,idx:idx+1,:] = present_key_values[2*layer_idx+1]

print(tokens)
print(tokenizer.decode(tokens, skip_special_tokens=True))