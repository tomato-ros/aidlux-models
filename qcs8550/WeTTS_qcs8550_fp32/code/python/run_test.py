import argparse
import scipy.io.wavfile as wavfile
import time
import re
from cli.model import load_model
import numpy as np
import os


def replace_punctuation(s):
    # 去掉标点符号 文字转语音中识别不了标点符号
    data = re.sub(r'[^\w\s]', ' ', s) 
    data = data.replace(" ","")
    return data
    
def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--text_path', help='text to synthesis')
    parser.add_argument('--wav_path', help='output wav file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = load_model()
    source = args.text_path
    wav_path =args.wav_path
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
    audio_rtf=[]
    all_time=[]
    fronted_time=[]
    vits_time=[]
    num=0
    if source.endswith(".txt"):
        with open(source,"r") as f:
            data_list = f.readlines()
        for i,data in enumerate(data_list):
            data=replace_punctuation(data).strip()
            out_name=f"{wav_path}/{len(data)}_{data[0]}.wav"
            t1=time.time()
            phones, audio ,frontedtime,vitstime= model.synthesis(data)
            t2=time.time()
            wavfile.write(out_name, 16000, audio)
            
            cost_time = (t2-t1)*1000
            audio_len=len(audio)/16000
            rtf = cost_time/audio_len
            audio_rtf.append(rtf)
            avg_token_cost = cost_time/len(data)
            all_time.append(avg_token_cost)
            fronted_time.append(frontedtime)
            vits_time.append(vitstime)
            num+=1  
            
            print(f"tokens :{len(data)},The length of audio ：{audio_len}, cost time： {round(cost_time)} ms , ms/tokens:{round(avg_token_cost)}")

        # 整体输出的时间
        print("================================")
        print("all time :")
        max_time = max(all_time)
        min_time = min((all_time))
        avg_time = sum((all_time))/num
        print(f"AVG time is {round(avg_time,2)}ms/tokens,\n MAX time is {round(max_time,2)}ms/tokens,\n MIN time is {round(min_time,2)}ms/tokens")
    
    else:
        data = replace_punctuation(source).strip()
        all_tokens=[]
        num=0
        for i in range(10):
            t1=time.time()
            phones, audio, frontedtime, vitstime = model.synthesis(data)
            t2=time.time()

            cost_time = (t2-t1)*1000
            audio_len=len(audio)/16000
            rtf = cost_time/audio_len
            audio_rtf.append(rtf)
            all_time.append(cost_time)
            fronted_time.append(frontedtime)
            vits_time.append(vitstime)
            num+=1
            print(f"tokens :{len(data)},The length of audio ：{audio_len}, cost time： {round(cost_time,2)} ms ")

        
        # fronted time
        print("================================")
        print("fronted model time :")
        max_time = max(fronted_time)
        min_time = min((fronted_time))
        avg_time = sum((fronted_time))/num
        print(f"AVG time is {avg_time}ms/{len(data)}words,\n MAX time is {max_time}ms/{len(data)}words,\n MIN time is {min_time}ms/{len(data)}words")

        # vits time
        print("================================")
        print("vits model time :")
        max_time = max(vits_time)
        min_time = min((vits_time))
        avg_time = sum((vits_time))/num
        print(f"AVG time is {avg_time}ms/{len(data)}words,\n MAX time is {max_time}ms/{len(data)}words,\n MIN time is {min_time}ms/{len(data)}words")

        # 整体输出的时间
        print("================================")
        print("all time :")
        max_time = max(all_time)
        min_time = min((all_time))
        avg_time = sum((all_time))/num
        print(f"AVG time is {avg_time}ms/{len(data)}words,\n MAX time is {max_time}ms/{len(data)}words,\n MIN time is {min_time}ms/{len(data)}words")
        
        out_name = f"{wav_path}/{len(data)}_{data[0]}.wav"
        wavfile.write(out_name, 16000, audio)


if __name__ == '__main__':
    main()
