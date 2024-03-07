import argparse 
import os
import re
import signal
import sys
import wave
from io import BytesIO 
from time import time as ttime 
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from feature_extractor import cnhubert
from _lib.mel_processing import spectrogram_torch
from _lib.models import SynthesizerTrn
from transformers import AutoModelForMaskedLM, AutoTokenizer 
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

# -------pretrained_models和训练好的模型，丢在下面的位置----------------
cnhubert_path = "./data/pretrained_models/chinese-hubert-base"
bert_path = "./data/pretrained_models/chinese-roberta-wwm-ext-large" 
# 从云端下载回来的训练好的内容
sovits_path = './data/_models/svc/0128-0359_e12_s144.pth'
gpt_path = './data/_models/gpt/0128-0359-e30.ckpt' 
# 推理引用的音频文件地址和文本信息 
default_refer_path =   './data/_models/000.wav'  
default_refer_text =  "云南凤庆给您发货，一斤装四十九，两斤装九十五，三斤装一百二十九，规格越大价格越划算。" 
# 语言，我个人把日语韩语乱七八糟的直接删除了，因为我用不上，大家需要的话自己做适配
default_refer_language = 'zh' 
# 使用cuda就是用英伟达的gpu，cpu就是用cpu
device = 'cpu'  # cpu cuda
is_half = False 
# -----------------------

# 如果要增加更多的参数选项，在这里设定
parser = argparse.ArgumentParser(description="GPT-SoVITS-SERVER") 
# parser.add_argument("-t", "--text", type=str, default='请输入文字，测试合成效果', help=f'.\runtime\python.exe .\api-ben2.py -t "输入要合成的文字"')
# parser.add_argument("-f", "--huashu", type=str, default='', help=f'.\runtime\python.exe .\app.py -f ./huahsu.json')
parser.add_argument("-drp", "--default_refer_path", type=str, default="", help="1/2")
parser.add_argument("-drt", "--default_refer_text", type=str, default="", help="1/2")

parser.add_argument("-p", "--port", type=int, default='8080', help="default: 9880")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")

args = parser.parse_args()

if args.default_refer_path and args.default_refer_text:
    default_refer_path = args.default_refer_path
    default_refer_text = args.default_refer_text

cnhubert.cnhubert_base_path = cnhubert_path
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


# https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api.py
def load_audio(file, sr):
    try: 
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        print(file)
        print( '~~~ os.path.exists(file)', os.path.exists(file) )
        if os.path.exists(file) == False:
            raise RuntimeError(
                "错误：You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e: 
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str): 
    # path_str = path_str.replace('/', '\\') #仅在windows下使用
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


n_semantic = 1024
dict_s2 = torch.load(sovits_path, map_location="cpu")
hps = dict_s2["config"]


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu")
config = dict_s1["config"]
ssl_model = cnhubert.get_model()
if is_half:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
if is_half:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
vq_model.load_state_dict(dict_s2["weight"], strict=False)
# print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
hz = 50
max_sec = config['data']['max_sec']
t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
    return spec

splits = {"，","。","？","！",",",".","?","!","~",":","：","—","…",}
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k=torch.cat([wav16k,zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime() 
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    texts = text.split("\n")
    if(len(get_first(text))<4):text+="。"if text!="en"else "."
    audio_opt = []
    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
        else:
            bert1 = torch.zeros(
                (1024, len(phones1)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
        if text_language == "zh":
            bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    # print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )
 
def tts(  text, filename = "", text_language="zh"):  
    refer_wav_path, prompt_text, prompt_language = (
            default_refer_path,
            default_refer_text,
            default_refer_language,
        ) 
    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language
        )
        sampling_rate, audio_data = next(gen) 
    if filename == '':
        filename = 'result.wav'
    with wave.open( filename ,'wb') as fw:
        fw.setnchannels(1)
        fw.setsampwidth(2)
        fw.setframerate(sampling_rate) 
        fw.writeframesraw(audio_data) 
    torch.cuda.empty_cache()

    
# refer_wav_path, prompt_text, prompt_language,
def handle(command,  text, text_language='zh'): 
    if command == "/restart":
        os.execl('./runtime/python.exe', './runtime/python.exe', *sys.argv)
    elif command == "ping":
        return 'pong'
    elif command == "/exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)

    refer_wav_path, prompt_text, prompt_language = (
            default_refer_path,
            default_refer_text,
            default_refer_language,
        ) 
    with torch.no_grad():
        gen = get_tts_wav(
            refer_wav_path, prompt_text, prompt_language, text, text_language
        )
        sampling_rate, audio_data = next(gen)

    wav = BytesIO()
    sf.write(wav, audio_data, sampling_rate, format="wav")
    wav.seek(0)

    torch.cuda.empty_cache()
    return StreamingResponse(wav, media_type="audio/wav")


app = FastAPI()


@app.post("/")
async def tts_endpoint(request: Request):
    json_post_raw = await request.json()
    return handle( json_post_raw.get("command"),   json_post_raw.get("text"),   )


@app.get("/")
async def tts_endpoint(command: str = None,text: str = None, ):
    print(command , text )
    return handle(command,  text  )


if __name__ == "__main__":
    uvicorn.run(app ,   host='0.0.0.0', port='8080', workers=1)