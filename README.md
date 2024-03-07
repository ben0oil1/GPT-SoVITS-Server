# GPT-SoVITS-Server

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)项目是一个目前为止最优秀的语音克隆项目。
很多朋友使用[云端的模型训练工具](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official)完成声音克隆的模型训练，推理合成的时候，其实不需要把完整的项目都克隆到本地，其实只需要从云端下载好模型文件，然后丢在本地或者服务器上即可。
训练模型对于算力要求很高，但是推理合成其实可以用CPU也行——绝大多数的服务器其实都是CPU计算，GPU服务器太贵了。

自身也懒得在很复杂的文件体系中做配置，索性从源项目提取出来了这个项目，核心内容在`server.py`里面，大家可以修改里面的配置，然后使用即可。

> 这个版本删除了日语和英语，纯粹的中文。

## 系统环境：
### 依赖
pretrained_models下载地址：https://huggingface.co/lj1995/GPT-SoVITS/tree/main 
把：`chinese-hubert-base` ,`chinese-roberta-wwm-ext-large`下载后丢到本地，记得修改server.py里面的路径即可。

### windows
我在本地使用的是源项目提供的windows的runtime运行时环境，所以已经装好了依赖，如果有使用相同环境的，直接使用`../runtime/python.exe ./server.py`即可。
图方便省心部署云服务器的话，也可以用这个，官方打包好的内容，只要里面的[runtime](https://gitee.com/utf16/gpt-so-vits-server/raw/master/runtime.zip)即可
切记下载[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)放在server.py同级目录；本项目已包含，linux下不需要，删除该文件即可

### Linux
很多云服务器上都是linux，其实只需要搭建所需环境接口： 官方[requirements.txt](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/requirements.txt)这个里面包含模型训练的完整环境，我项目里面提供的requirements理论上只有推理所需的依赖，但是未经测试，需要的自己测试。

### termux
实验并不成功，termux环境过于特殊，在安装numpy的时候用apt解决了，系统自动安装的py3.11跟numba(librosa源码安装的0.9.2，依赖于numba)不适配。可能曲线救国的方式是termux里面安装完整的Linux环境，或可解决问题。
```
uvicorn ffmpeg librosa numpy  soundfile fastapi feature_extractor transformers typing einops tqdm scipy 
pip install  pypinyin jieba_fast   contextlib3 gruut typeguard pyaml cn2an 
apt install matplotlib pybind11 pystring python-ensurepip-wheels python-future python-numpy python-lxml python-pandas python-scipy python-static 
```
## 使用：
如果是windows的话，直接用，我自己测试了，可以正常运行。
如果自己要进一步魔改，配置到云服务器的话，所有的核心关键都在server.py里面。

- 对于熟悉fastapi的朋友来说，server.py里面可以自行发挥改写。
- 下载好的模型存放路径，还有一些指令方面的东西，自己看server代码，非常简单；
- 前端套nginx的事情自己去弄，搜索nginx + fastapi
