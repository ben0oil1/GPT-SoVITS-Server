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


### termux - UbuntuLts
```
# 增加快捷命令
nano ~/.bashrc
# 写入
alias lts='proot-distro login ubuntu'
# 保存生效
source ~/.bashrc
# 进入ubuntu
lts
```
ubuntu 中的操作：
```
# 安装python3.10
apt insall wget python3 python3-dev python3-venv libsndfile1
# libsndfile1 一定要安装，否则后面soundfile中会出现错误：cannot load library ‘libsndfile.so‘: libsndfile.so: cannot open shared object file: No such file or

python3 -V
# Python 3.10.12

wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip -V
# pip 24.0 from /usr/local/lib/python3.10/dist-packages/pip (python 3.10)
# 直接apt install python3-pip的时候，会出现一些错误，

```

`pip install ffmpeg soundfile tqdm scipy cn2an pypinyin torchaudio transformers jieba_fast jieba PyYAML pytorch-lightning` 
> Installing collected packages: mpmath, jieba_fast, jieba, ffmpeg, urllib3, typing-extensions, tqdm, sympy, safetensors, regex, pypinyin, pycparser, proces, packaging, numpy, networkx, multidict, MarkupSafe, idna, fsspec, frozenlist, filelock, charset-normalizer, certifi, attrs, async-timeout, yarl, scipy, requests, lightning-utilities, jinja2, cn2an, cffi, aiosignal, torch, soundfile, huggingface-hub, aiohttp, torchmetrics, torchaudio, tokenizers, transformers, pytorch-lightning
> 
> Successfully installed MarkupSafe-2.1.5 aiohttp-3.9.3 aiosignal-1.3.1 async-timeout-4.0.3 attrs-23.2.0 certifi-2024.2.2 cffi-1.16.0 charset-normalizer-3.3.2 cn2an-0.5.22 ffmpeg-1.4 filelock-3.13.1 frozenlist-1.4.1 fsspec-2024.2.0 huggingface-hub-0.21.4 idna-3.6 jieba-0.42.1 jieba_fast-0.53 jinja2-3.1.3 lightning-utilities-0.10.1 mpmath-1.3.0 multidict-6.0.5 networkx-3.2.1 numpy-1.26.4 packaging-23.2 proces-0.1.7 pycparser-2.21 pypinyin-0.50.0 pytorch-lightning-2.2.1 regex-2023.12.25 requests-2.31.0 safetensors-0.4.2 scipy-1.12.0 soundfile-0.12.1 sympy-1.12 tokenizers-0.15.2 torch-2.2.1 torchaudio-2.2.1 torchmetrics-1.3.1 tqdm-4.66.2 transformers-4.38.2 typing-extensions-4.10.0 urllib3-2.2.1 yarl-1.9.4

`pip install librosa==0.9.2`
> Installing collected packages: threadpoolctl, llvmlite, joblib, decorator, audioread, scikit-learn, pooch, numba, resampy, librosa
> Successfully installed audioread-3.0.1 decorator-5.1.1 joblib-1.3.2 librosa-0.9.2 llvmlite-0.42.0 numba-0.59.0 pooch-1.8.1 resampy-0.4.3 scikit-learn-1.4.1.post1 threadpoolctl-3.3.0

`pip install numba==0.56.4`
> Installing collected packages: numpy, llvmlite, numba
>   Attempting uninstall: numpy
>     Found existing installation: numpy 1.26.4
>     Uninstalling numpy-1.26.4:
>       Successfully uninstalled numpy-1.26.4
>   Attempting uninstall: llvmlite
>     Found existing installation: llvmlite 0.42.0
>     Uninstalling llvmlite-0.42.0:
>       Successfully uninstalled llvmlite-0.42.0
>   Attempting uninstall: numba
>     Found existing installation: numba 0.59.0
>     Uninstalling numba-0.59.0:
>       Successfully uninstalled numba-0.59.0
> Successfully installed llvmlite-0.39.1 numba-0.56.4 numpy-1.23.5

` pip install uvicorn`
> Installing collected packages: h11, click, uvicorn
> Successfully installed click-8.1.7 h11-0.14.0 uvicorn-0.27.1

`pip install matplotlib`
> Installing collected packages: six, pyparsing, pillow, kiwisolver, fonttools, cycler, contourpy, python-dateutil, matplotlib
> Successfully installed contourpy-1.2.0 cycler-0.12.1 fonttools-4.49.0 kiwisolver-1.4.5 matplotlib-3.8.3 pillow-10.2.0 pyparsing-3.1.2 python-dateutil-2.9.0.post0 six-1.16.0

`pip install fastapi`
> Installing collected packages: sniffio, pydantic-core, exceptiongroup, annotated-types, pydantic, anyio, starlette, fastapi
> Successfully installed annotated-types-0.6.0 anyio-4.3.0 exceptiongroup-1.2.0 fastapi-0.110.0 pydantic-2.6.3 pydantic-core-2.16.3 sniffio-1.3.1 starlette-0.36.3

`pip install einops`

~~ 在load_audio中可能出现错误，必须先卸载之前安装的ffmpeg：`pip uninstall ffmpeg`，重新安装`ffmpeg-python` ~~
~~`pip install ffmpeg-python`~~
> ~~Installing collected packages: future, ffmpeg-python~~
> ~~Successfully installed ffmpeg-python-0.2.0 future-1.0.0~~

ubuntu必须先安装好ffmpeg

这个
### 电脑文件互访
先必须授予[termux文件的访问权限](https://wiki.termux.com/wiki/Termux-setup-storage):`termux-setup-storage` 
然后在termux中执行命令：`ls`，可以看到多了一个`storage`的文件夹。
电脑连接手机，访问到的目录位于Ubuntu中的 : `/data/data/com.termux/files/home/storage/shared` ,位于termux中的 `storage/shared`


## ffmpeg<7
下载低版本的ffmpeg：https://johnvansickle.com/ffmpeg/ 手机上下载：https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz
安装方式在[这里](https://www.johnvansickle.com/ffmpeg/faq/)
```bash
tar xvf ffmpeg-release-arm64-static.tar.xz
ls ffmpeg-git-20180203-amd64-static
# ffmpeg  ffprobe  GPLv3.txt  manpages  model  qt-faststart  readme.txt
# mv ffmpeg-git-20180203-amd64-static/ffmpeg /usr/local/bin/ffmpeg
# mv ffmpeg-git-20180203-amd64-static/ffprobe /usr/local/bin/ffprobe
# chmod +x /usr/local/bin/ffmpeg
# chmod +x /usr/local/bin/ffprobe
# 我的手机上应该是/usr/bin/ffmpeg
mv ffmpeg-git-20180203-amd64-static/ffmpeg /usr/bin/ffmpeg
chmod +x /usr/bin/ffmpeg
```
文件路径的问题，如果实在windows，server.py中的87行位置，要启用一下，如果是ubuntu的话，这个地方去掉。