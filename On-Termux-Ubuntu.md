# 在手机上运行GPT-SoVITS推理工具

- 这个版本删除了日语和英语，纯粹的中文。如果有需要，在text下查看文件内容，根据官方自己补全一下即可；
- 安装termux，并执行如下操作：
    ```
    # 安装依赖
    pkg install proot proot-distro
    # 安装ubuntu
    proot-distro install ubuntu-lts
    # 进入ubuntu
    proot-distro login ubuntu

    # termux中增加快捷命令，没有必要，只是为了方便；
    nano ~/.bashrc
    # 写入
    alias lts='proot-distro login ubuntu'
    # 保存生效
    source ~/.bashrc
    # 进入ubuntu
    lts
    ```
- 在termux中安装ubuntu-lts(22.04)
- 安装并运行项目的细节:
    ```
    # 安装python3.10
    apt insall wget python3 python3-dev python3-venv libsndfile1
    # libsndfile1 一定要安装，否则后面soundfile中会出现错误：cannot load library ‘libsndfile.so‘: libsndfile.so: cannot open shared object file: No such file or

    python3 -V
    # Python 3.10.12

    # 直接apt install python3-pip的时候，会出现一些错误，用下面的方法安装pip
    wget https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py
    pip -V
    # pip 24.0 from /usr/local/lib/python3.10/dist-packages/pip (python 3.10) 

    ```
- 克隆仓库中的代码，安装依赖；这个是成功调试的版本：
    ```
    pip list
    Package             Version
    ------------------- -------------
    aiohttp             3.9.3
    aiosignal           1.3.1
    annotated-types     0.6.0
    anyio               4.3.0
    async-timeout       4.0.3
    attrs               23.2.0
    audioread           3.0.1
    blinker             1.4
    certifi             2024.2.2
    cffi                1.16.0
    charset-normalizer  3.3.2
    click               8.1.7
    cn2an               0.5.22
    contourpy           1.2.0
    cryptography        3.4.8
    cycler              0.12.1
    dbus-python         1.2.18
    decorator           5.1.1
    distro              1.7.0
    distro-info         1.1+ubuntu0.2
    einops              0.7.0
    exceptiongroup      1.2.0
    fastapi             0.110.0
    feature-extractor   0.0.1
    ffmpeg-python       0.1.18
    filelock            3.13.1
    fonttools           4.49.0
    frozenlist          1.4.1
    fsspec              2024.2.0
    future              1.0.0
    h11                 0.14.0
    httplib2            0.20.2
    huggingface-hub     0.21.4
    idna                3.6
    importlib-metadata  4.6.4
    jeepney             0.7.1
    jieba               0.42.1
    jieba-fast          0.53
    Jinja2              3.1.3
    joblib              1.3.2
    keyring             23.5.0
    kiwisolver          1.4.5
    launchpadlib        1.10.16
    lazr.restfulclient  0.14.4
    lazr.uri            1.0.6
    librosa             0.9.2
    lightning-utilities 0.10.1
    llvmlite            0.39.1
    MarkupSafe          2.1.5
    matplotlib          3.8.3
    more-itertools      8.10.0
    mpmath              1.3.0
    multidict           6.0.5
    networkx            3.2.1
    numba               0.56.4
    numpy               1.23.5
    oauthlib            3.2.0
    packaging           23.2
    pillow              10.2.0
    pip                 24.0
    platformdirs        2.5.1
    pooch               1.8.1
    proces              0.1.7
    pycparser           2.21
    pydantic            2.6.3
    pydantic_core       2.16.3
    PyGObject           3.42.1
    PyJWT               2.3.0
    pyparsing           3.1.2
    pypinyin            0.50.0
    python-apt          2.4.0+ubuntu3
    python-dateutil     2.9.0.post0
    pytorch-lightning   2.2.1
    PyYAML              6.0.1
    regex               2023.12.25
    requests            2.31.0
    resampy             0.4.3
    safetensors         0.4.2
    scikit-learn        1.4.1.post1
    scipy               1.12.0
    SecretStorage       3.3.1
    setuptools          59.6.0
    six                 1.16.0
    sniffio             1.3.1
    soundfile           0.12.1
    starlette           0.36.3
    sympy               1.12
    threadpoolctl       3.3.0
    tokenizers          0.15.2
    torch               2.2.1
    torchaudio          2.2.1
    torchmetrics        1.3.1
    tqdm                4.66.2
    transformers        4.38.2
    typing_extensions   4.10.0
    unattended-upgrades 0.1
    urllib3             2.2.1
    uvicorn             0.27.1
    wadllib             1.3.6
    wheel               0.37.1
    yarl                1.9.4
    zipp                1.0.0
    ```
- 将自己训练好的模型根据server.py中的提示放置；
- 对于熟悉fastapi的朋友来说，server.py里面可以自行发挥改写。
- 前端套nginx的事情自己去弄，搜索nginx + fastapi 
- 下载好的模型存放路径，还有一些指令方面的东西，自己看server代码，非常简单；



##  手工操作记录
ubuntu 中的操作： 
```
pip install soundfile tqdm scipy cn2an pypinyin torchaudio transformers jieba_fast jieba PyYAML pytorch-lightning  
pip install librosa==0.9.2
pip install numba==0.56.4
pip install uvicorn matplotlib fastapi einops ffmpeg-python
```
./ffmpeg 是我编译好的适合手机的ffmpeg6.0 版本，用的时候不用apt安装ffmpeg了(版本高于7.0会报错)，移动到Ubuntu的/usr/local/bin下即可(要授予执行权限：chmod +x /usr/bin/ffmpeg). 

### 电脑文件互访
1. 用USB连接电脑和手机，打开USB文件传输，
2. 先必须授予[termux文件的访问权限](https://wiki.termux.com/wiki/Termux-setup-storage):`termux-setup-storage` 
3. 然后在termux中执行命令：`ls`，可以看到多了一个`storage`的文件夹。
4. 电脑连接手机，访问到的目录位于Ubuntu中的 : `/data/data/com.termux/files/home/storage/shared` ,位于termux中的 `storage/shared`

## 源码中环境差异
在server.py中，clean_path函数中，需要的时候改一下，具体看注释； 

手机Ubuntu环境中的目录情况：
```
root@localhost:/data/data/com.termux/files/home/storage/shared/GPT-SoVITS-Server# tree ./
./
├── 000.wav
├── AR
│   ├── data
│   │   ├── bucket_sampler.py
│   │   ├── data_module.py
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── t2s_lightning_module.cpython-310.pyc
│   │   │   ├── t2s_model.cpython-310.pyc
│   │   │   └── utils.cpython-310.pyc
│   │   ├── t2s_lightning_module.py
│   │   ├── t2s_model.py
│   │   └── utils.py
│   ├── modules
│   │   ├── activation.py
│   │   ├── embedding.py
│   │   ├── __init__.py
│   │   ├── lr_schedulers.py
│   │   ├── optim.py
│   │   ├── patched_mha_with_cache.py
│   │   ├── __pycache__
│   │   │   ├── activation.cpython-310.pyc
│   │   │   ├── embedding.cpython-310.pyc
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── lr_schedulers.cpython-310.pyc
│   │   │   ├── optim.cpython-310.pyc
│   │   │   ├── patched_mha_with_cache.cpython-310.pyc
│   │   │   ├── scaling.cpython-310.pyc
│   │   │   └── transformer.cpython-310.pyc
│   │   ├── scaling.py
│   │   └── transformer.py
│   ├── __pycache__
│   │   └── __init__.cpython-310.pyc
│   ├── text_processing
│   │   ├── __init__.py
│   │   ├── phonemizer.py
│   │   └── symbols.py
│   └── utils
│       ├── initialize.py
│       ├── __init__.py
│       └── io.py
├── data
│   ├── _models
│   │   ├── 000.wav
│   │   ├── gpt
│   │   │   └── 0128-0359-e30.ckpt
│   │   └── svc
│   │       └── 0128-0359_e12_s144.pth
│   └── pretrained_models
│       ├── chinese-hubert-base
│       │   ├── config.json
│       │   ├── preprocessor_config.json
│       │   └── pytorch_model.bin
│       └── chinese-roberta-wwm-ext-large
│           ├── config.json
│           ├── pytorch_model.bin
│           └── tokenizer.json
├── feature_extractor
│   ├── cnhubert.py
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── cnhubert.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   └── whisper_enc.cpython-310.pyc
│   └── whisper_enc.py
├── ffmpeg
├── fm-test.py
├── _lib
│   ├── attentions.py
│   ├── commons.py
│   ├── core_vq.py
│   ├── mel_processing.py
│   ├── models.py
│   ├── modules.py
│   ├── mrte_model.py
│   ├── __pycache__
│   │   ├── attentions.cpython-310.pyc
│   │   ├── attentions.cpython-39.pyc
│   │   ├── commons.cpython-310.pyc
│   │   ├── commons.cpython-39.pyc
│   │   ├── core_vq.cpython-310.pyc
│   │   ├── core_vq.cpython-39.pyc
│   │   ├── mel_processing.cpython-310.pyc
│   │   ├── mel_processing.cpython-39.pyc
│   │   ├── models.cpython-310.pyc
│   │   ├── models.cpython-39.pyc
│   │   ├── modules.cpython-310.pyc
│   │   ├── modules.cpython-39.pyc
│   │   ├── mrte_model.cpython-310.pyc
│   │   ├── mrte_model.cpython-39.pyc
│   │   ├── quantize.cpython-310.pyc
│   │   ├── quantize.cpython-39.pyc
│   │   ├── transforms.cpython-310.pyc
│   │   ├── transforms.cpython-39.pyc
│   │   └── utils.cpython-310.pyc
│   ├── quantize.py
│   ├── transforms.py
│   └── utils.py
├── LICENSE
├── __pycache__
│   ├── server.cpython-310.pyc
│   └── utils.cpython-310.pyc
├── README.md
├── requirements.txt
├── server.py
├── text
│   ├── chinese.py
│   ├── cleaner.py
│   ├── cmudict_cache.pickle
│   ├── cmudict.rep
│   ├── __init__.py
│   ├── opencpop-strict.txt
│   ├── __pycache__
│   │   ├── chinese.cpython-310.pyc
│   │   ├── cleaner.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   ├── symbols.cpython-310.pyc
│   │   └── tone_sandhi.cpython-310.pyc
│   ├── symbols.py
│   └── tone_sandhi.py
└── utils.py
```