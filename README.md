# GPT-SoVITS-Server

> 我在一台realme手机上运行了这个项目！
> 如果是ubuntu上进行推理的话，你也可以看[这里](./On-Termux-Ubuntu.md)

下面的废话基本没有必要看
----

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
- 我在本地使用的是源项目提供的windows的runtime运行时环境，所以已经装好了依赖，如果有使用相同环境的，直接使用`../runtime/python.exe ./server.py`即可。 
- 切记下载[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)放在server.py同级目录；本项目已包含，linux下不需要，删除该文件即可
- server.py里面clean_path函数一定要看注释，修改一下；

### 云服务器 或者 自己的安卓手机
看文章开始的提示，直接看[这里](./On-Termux-Ubuntu.md)
 