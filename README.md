# GPT-SoVITS-Server

[GITEE](https://gitee.com/utf16/gpt-so-vits-server) github经常不稳定，gitee环境更舒适一些。

感谢GPT-SoVITS项目！
- 对于大多数人的应用场景而言，最主要的还是写应用；
- 从云端训练好的模型下载到本地，本地只是推理即可；（可以CPU或者GPU）
- GPT-SoVITS项目是一个完整的研究项目，作者已经极尽细致的做了WebUI和QT，也提供了api接口，但对应用者而言太过复杂和臃肿。

所以我做了一些粗浅的工作，把原项目中的接口部分提取了出来。
- 对于熟悉fastapi的朋友来说，server.py里面可以自行发挥改写。
- 下载好的模型存放路径，还有一些指令方面的东西，自己看server代码，非常简单；
- 前端套nginx的事情自己去弄，搜索nginx + fastapi
