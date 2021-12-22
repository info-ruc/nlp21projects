## 2021-NLP大作业 

黄钊恒 2018202064

***
- datasets: PChatbot

        url: https://github.com/qhjqhj00/SIGIR2021-Pchatbot

        src: weibo-based-bert

- 代码流程：

        见slides.pdf

- directory

1. preprocess/: 预处理phase2的用户、语句sampling；生成json文件、bert-pretrain、infer全流程

2. scripts/: 运行程序（可更改内部参数）

3. README.md: 本文件

4. runModel.py: 项目运行入口

5. seq2seq/: 端到端模型，内含dataset/下重写Datasets的子类；models/下包含phase1~4的所有Model代码；utils/下包含常用函数

6. slides.pdf: 课堂展示文件，包含代码流程