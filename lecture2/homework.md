# contents
- [基础作业](#基础作业)
   - [使用 InternLM2-Chat-1.8B 模型生成 300 字的小故事](#使用InternLM2-Chat-1.8B模型生成300字的小故事)
   - [使用书生·浦语Web和浦语对话](#使用书生·浦语Web和浦语对话)
- [进阶作业](#进阶作业)
   - [熟悉 huggingface 下载功能，使用huggingface_hub python包](#熟悉huggingface下载功能，使用huggingface_hub_python包)
   - [浦语·灵笔2的图文创作及视觉问答部署](#浦语·灵笔2的图文创作及视觉问答部署)
   - [Lagent工具调用_数据分析_Demo部署](#Lagent工具调用_数据分析_Demo部署)
      - [相关知识](#相关知识)
      - [相关操作](#相关知识)
     
# 基础作业

## 使用InternLM2-Chat-1.8B模型生成 300字的小故事

操作截图见笔记部分

<img width="987" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/fefb1c8f-0d3b-4b96-b81f-df3ea28b54ca">

## 使用书生·浦语Web和浦语对话

<img width="922" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/de46637f-de7c-4ac7-82a0-8de34007048a">

<img width="913" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/bdec1900-8f6b-413f-8671-491b480f4dcf">

这一部分我觉得不好，虽然能理解LLM在写输出数学公式的时候因为只有语言模态的问题只能输出latex格式的数学公式文本，但是还是觉得这是LLM的缺陷，像gpt3.5也这种问题，我个人认为能否将latex语言的数学公式与图片对齐，使其输出图片，或许是一种更好的方式。

<img width="922" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/1b9373c0-434f-4775-870d-8bd726031743">

这里我想要说的SAM是 segment anything model这篇论文，应该是缩写的缘故，导致有歧义。

<img width="922" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/95129fd9-7d4f-42ae-9728-0b249b069b41">

# 进阶作业

## 熟悉huggingface下载功能，使用huggingface_hub_python包

## 浦语·灵笔2的图文创作及视觉问答部署


## Lagent工具调用_数据分析_Demo部署

### 相关知识

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/76bfe70b-6b32-4e51-b17c-526f7d39176e)

Lagent 的特性总结如下：

* 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。

* 接口统一，设计全面升级，提升拓展性，包括：
  
	* Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
   
	* Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
   
	* Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
   
* 文档全面升级，API 文档全覆盖。

### 相关操作

激活demo环境后，在demo文件夹下克隆仓库
	conda activate demo
 	cd /root/demo
  	
	git clone https://gitee.com/internlm/lagent.git
	# git clone https://github.com/internlm/lagent.git
	cd /root/demo/lagent
	git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
	pip install -e . # 源码安装

<img width="1217" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/83a41a04-c121-4cf4-bc3f-e0fa2480c0b1">

在lagent文件夹中构造软链接快捷访问方式

	cd /root/demo/lagent
	
	ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
 
打开 lagent 路径下 examples/internlm2_agent_web_demo_hf.py 文件，并修改对应位置 (71行左右) 代码：

	# 其他代码...
	value='/root/models/internlm2-chat-7b'
	# 其他代码...
 
<img width="485" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/19c06499-02bd-429e-82cc-8b075d9b7cba">

运行一下命令以加载模型

	streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006

对本地端口环境配置本地 PowerShell，并替换端口号，输入密码

	# 从本地使用 ssh 连接 studio 端口
	# 将下方端口号 38374 替换成自己的端口号
	ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374

<img width="863" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/2209387c-cfdc-47bd-aeda-3c2f9d49432b">

让其计算1000以内质数的和，计算时间有点长

<img width="878" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/7b7bd9da-042c-4be6-abd9-30e5a5aee81d">

这里我让它求ln(x)的泰勒展开式，它告诉我说要具体的数字

<img width="876" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/f6fa4b06-c7ca-4937-ae6c-de5edcec89c6">

更换为具体的数字之后还是没弄懂我的意思。。。
<img width="886" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/06e504ed-0449-4b0e-88eb-45011885e941">


