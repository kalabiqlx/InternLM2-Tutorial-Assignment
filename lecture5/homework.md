- [基础作业](#基础作业)
   - [配置LMDeploy运行环境](#配置LMDeploy运行环境)
   - [与InternLM2-Chat-1.8B模型对话](#与InternLM2-Chat-1.8B模型对话)
- [进阶作业](#进阶作业)
   - [LMDeploy模型量化](#LMDeploy模型量化)
   - [LMDeploy服务](#LMDeploy服务)
   - [Python代码集成](#Python代码集成)
   - [使用LMDeploy运行视觉多模态大模型llava](#使用LMDeploy运行视觉多模态大模型llava)

# 基础作业

## 配置LMDeploy运行环境

选择镜像Cuda12.2-conda与10% A100*1GPU来创建开发机

创建conda环境

	studio-conda -t lmdeploy -o pytorch-2.1.2

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/7cee8b7a-8ef5-4be2-a9aa-7d4e10e7d856)

安装LMDeploy

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8acd1286-8e16-4162-9390-ed62bb90736a)


## 与InternLM2-Chat-1.8B模型对话

**使用Transformer库运行模型**

首先cd到Home目录，把开发机的共享目录中准备好了常用的预训练模型copy到Home目录下

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/d8bfabd4-b878-4473-a53d-3840cb78047f)

在终端中输入如下指令，新建pipeline_transformer.py。

	touch /root/pipeline_transformer.py
 
 将以下内容复制粘贴进入pipeline_transformer.py。
 
	 import torch
	from transformers import AutoTokenizer, AutoModelForCausalLM
	
	tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)
	
	# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
	model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
	model = model.eval()
	
	inp = "hello"
	print("[INPUT]", inp)
	response, history = model.chat(tokenizer, inp, history=[])
	print("[OUTPUT]", response)
	
	inp = "please provide three suggestions about time management"
	print("[INPUT]", inp)
	response, history = model.chat(tokenizer, inp, history=history)
	print("[OUTPUT]", response)

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/07288214-7694-4ca0-ac46-cc6c7ea47d1a)

 之后激活conda环境并且运行python代码

 ![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/82e2e0f3-b810-4bc4-9702-00f9f5f672b9)

**使用LMDeploy与模型对话**

使用LMDeploy与模型进行对话的通用命令格式为：

	lmdeploy chat [HF格式模型路径/TurboMind格式模型路径]

运行下载的1.8B模型:

	lmdeploy chat /root/internlm2-chat-1_8b

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/b399b3a4-a528-4f38-8997-63579602e990)


![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/150b6ec0-8c41-4831-a7a6-9d83cf18cf43)

