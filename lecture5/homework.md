- [基础作业](#基础作业)
   - [配置LMDeploy运行环境](#配置LMDeploy运行环境)
   - [与InternLM2-Chat-1.8B模型对话](#与InternLM2-Chat-1.8B模型对话)
- [进阶作业](#进阶作业)

# 基础作业

## 配置LMDeploy运行环境

选择镜像Cuda12.2-conda与10% A100*1GPU来创建开发机

创建conda环境

	studio-conda -t lmdeploy -o pytorch-2.1.2

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/7cee8b7a-8ef5-4be2-a9aa-7d4e10e7d856)

安装LMDeploy

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8acd1286-8e16-4162-9390-ed62bb90736a)


## 与InternLM2-Chat-1.8B模型对话

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
