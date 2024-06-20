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

# 进阶作业

## LMDeploy模型量化

**量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。**

* 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速度。

* 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。

可以使用KV8量化和W4A16量化。KV8量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。W4A16 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。

**设置最大KV Cache缓存大小**

KV Cache是一种缓存技术，通过存储键值对的形式来复用计算结果，以达到提高性能和降低内存消耗的目的。在大规模训练和推理中，KV Cache可以显著减少重复计算量，从而提升模型的推理速度。理想情况下，KV Cache全部存储于显存，以加快访存速度。当显存空间不足时，也可以将KV Cache放在内存，通过缓存管理器控制将当前需要使用的数据放入显存。

模型在运行时，占用的显存可大致分为三部分：模型参数本身占用的显存、KV Cache占用的显存，以及中间运算结果占用的显存。

输入命令，控制KV缓存占用剩余显存的最大比例为0.4

	lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.5
 
![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/bf5e3966-af06-448d-b110-a2389e8a7a92)

此时显存占用仅为6152MB，代价是会降低模型推理速度。

**使用W4A16量化**

安装一个依赖库

	pip install einops==0.7.0
 
执行以下命令完成模型量化工作

	lmdeploy lite auto_awq \
	   /root/internlm2-chat-1_8b \
	  --calib-dataset 'ptb' \
	  --calib-samples 128 \
	  --calib-seqlen 1024 \
	  --w-bits 4 \
	  --w-group-size 128 \
	  --work-dir /root/internlm2-chat-1_8b-4bit
   
我们将KV Cache比例再次调为0.4，查看显存占用情况。

	lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.4

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/18e04577-da9e-4ef7-a6dd-9cef045890f3)

可以看到，显存占用变为4932MB，明显降低。


