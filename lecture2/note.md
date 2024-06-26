# 第2节课·轻松分钟玩转书生·浦语大模型趣味Demo(笔记）

[视频地址](https://www.bilibili.com/video/BV1AH4y1H78d/)|[官方文档](https://github.com/InternLM/Tutorial/blob/camp2/helloworld/hello_world.md)

<img width="922" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/b16420e2-a1eb-460a-9ae5-584508bfd281">

# contents
- [实战部署InternLM2-Chat-1.8B](#实战部署InternLM2-Chat-1.8B)
- [实战部署优秀作品八戒-Chat-1.8B](#实战部署优秀作品八戒--Chat-1.8B)
- [实战进阶运行Lagent智能体Demo](#实战进阶运行Lagent智能体Demo)
- [实战进阶灵笔InternLM-XCompser2](#实战进阶灵笔InternLM-XCompser2)

# 实战部署InternLM2-Chat-1.8B

<img width="377" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/3c95bd75-2201-47d9-b5a9-0e19a42a4642">

使用 Cuda11.7-conda 镜像，10% A100 * 1创建开发机。

配置环境

	studio-conda -o internlm-base -t demo

激活环境并完成环境包的安装

	conda activate demo
	pip install huggingface-hub==0.17.3
	pip install transformers==4.34 
	pip install psutil==5.9.8
	pip install accelerate==0.24.1
	pip install streamlit==1.32.2 
	pip install matplotlib==3.8.3 
	pip install modelscope==1.9.5
	pip install sentencepiece==0.1.99

到/root/demo/download_mini.py文件下复制命令并执行，以下载internlm2-chat-1_8b

	import os
	from modelscope.hub.snapshot_download import snapshot_download
	
	# 创建保存模型目录
	os.system("mkdir /root/models")
	
	# save_dir是模型保存到本地的目录
	save_dir="/root/models"
	
	snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
	                  cache_dir=save_dir, 
	                  revision='v1.1.0')


复制以下代码到/root/demo/cli_demo.py文件并运行

	import torch
	from transformers import AutoTokenizer, AutoModelForCausalLM
	
	
	model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
	
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
	model = model.eval()
	
	system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
	- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
	- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
	"""
	
	messages = [(system_prompt, '')]
	
	print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")
	
	while True:
	    input_text = input("\nUser  >>> ")
	    input_text = input_text.replace(' ', '')
	    if input_text == "exit":
	        break
	
	    length = 0
	    for response, _ in model.stream_chat(tokenizer, input_text, messages):
	        if response is not None:
	            print(response[length:], flush=True, end="")
	            length = len(response)

之后便可以创作小故事

<img width="987" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/9c2a5166-6c38-4c68-92a6-08c070eb3661">

# 实战部署优秀作品八戒--Chat-1.8B

<img width="382" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/1a3db9f4-f973-409e-a95f-2789adf0aeb4">

首先激活demo环境，并使用以下命令来获得仓库内的 Demo 文件

	cd /root/
	git clone https://gitee.com/InternLM/Tutorial -b camp2
	# git clone https://github.com/InternLM/Tutorial -b camp2
	cd /root/Tutorial

执行命令运行bajie_download.py文件

	python /root/Tutorial/helloworld/bajie_download.py
 
<img width="977" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/73d05448-79ac-485f-9678-a25fc4b288de">

待下载完成后输入以下命令：

	streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
 
配置本地 PowerShell，输入

	ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 自己的端口号
 
 <img width="836" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/2b3c656a-8465-4ebe-adda-1fb019267638">

 键入密码，并打开[](http://127.0.0.1:6006)即可开启对话

 <img width="636" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/7c7585f4-b88c-4720-a20b-2a21d252f705">

<img width="553" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8027e6f1-050a-47f4-a5b8-68d228351a73">

<img width="558" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/30854890-9909-46ef-a4b9-2a3128be71b9">

# 实战进阶运行Lagent智能体Demo
见homework部分

<img width="362" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/feaf1bb6-3217-45e1-a4d3-45930a64fe4c">


# 实战进阶灵笔InternLM-XCompser2
见homework部分

<img width="341" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/58a316ac-dc73-4c92-836c-d43eb4264989">
