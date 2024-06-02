# contents
- [基础作业](#基础作业)
   - [环境安装](#环境安装)
   - [前期准备](#前期准备)
   - [配置文件修改](#配置文件修改)
   - [模型训练](#模型训练)
   - [模型转换、整合、测试及部署](#模型转换、整合、测试及部署)
- [进阶作业](#进阶作业)

# 基础作业

**xutuner运行原理**

<img width="739" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/9c01d40b-8847-4a6e-ad2e-6d3a5612cb41">
	
	studio-conda xtuner0.1.17
	# 激活环境
	conda activate xtuner0.1.17
	# 进入家目录 （~的意思是 “当前用户的home路径”）
	cd ~
	# 创建版本文件夹并进入，以跟随本教程
	mkdir -p /root/xtuner0117 && cd /root/xtuner0117
	
	# 拉取 0.1.17 的版本源码
	git clone -b v0.1.17  https://github.com/InternLM/xtuner
	# 无法访问github的用户请从 gitee 拉取:
	# git clone -b v0.1.15 https://gitee.com/Internlm/xtuner
	
	# 进入源码目录
	cd /root/xtuner0117/xtuner
	
	# 从源码安装 XTuner
	pip install -e '.[all]'
 
## 前期准备

**数据集准备**

为了让模型能够让模型认清使用者的身份，知道在询问自己是谁的时候回复成我们想要的样子，就需要通过在微调数据集中大量掺杂这部分的数据。

创建一个文件夹来存放我们这次训练所需要的所有文件。
	
	mkdir -p /root/ft && cd /root/ft
	
	# 在ft这个文件夹里再创建一个存放数据的data文件夹
	mkdir -p /root/ft/data && cd /root/ft/data
 
在 data 目录下新建一个 generate_data.py 文件

	# 创建 `generate_data.py` 文件
	touch /root/ft/data/generate_data.py
 
将以下代码复制进去，然后运行该脚本即可生成数据集。把n的值调大可以让他能够完完全全识别身份。

	import json
	
	# 设置用户的名字
	name = '不要姜葱蒜大佬'
	# 设置需要重复添加的数据次数
	n =  10000
	
	# 初始化OpenAI格式的数据结构
	data = [
	    {
	        "messages": [
	            {
	                "role": "user",
	                "content": "请做一下自我介绍"
	            },
	            {
	                "role": "assistant",
	                "content": "我是{}的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦".format(name)
	            }
	        ]
	    }
	]
	
	# 通过循环，将初始化的对话数据重复添加到data列表中
	for i in range(n):
	    data.append(data[0])
	
	# 将data列表中的数据写入到一个名为'personal_assistant.json'的文件中
	with open('personal_assistant.json', 'w', encoding='utf-8') as f:
	    # 使用json.dump方法将数据以JSON格式写入文件
	    # ensure_ascii=False 确保中文字符正常显示
	    # indent=4 使得文件内容格式化，便于阅读
	    json.dump(data, f, ensure_ascii=False, indent=4)

## 配置文件修改

## 模型训练

## 模型转换、整合、测试及部署

# 进阶作业


