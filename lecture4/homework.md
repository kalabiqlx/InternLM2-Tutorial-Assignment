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
	name = 'yihui大佬'
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
     
这里将 name = '不要姜葱蒜大佬' 修改为 name = 'yihui大佬'，之后运行该文件。可以看到在data的路径下便生成了一个名为 personal_assistant.json 的文件，里面就包含了 5000 条 input 和 output 的数据对。

<img width="406" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/fbcce2cf-31ce-4703-bc1c-ce671d117416">

**模型准备**

通过以下代码一键创建文件夹并将所有文件复制进去。

	# 创建目标文件夹，确保它存在。
	# -p选项意味着如果上级目录不存在也会一并创建，且如果目标文件夹已存在则不会报错。
	mkdir -p /root/ft/model
	
	# 复制内容到目标文件夹。-r选项表示递归复制整个文件夹。
	cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/ft/model/
 
 <img width="846" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/117fc313-2b64-4ef3-a302-573d6346cd00">

**配置文件选择**

配置文件（config），其实是一种用于定义和控制模型训练和测试过程中各个方面的参数和设置的工具。准备好的配置文件只要运行起来就代表着模型就开始训练或者微调了。

XTuner 提供多个开箱即用（意味着假如能够连接上 Huggingface 以及有足够的显存，其实就可以直接运行这些配置文件，XTuner就能够直接下载好这些模型和数据集然后开始进行微调）的配置文件，用户可以通过下列命令查看：

	# 列出所有内置配置文件
	# xtuner list-cfg
	
	# 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
	xtuner list-cfg -p internlm2_1_8b
 
 <img width="419" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/1085cc43-4863-448f-861b-011e4714decf">

虽然我们用的数据集并不是 alpaca 而是我们自己通过脚本制作的小助手数据集 ，但是由于我们是通过 QLoRA 的方式对 internlm2-chat-1.8b 进行微调。而最相近的配置文件应该就是 internlm2_1_8b_qlora_alpaca_e3 ，因此我们可以选择拷贝这个配置文件到当前目录：

	# 创建一个存放 config 文件的文件夹
	mkdir -p /root/ft/config
	
	# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
	xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config

 XTuner 工具箱中的第二个工具 copy-cfg ，该工具有两个必须要填写的参数 {CONFIG_NAME} 和 {SAVE_PATH}，能够把 config 文件复制到指定的位置
 
 <img width="845" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/c3e71c2c-5c9d-414f-880a-a17a116b7281">

切记在微调的时候最重要的还是要自己准备一份高质量的数据集，这个才是你能否真微调出效果最核心的利器。如炼丹的材料（就是数据集）本来就是垃圾，那无论怎么炼（微调参数的调整），炼多久（训练的轮数），炼出来的东西还只能且只会是垃圾。

## 配置文件修改

将以下代码复制到 /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py 文件中

	# Copyright (c) OpenMMLab. All rights reserved.
	import torch
	from datasets import load_dataset
	from mmengine.dataset import DefaultSampler
	from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
	                            LoggerHook, ParamSchedulerHook)
	from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
	from peft import LoraConfig
	from torch.optim import AdamW
	from transformers import (AutoModelForCausalLM, AutoTokenizer,
	                          BitsAndBytesConfig)
	
	from xtuner.dataset import process_hf_dataset
	from xtuner.dataset.collate_fns import default_collate_fn
	from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
	from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
	                                 VarlenAttnArgsToMessageHubHook)
	from xtuner.engine.runner import TrainLoop
	from xtuner.model import SupervisedFinetune
	from xtuner.parallel.sequence import SequenceParallelSampler
	from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
	
	#######################################################################
	#                          PART 1  Settings                           #
	#######################################################################
	# Model
	pretrained_model_name_or_path = '/root/ft/model'
	use_varlen_attn = False
	
	# Data
	alpaca_en_path = '/root/ft/data/personal_assistant.json'
	prompt_template = PROMPT_TEMPLATE.internlm2_chat
	max_length = 1024
	pack_to_max_length = True
	
	# parallel
	sequence_parallel_size = 1
	
	# Scheduler & Optimizer
	batch_size = 1  # per_device
	accumulative_counts = 16
	accumulative_counts *= sequence_parallel_size
	dataloader_num_workers = 0
	max_epochs = 2
	optim_type = AdamW
	lr = 2e-4
	betas = (0.9, 0.999)
	weight_decay = 0
	max_norm = 1  # grad clip
	warmup_ratio = 0.03
	
	# Save
	save_steps = 300
	save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)
	
	# Evaluate the generation performance during the training
	evaluation_freq = 300
	SYSTEM = ''
	evaluation_inputs = ['请你介绍一下你自己', '你是谁', '你是我的小助手吗']
	
	#######################################################################
	#                      PART 2  Model & Tokenizer                      #
	#######################################################################
	tokenizer = dict(
	    type=AutoTokenizer.from_pretrained,
	    pretrained_model_name_or_path=pretrained_model_name_or_path,
	    trust_remote_code=True,
	    padding_side='right')
	
	model = dict(
	    type=SupervisedFinetune,
	    use_varlen_attn=use_varlen_attn,
	    llm=dict(
	        type=AutoModelForCausalLM.from_pretrained,
	        pretrained_model_name_or_path=pretrained_model_name_or_path,
	        trust_remote_code=True,
	        torch_dtype=torch.float16,
	        quantization_config=dict(
	            type=BitsAndBytesConfig,
	            load_in_4bit=True,
	            load_in_8bit=False,
	            llm_int8_threshold=6.0,
	            llm_int8_has_fp16_weight=False,
	            bnb_4bit_compute_dtype=torch.float16,
	            bnb_4bit_use_double_quant=True,
	            bnb_4bit_quant_type='nf4')),
	    lora=dict(
	        type=LoraConfig,
	        r=64,
	        lora_alpha=16,
	        lora_dropout=0.1,
	        bias='none',
	        task_type='CAUSAL_LM'))
	
	#######################################################################
	#                      PART 3  Dataset & Dataloader                   #
	#######################################################################
	alpaca_en = dict(
	    type=process_hf_dataset,
	    dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
	    tokenizer=tokenizer,
	    max_length=max_length,
	    dataset_map_fn=openai_map_fn,
	    template_map_fn=dict(
	        type=template_map_fn_factory, template=prompt_template),
	    remove_unused_columns=True,
	    shuffle_before_pack=True,
	    pack_to_max_length=pack_to_max_length,
	    use_varlen_attn=use_varlen_attn)
	
	sampler = SequenceParallelSampler \
	    if sequence_parallel_size > 1 else DefaultSampler
	train_dataloader = dict(
	    batch_size=batch_size,
	    num_workers=dataloader_num_workers,
	    dataset=alpaca_en,
	    sampler=dict(type=sampler, shuffle=True),
	    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))
	
	#######################################################################
	#                    PART 4  Scheduler & Optimizer                    #
	#######################################################################
	# optimizer
	optim_wrapper = dict(
	    type=AmpOptimWrapper,
	    optimizer=dict(
	        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
	    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
	    accumulative_counts=accumulative_counts,
	    loss_scale='dynamic',
	    dtype='float16')
	
	# learning policy
	# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
	param_scheduler = [
	    dict(
	        type=LinearLR,
	        start_factor=1e-5,
	        by_epoch=True,
	        begin=0,
	        end=warmup_ratio * max_epochs,
	        convert_to_iter_based=True),
	    dict(
	        type=CosineAnnealingLR,
	        eta_min=0.0,
	        by_epoch=True,
	        begin=warmup_ratio * max_epochs,
	        end=max_epochs,
	        convert_to_iter_based=True)
	]
	
	# train, val, test setting
	train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
	
	#######################################################################
	#                           PART 5  Runtime                           #
	#######################################################################
	# Log the dialogue periodically during the training process, optional
	custom_hooks = [
	    dict(type=DatasetInfoHook, tokenizer=tokenizer),
	    dict(
	        type=EvaluateChatHook,
	        tokenizer=tokenizer,
	        every_n_iters=evaluation_freq,
	        evaluation_inputs=evaluation_inputs,
	        system=SYSTEM,
	        prompt_template=prompt_template)
	]
	
	if use_varlen_attn:
	    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]
	
	# configure default hooks
	default_hooks = dict(
	    # record the time of every iteration.
	    timer=dict(type=IterTimerHook),
	    # print log every 10 iterations.
	    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
	    # enable the parameter scheduler.
	    param_scheduler=dict(type=ParamSchedulerHook),
	    # save checkpoint per `save_steps`.
	    checkpoint=dict(
	        type=CheckpointHook,
	        by_epoch=False,
	        interval=save_steps,
	        max_keep_ckpts=save_total_limit),
	    # set sampler seed in distributed evrionment.
	    sampler_seed=dict(type=DistSamplerSeedHook),
	)
	
	# configure environment
	env_cfg = dict(
	    # whether to enable cudnn benchmark
	    cudnn_benchmark=False,
	    # set multi process parameters
	    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
	    # set distributed parameters
	    dist_cfg=dict(backend='nccl'),
	)
	
	# set visualizer
	visualizer = None
	
	# set log level
	log_level = 'INFO'
	
	# load from which checkpoint
	load_from = None
	
	# whether to resume training from the loaded checkpoint
	resume = False
	
	# Defaults to use random seed and disable `deterministic`
	randomness = dict(seed=None, deterministic=False)
	
	# set log processor
	log_processor = dict(by_epoch=False)
 
## 模型训练

使用xtuner train 指令即可开始训练。

我们可以通过添加 --work-dir 指定特定的文件保存位置，比如说就保存在 /root/ft/train 路径下。假如不添加的话模型训练的过程文件将默认保存在 ./work_dirs/internlm2_1_8b_qlora_alpaca_e3_copy 的位置

	# 指定保存路径
	xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train
 
输出文件如下

<img width="188" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/00c55745-903f-4e28-a45a-ca588e9e49e3">

 
**训练前**

<img width="900" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/45e8c6b8-07be-46b9-ad48-ac6d42cc6f03">

**768轮训练后**

<img width="497" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/5f514912-08fd-4944-9638-2e3155904e6e">

## 模型转换、整合、测试及部署

**模型转换**
可以看到Pytorch 训练出来的模型权重文件模型输出的文件是.pth格式的，我们要将其转换为 Huggingface 中常用的 .bin 格式文件，可以通过以下指令来实现一键转换。

	# 创建一个保存转换后 Huggingface 格式的文件夹
	mkdir -p /root/ft/huggingface
	
	# 模型转换
	# xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
	xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
 
 <img width="178" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/13cd1715-1021-49ef-9d97-216947a57002">

这时的huggingface 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”

**模型整合**

对于 LoRA 或者 QLoRA 微调出来的模型其实并不是一个完整的模型，而是一个额外的层（adapter）。那么训练完的这个层最终还是要与原模型进行组合才能被正常的使用。

而对于全量微调的模型（full）其实是不需要进行整合这一步的，因为全量微调修改的是原模型的权重而非微调一个新的 adapter ，因此是不需要进行模型整合的。

![image](https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/178fc7ce-0447-4224-a9c0-35bbae169b5c)

在 XTuner 中提供了一键整合的指令，我们需要准备好三个地址，包括原模型的地址、训练好的 adapter 层的地址（转为 Huggingface 格式后保存的部分）以及最终保存的地址。

	# 创建一个名为 final_model 的文件夹存储整合后的模型文件
	mkdir -p /root/ft/final_model
	
	# 解决一下线程冲突的 Bug 
	export MKL_SERVICE_FORCE_INTEL=1
	
	# 进行模型整合
	# xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
	xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
 
<img width="1096" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/4805546f-bcc2-41ff-bf4b-6f4cce827962">

**对话测试**

准备好我们刚刚转换好的模型路径并选择对应的提示词模版（prompt-template）即可进行对话。rompt-tempolate，可以到 XTuner 源码中的 xtuner/utils/templates.py 这个文件中进行查找。

	# 与模型进行对话
	xtuner chat /root/ft/final_model --prompt-template internlm2_chat

 <img width="415" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/c97e40a6-3310-4ab0-a938-ea83b5f40f8a">

可以看到模型已经过拟合，不同的问题回复的话相同。未微调前的模型表现如下：可以看到在没有进行我们数据的微调前，原模型是能够输出有逻辑的回复，并且也不会认为他是我们特有的小助手。

<img width="980" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/f4b0708e-f04b-4706-a9a7-35d7c55c7c99">


**demo部署**



# 进阶作业


