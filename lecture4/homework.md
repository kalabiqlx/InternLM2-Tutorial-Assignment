# contents
- [基础作业](#基础作业)
   - [环境安装](#环境安装)
   - [前期准备](#前期准备)
   - [配置文件修改](#配置文件修改)
   - [模型训练](#模型训练)
   - [模型转换、整合、测试及部署](#模型转换、整合、测试及部署)
- [进阶作业](#进阶作业)
  - [部署OpenXLab](#部署OpenXLab)
  - [复现多模态微调](#复现多模态微调)

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

<img width="1096" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/178fc7ce-0447-4224-a9c0-35bbae169b5c">

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

下载网页端 web demo 所需要的依赖:

	pip install streamlit==1.24.0
 
下载 InternLM 项目代码

	# 创建存放 InternLM 文件的代码
	mkdir -p /root/ft/web_demo && cd /root/ft/web_demo
	
	# 拉取 InternLM 源文件
	git clone https://github.com/InternLM/InternLM.git
	
	# 进入该库中
	cd /root/ft/web_demo/InternLM

<img width="505" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/d0cfbdc3-94ed-40fb-b388-f241126eff5a">

将 /root/ft/web_demo/InternLM/chat/web_demo.py 中的内容替换为以下的代码

	"""This script refers to the dialogue example of streamlit, the interactive
	generation code of chatglm2 and transformers.
	
	We mainly modified part of the code logic to adapt to the
	generation of our model.
	Please refer to these links below for more information:
	    1. streamlit chat example:
	        https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
	    2. chatglm2:
	        https://github.com/THUDM/ChatGLM2-6B
	    3. transformers:
	        https://github.com/huggingface/transformers
	Please run with the command `streamlit run path/to/web_demo.py
	    --server.address=0.0.0.0 --server.port 7860`.
	Using `python path/to/web_demo.py` may cause unknown problems.
	"""
	# isort: skip_file
	import copy
	import warnings
	from dataclasses import asdict, dataclass
	from typing import Callable, List, Optional
	
	import streamlit as st
	import torch
	from torch import nn
	from transformers.generation.utils import (LogitsProcessorList,
	                                           StoppingCriteriaList)
	from transformers.utils import logging
	
	from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip
	
	logger = logging.get_logger(__name__)
	
	
	@dataclass
	class GenerationConfig:
	    # this config is used for chat to provide more diversity
	    max_length: int = 2048
	    top_p: float = 0.75
	    temperature: float = 0.1
	    do_sample: bool = True
	    repetition_penalty: float = 1.000
	
	
	@torch.inference_mode()
	def generate_interactive(
	    model,
	    tokenizer,
	    prompt,
	    generation_config: Optional[GenerationConfig] = None,
	    logits_processor: Optional[LogitsProcessorList] = None,
	    stopping_criteria: Optional[StoppingCriteriaList] = None,
	    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
	                                                List[int]]] = None,
	    additional_eos_token_id: Optional[int] = None,
	    **kwargs,
	):
	    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
	    input_length = len(inputs['input_ids'][0])
	    for k, v in inputs.items():
	        inputs[k] = v.cuda()
	    input_ids = inputs['input_ids']
	    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
	    if generation_config is None:
	        generation_config = model.generation_config
	    generation_config = copy.deepcopy(generation_config)
	    model_kwargs = generation_config.update(**kwargs)
	    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
	        generation_config.bos_token_id,
	        generation_config.eos_token_id,
	    )
	    if isinstance(eos_token_id, int):
	        eos_token_id = [eos_token_id]
	    if additional_eos_token_id is not None:
	        eos_token_id.append(additional_eos_token_id)
	    has_default_max_length = kwargs.get(
	        'max_length') is None and generation_config.max_length is not None
	    if has_default_max_length and generation_config.max_new_tokens is None:
	        warnings.warn(
	            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
	                to control the generation length. "
	            'This behaviour is deprecated and will be removed from the \
	                config in v5 of Transformers -- we'
	            ' recommend using `max_new_tokens` to control the maximum \
	                length of the generation.',
	            UserWarning,
	        )
	    elif generation_config.max_new_tokens is not None:
	        generation_config.max_length = generation_config.max_new_tokens + \
	            input_ids_seq_length
	        if not has_default_max_length:
	            logger.warn(  # pylint: disable=W4902
	                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
	                f"and 'max_length'(={generation_config.max_length}) seem to "
	                "have been set. 'max_new_tokens' will take precedence. "
	                'Please refer to the documentation for more information. '
	                '(https://huggingface.co/docs/transformers/main/'
	                'en/main_classes/text_generation)',
	                UserWarning,
	            )
	
	    if input_ids_seq_length >= generation_config.max_length:
	        input_ids_string = 'input_ids'
	        logger.warning(
	            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
	            f"but 'max_length' is set to {generation_config.max_length}. "
	            'This can lead to unexpected behavior. You should consider'
	            " increasing 'max_new_tokens'.")
	
	    # 2. Set generation parameters if not already defined
	    logits_processor = logits_processor if logits_processor is not None \
	        else LogitsProcessorList()
	    stopping_criteria = stopping_criteria if stopping_criteria is not None \
	        else StoppingCriteriaList()
	
	    logits_processor = model._get_logits_processor(
	        generation_config=generation_config,
	        input_ids_seq_length=input_ids_seq_length,
	        encoder_input_ids=input_ids,
	        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
	        logits_processor=logits_processor,
	    )
	
	    stopping_criteria = model._get_stopping_criteria(
	        generation_config=generation_config,
	        stopping_criteria=stopping_criteria)
	    logits_warper = model._get_logits_warper(generation_config)
	
	    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
	    scores = None
	    while True:
	        model_inputs = model.prepare_inputs_for_generation(
	            input_ids, **model_kwargs)
	        # forward pass to get next token
	        outputs = model(
	            **model_inputs,
	            return_dict=True,
	            output_attentions=False,
	            output_hidden_states=False,
	        )
	
	        next_token_logits = outputs.logits[:, -1, :]
	
	        # pre-process distribution
	        next_token_scores = logits_processor(input_ids, next_token_logits)
	        next_token_scores = logits_warper(input_ids, next_token_scores)
	
	        # sample
	        probs = nn.functional.softmax(next_token_scores, dim=-1)
	        if generation_config.do_sample:
	            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
	        else:
	            next_tokens = torch.argmax(probs, dim=-1)
	
	        # update generated ids, model inputs, and length for next step
	        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
	        model_kwargs = model._update_model_kwargs_for_generation(
	            outputs, model_kwargs, is_encoder_decoder=False)
	        unfinished_sequences = unfinished_sequences.mul(
	            (min(next_tokens != i for i in eos_token_id)).long())
	
	        output_token_ids = input_ids[0].cpu().tolist()
	        output_token_ids = output_token_ids[input_length:]
	        for each_eos_token_id in eos_token_id:
	            if output_token_ids[-1] == each_eos_token_id:
	                output_token_ids = output_token_ids[:-1]
	        response = tokenizer.decode(output_token_ids)
	
	        yield response
	        # stop when each sentence is finished
	        # or if we exceed the maximum length
	        if unfinished_sequences.max() == 0 or stopping_criteria(
	                input_ids, scores):
	            break
	
	
	def on_btn_click():
	    del st.session_state.messages
	
	
	@st.cache_resource
	def load_model():
	    model = (AutoModelForCausalLM.from_pretrained('/root/ft/final_model',
	                                                  trust_remote_code=True).to(
	                                                      torch.bfloat16).cuda())
	    tokenizer = AutoTokenizer.from_pretrained('/root/ft/final_model',
	                                              trust_remote_code=True)
	    return model, tokenizer
	
	
	def prepare_generation_config():
	    with st.sidebar:
	        max_length = st.slider('Max Length',
	                               min_value=8,
	                               max_value=32768,
	                               value=2048)
	        top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
	        temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
	        st.button('Clear Chat History', on_click=on_btn_click)
	
	    generation_config = GenerationConfig(max_length=max_length,
	                                         top_p=top_p,
	                                         temperature=temperature)
	
	    return generation_config
	
	
	user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
	robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
	cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
	    <|im_start|>assistant\n'
	
	
	def combine_history(prompt):
	    messages = st.session_state.messages
	    meta_instruction = ('')
	    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
	    for message in messages:
	        cur_content = message['content']
	        if message['role'] == 'user':
	            cur_prompt = user_prompt.format(user=cur_content)
	        elif message['role'] == 'robot':
	            cur_prompt = robot_prompt.format(robot=cur_content)
	        else:
	            raise RuntimeError
	        total_prompt += cur_prompt
	    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
	    return total_prompt
	
	
	def main():
	    # torch.cuda.empty_cache()
	    print('load model begin.')
	    model, tokenizer = load_model()
	    print('load model end.')
	
	
	    st.title('InternLM2-Chat-1.8B')
	
	    generation_config = prepare_generation_config()
	
	    # Initialize chat history
	    if 'messages' not in st.session_state:
	        st.session_state.messages = []
	
	    # Display chat messages from history on app rerun
	    for message in st.session_state.messages:
	        with st.chat_message(message['role'], avatar=message.get('avatar')):
	            st.markdown(message['content'])
	
	    # Accept user input
	    if prompt := st.chat_input('What is up?'):
	        # Display user message in chat message container
	        with st.chat_message('user'):
	            st.markdown(prompt)
	        real_prompt = combine_history(prompt)
	        # Add user message to chat history
	        st.session_state.messages.append({
	            'role': 'user',
	            'content': prompt,
	        })
	
	        with st.chat_message('robot'):
	            message_placeholder = st.empty()
	            for cur_response in generate_interactive(
	                    model=model,
	                    tokenizer=tokenizer,
	                    prompt=real_prompt,
	                    additional_eos_token_id=92542,
	                    **asdict(generation_config),
	            ):
	                # Display robot response in chat message container
	                message_placeholder.markdown(cur_response + '▌')
	            message_placeholder.markdown(cur_response)
	        # Add robot response to chat history
	        st.session_state.messages.append({
	            'role': 'robot',
	            'content': cur_response,  # pylint: disable=undefined-loop-variable
	        })
	        torch.cuda.empty_cache()
	
	
	if __name__ == '__main__':
	    main()

win+R 打开powershell，输入以下命令，并将端口号替换为自己的端口，输入密码

	# 从本地使用 ssh 连接 studio 端口
	# 将下方端口号 38374 替换成自己的端口号
	ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374

<img width="638" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/c3650cb2-1c2b-49e7-bfe9-1193c649851b">

输入以下命令运行 /root/personal_assistant/code/InternLM 目录下的 web_demo.py 文件。打开 http://127.0.0.1:6006 后，等待加载完成即可进行对话

	streamlit run /root/ft/web_demo/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006

<img width="997" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/d0bfaa69-6b34-47d5-98a4-f0644e99a6de">

<img width="1043" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/95ccae20-c9d7-4b39-95e5-4bee80ec0e3a">

可以看到原先在终端执行的模型被部署到了web上。

# 进阶作业

## 部署OpenXLab

## 复现多模态微调
