# 第一节课 《书生·浦语大模型全链路开源体系》
[官方github地址](https://github.com/InternLM)|[官方地址](https://internlm.intern-ai.org.cn/)

# contents
- [视频笔记](#视频笔记)
   - [数据](#数据)
   - [预训练](#预训练)
   - [微调](#微调)
   - [评测](#评测)
   - [部署](#部署)
   - [应用](#应用)
- [HInternLM2 技术报告笔记](#InternLM2技术报告笔记)

   
## 视频笔记
[视频](https://www.bilibili.com/video/BV1Vx421X72D/) 2024.5.18

语言模型的发展历程
<img width="991" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/e96e2e1f-7ce5-4afc-a788-367bc17d7db8">

书生·浦语大模型开源历程
<img width="991" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/ec7e9dfc-1446-4524-b6c8-dadde00a6601">

书生·浦语2.0体系
<img width="949" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/9aee2c4b-9157-4711-84e3-a37dc8b34026">

书生·浦语2.0主要亮点
<img width="937" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/e1ba287e-4546-4e62-93ce-bd675a49e19a">

模型应用流程
<img width="977" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/ff579729-027c-45e8-bd1f-7e3b07baab46">

书生·浦语全链条开源开放体系
<img width="983" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/1669b1cc-acd0-468f-8e64-306656bcf920">

### 数据
[书生·万卷](https://opendatalab.org.cn/)
书生·万卷1.0 为书生·万卷多模态语料库的首个开源版本，包含文本数据集、图文数据集、视频数据集三部分，数据总量超过2TB。基于大模型数据联盟构建的语料库，上海AI实验室对其中部分数据进行细粒度清洗、去重以及价值对齐，形成了书生·万卷1.0，具备多元融合、精细处理、价值对齐、易用高效等四大特征。
<img width="953" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/151a8aeb-adaf-4ea4-b885-f16e34421053">

### 预训练
<img width="581" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/b94b684e-ed3c-4d97-b700-7ee74e115ff3">

### 微调
[xtuner](https://github.com/InternLM/tutorial/tree/main/xtuner)|[xtuner中文文档](https://xtuner.readthedocs.io/zh-cn/latest/index.html)
XTuner 是一个高效、灵活、全能的轻量化大模型微调工具库。支持大语言模型 LLM、多模态图文模型 VLM 的预训练及轻量级微调。XTuner 支持在 8GB 显存下微调 7B 模型，同时也支持多节点跨设备微调更大尺度模型（70B+）。支持增量预训练、指令微调与 Agent 微调。预定义众多开源对话模版，支持与开源或训练所得模型进行对话。
<img width="588" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8676340d-05d7-4f04-a84b-71132af96318">

### 评测
[OpenCompass 2.0](https://opencompass.org.cn/home)
<img width="624" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8d7bfa57-007b-4b00-9dc7-a2b20a471857">

本次发布的OpenCompass2.0，首次推出支撑大模型评测的“铁三角”：
·权威评测榜单CompassRank
·高质量评测基准社区CompassHub
·评测工具链体系CompassKit
基于全新升级的能力体系和工具链，OpenCompass2.0构建了一套高质量的中英文双语评测基准，涵盖语言与理解、常识与逻辑推理、数学计算与应用、多编程语言代码能力、智能体、创作与对话等多个方面对大模型进行评测分析。通过高质量、多层次的综合性能力评测基准，OpenCompass2.0创新了多项能力评测方法，实现了对模型真实能力的全面诊断。

[CompassRank 榜单地址](https://rank.opencompass.org.cn/home)|[CompassHub社区地址](https://hub.opencompass.org.cn/home)
<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/4ff75582-5219-47ae-8c98-7070c466110d">

### 部署
[LMDeploy](https://github.com/InternLM/lmdeploy)
<img width="578" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/397a4bb9-6c25-4710-a8cc-1f582817a3a5">

### 应用
[AgentLego](https://github.com/InternLM/agentlego)
<img width="568" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/33e70270-11e7-4c54-9c13-3e118bc01917">

## InternLM2技术报告笔记
[InternLM2 技术报告](https://arxiv.org/pdf/2403.17297.pdf)

