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

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/e96e2e1f-7ce5-4afc-a788-367bc17d7db8">

书生·浦语大模型开源历程

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/ec7e9dfc-1446-4524-b6c8-dadde00a6601">

书生·浦语2.0体系

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/9aee2c4b-9157-4711-84e3-a37dc8b34026">

书生·浦语2.0主要亮点

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/e1ba287e-4546-4e62-93ce-bd675a49e19a">

模型应用流程

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/ff579729-027c-45e8-bd1f-7e3b07baab46">

书生·浦语全链条开源开放体系

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/1669b1cc-acd0-468f-8e64-306656bcf920">

### 数据
[书生·万卷](https://opendatalab.org.cn/)

书生·万卷1.0 为书生·万卷多模态语料库的首个开源版本，包含文本数据集、图文数据集、视频数据集三部分，数据总量超过2TB。基于大模型数据联盟构建的语料库，上海AI实验室对其中部分数据进行细粒度清洗、去重以及价值对齐，形成了书生·万卷1.0，具备多元融合、精细处理、价值对齐、易用高效等四大特征。

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/151a8aeb-adaf-4ea4-b885-f16e34421053">

### 预训练
<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/b94b684e-ed3c-4d97-b700-7ee74e115ff3">

### 微调
[xtuner](https://github.com/InternLM/tutorial/tree/main/xtuner)|[xtuner中文文档](https://xtuner.readthedocs.io/zh-cn/latest/index.html)

XTuner 是一个高效、灵活、全能的轻量化大模型微调工具库。支持大语言模型 LLM、多模态图文模型 VLM 的预训练及轻量级微调。XTuner 支持在 8GB 显存下微调 7B 模型，同时也支持多节点跨设备微调更大尺度模型（70B+）。支持增量预训练、指令微调与 Agent 微调。预定义众多开源对话模版，支持与开源或训练所得模型进行对话。

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8676340d-05d7-4f04-a84b-71132af96318">

### 评测
[OpenCompass 2.0](https://opencompass.org.cn/home)

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/8d7bfa57-007b-4b00-9dc7-a2b20a471857">

本次发布的OpenCompass2.0，首次推出支撑大模型评测的“铁三角”：
·权威评测榜单CompassRank
·高质量评测基准社区CompassHub
·评测工具链体系CompassKit
基于全新升级的能力体系和工具链，OpenCompass2.0构建了一套高质量的中英文双语评测基准，涵盖语言与理解、常识与逻辑推理、数学计算与应用、多编程语言代码能力、智能体、创作与对话等多个方面对大模型进行评测分析。通过高质量、多层次的综合性能力评测基准，OpenCompass2.0创新了多项能力评测方法，实现了对模型真实能力的全面诊断。

[CompassRank 榜单地址](https://rank.opencompass.org.cn/home)|[CompassHub社区地址](https://hub.opencompass.org.cn/home)

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/4ff75582-5219-47ae-8c98-7070c466110d">

### 部署
[LMDeploy](https://github.com/InternLM/lmdeploy)

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/397a4bb9-6c25-4710-a8cc-1f582817a3a5">

### 应用
[AgentLego](https://github.com/InternLM/agentlego)

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/33e70270-11e7-4c54-9c13-3e118bc01917">

## InternLM2技术报告笔记
[InternLM2 技术报告](https://arxiv.org/pdf/2403.17297.pdf)

本文主要分为七个部分：摘要，介绍，基础设施，预训练，对齐，评测分析，结论。

**摘要：** 本文介绍InternLM2，一个开源的大语言模型，它在6个维度和30个基准的全面评估中超越了其前身，特别是在长序列建模和开放性主观评估方面，通过创新的预训练和优化技术实现了这一突破。InternLM2详细阐述了预训练过程中各类数据的准备，包括文本、代码和长文本数据。InternLM2有效地捕捉长期依赖性，预训练阶段从4k个token开始，然后扩展到32k个token，其在200k个“大海捞针”测试中的表现优异。InternLM2还通过监督微调（SFT）和一种基于人类反馈的新型条件在线强化学习方法（COOL RLHF）策略进行进一步校准，以解决人类偏好冲突和奖励策略滥用问题。

**介绍：** 大语言模型的发展包括预训练、监督微调（SFT）和基于人类反馈的强化学习（RLHF）等主要阶段。预训练主要基于利用大量的自然文本语料库，积累数万亿的token。这个阶段的目标是为大语言模型配备广泛的知识库和基本技能。预训练阶段的数据质量被认为是最重要的因素。然而，过去关于大语言模型的技术报告很少关注预训练数据的处理。InternLM2首先采用分组查询注意力（GQA）来在推断长序列时减少内存占用。在预训练阶段，我们首先使用4k个上下文文本训练InternLM2，然后将训练语料库过渡到高质量的32k文本进行进一步训练。最终，通过位置编码外推 ，InternLM2在200k个上下文中通过了“大海捞针”测试，表现出色。预训练后，使用监督微调（SFT）和基于人类反馈的强化学习（RLHF）来确保模型能很好地遵循人类指令并符合人类价值观。我们还在此过程中构建了相应的32k数据，以进一步提升InternLM2的长上下文处理能力。

**基础设施：** 介绍了在预训练、SFT 和 RLHF 中使用的训练框架 InternEvo。使用高效的轻量级预训练框架InternEvo进行模型训练，这个框架使得我们能够在数 千个GPU上扩展模型训练。它通过数据、张量、序列和管道并行技术来实现这一点。

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/5b119c15-62a1-4b48-94f7-6899e1dd5b2b">

**预训练：** 介绍预训练数据、预训练设置以及三个预训练阶段。

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/0954a0dd-081c-4780-9a54-a8ff46f31216">

**对齐：** 预训练阶段为大型语言模型（LLMs）赋予了解决各种任务所需的基础能力和知识。进一步微调LLMs，以充分激发其能力，并指导LLMs作为有益和无害的AI助手。这一阶段，也常被称为“对齐”（Alignment），通常包含两个阶段：监督微调（SFT）和基于人类反馈的强化学习（RLHF）。在SFT阶段，通过高质量指令数据（见4.1 监督微调）微调模型，使其遵循多种人类指令。然后我们提出了带人类反馈的条件在线强化学习（COnditionalOnLine Reinforcement Learning with Human Feedback，COOL RLHF），它应用了一种新颖的条件奖励模型，可以调和不同的人类偏好（例如，多步推理准确性、有益性、无害性），并进行三轮在线RLHF以减少奖励黑客攻击。在对齐阶段，我们通过在SFT和RLHF阶段利用长上下文预训练数据来保持LLMs的长上下文能力。我们还介绍了我们提升LLMs工具利用能力的实践。

<img width="585" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/489c8020-d7e2-4d33-a142-12116baa99b2">

**评测分析：** 全面评测和分析了语言模型在多个领域和任务中的表现。评测主要分为两种类别：(a)下游任务和(b)对齐性。对于每个类别，进一步将评测任务细分为具体的子任务，以详细了解模型的优点和缺点。最后，讨论了语言模型中潜在的数据污染问题及其对模型性能和可靠性的影响。除非另有明确说明，所有评测都使用OpenCompass进行。

**结论：** 本报告介绍了InternLM2大型语言模型，它在主观和客观评测中表现出色。InternLM2基于超过2T的高质量预训练数据进行训练，涵盖了1.8B、7B和20B参数的模型规模，适用于多种场景。为了更好地支持长文本处理，InternLM2采用了GQA来降低推理成本，并额外训练在多达32000个上下文中。除了开源模型本身，还提供了训练过程中的多个阶段检查点，以便利后续研究者的研究。除了开源模型，还详细阐述了InternLM2的训练过程，包括训练框架、预训练文本数据、预训练代码数据、预训练长文本数据和对齐数据。此外，针对强化学习后训练（RLHF）过程中遇到的偏好冲突，提出了条件在线RLHF方法，以协调不同的偏好。这些信息对于理解如何准备预训练数据以及如何更有效地训练大型模型具有参考价值。

**贡献点：** 贡献有两个方面，不仅体现在模型在各种基准测试中的卓越性能，还体现在在不同发展阶段全面开发模型的方法。关键点包括
·开源InternLM2模型展现卓越性能: 我们已经开源了不同规模的模型包括1.8B、7B和20B，它们在主观和客观评估中都表现出色。此外，我们还发布了不同阶段的模型，以促进社区分析SFT和RLHF训练后的变化。
·设计带有200k上下文窗口: InternLM2在长序列任务中表现出色，在带有200k上下文的“大海捞针”实验中，几乎完美地识别出所有的“针”。此外，我们提供了所有阶段包括预训练、SFT和RLHF的长文本语言模型的经验。
·综合数据准备指导: 我们详细阐述了为大语言模型（LLM）准备数据的方法，包括预训练数据、特定领域增强数据、监督微调（SFT）和基于人类监督的强化学习（RLHF）数据。这些细节将有助于社区更好地训练LLM。
·创新的RLHF训练技术: 我们引入了条件在线RLHF（COOL RLHF）来调整各种偏好，显著提高了InternLM2在各种主观对话评估中的表现。我们还对RLHF的主观和客观结果进行了初步分析和比较，为社区提供对RLHF的深入理解。

