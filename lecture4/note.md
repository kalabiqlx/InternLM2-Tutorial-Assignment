[视频](https://www.bilibili.com/video/BV15m421j78d)|[官方文档](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/personal_assistant_document.md)

# contents
- [微调](#微调)
   - [定义](#定义)
   - [微调方法](#微调方法)
   - [数据处理](#数据处理)
      - [增量预训练和指令跟随](#增量预训练和指令跟随)
      - [全量微调FFT](#全量微调FFT)
      - [PEFT](#PEFT)
- [XTuner](#XTuner)
- [8GB显存玩转LLM](#8GB显存玩转LLM)
- [InternLM2_1.8B_模型](#InternLM2_1.8B_模型)
- [多模态LLM微调](#多模态LLM微调)


## 微调

### 定义
**Fine-tuning（微调）**：通过特定领域数据对预训练模型进行针对性优化，以提升其在特定任务上的性能。

大模型微调是利用特定领域的数据集对已预训练的大模型进行进一步训练的过程。它旨在优化模型在特定任务上的性能，使模型能够更好地适应和完成特定领域的任务。

**微调的核心原因:**

* 定制化功能：微调的核心原因是赋予大模型更加定制化的功能。通用大模型虽然强大，但在`特定领域`可能表现不佳。通过微调，可以使模型更好地适应`特定领域`的需求和特征。
* 领域知识学习：通过引入特定领域的数据集进行微调，大模型可以学习该领域的知识和语言模式。这有助于模型在特定任务上取得更好的性能。

**微调与超参数优化：**

微调过程中，超参数的调整至关重要。超参数如学习率、批次大小和训练轮次等需要根据特定任务和数据集进行调整，以确保模型在训练过程中的有效性和性能。

### 微调方法

#### 增量预训练和指令跟随

`增量预训练`和`指令跟随`是经常会用到两种的微调模式。`增量预训练`通过文章、书籍、代码等往基础模型中添加持垂直领域的知识，一般是陈述语句。`指令跟随`通过QA pair来训练模型的对话能力。

<img width="977" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/19f9ef89-7ae1-4be3-862a-147159594778">

如图所示，预训练后的模型在输入问题语句时会根据相似度匹配类似的语句，而经过`指令微调`之后能够针对输入的问题给出相应的回答。

<img width="994" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/2c040a00-9348-464c-ab38-1bd22b28e6e5">

从训练数据的来源、以及训练的方法的角度，大模型的微调有以下几条技术路线：

* 一个是**监督式微调SFT(Supervised Fine Tuning)**，这个方案主要是用人工标注的数据，用传统机器学习中监督学习的方法，对大模型进行微调；
* 一个是**基于人类反馈的强化学习微调RLHF(Reinforcement Learning with Human Feedback)**，这个方案的主要特点是把人类的反馈，通过强化学习的方式，引入到对大模型的微调中去，让大模型生成的结果，更加符合人类的一些期望；
* 一个是**基于AI反馈的强化学习微调RLAIF(Reinforcement Learning with AI Feedback)**，这个原理大致跟RLHF类似，但是反馈的来源是AI。这里是想解决反馈系统的效率问题，因为收集人类反馈，相对来说成本会比较高、效率比较低。

从参数规模的角度，大模型的微调分成两条技术路线：

* 对全量的参数，进行全量的训练，这条路径叫**全量微调FFT(Full Fine Tuning)**。
* 只对部分的参数进行训练，这条路径叫**PEFT(Parameter-Efficient Fine Tuning)**。

#### 全量微调FFT
FFT的原理，就是用特定的数据，对大模型进行训练，将W变成W`，W`相比W ，最大的优点就是上述特定数据领域的表现会好很多。

但FFT也会带来一些问题，影响比较大的问题，主要有以下两个：

* 一个是训练的成本会比较高，因为微调的参数量跟预训练的是一样的多的；

* 一个是叫灾难性遗忘(Catastrophic Forgetting)，用特定训练数据去微调可能会把这个领域的表现变好，但也可能会把原来表现好的别的领域的能力变差。

PEFT主要想解决的问题，就是FFT存在的上述两个问题，PEFT也是目前比较主流的微调方案。

#### PEFT

从成本和效果的角度综合考虑，PEFT是目前业界比较流行的微调方案。这里介绍几种比较流行的PEFT微调方案，包括**Prompt Tuning**，**Prefix Tuning**，**LoRA**，**QLoRA**

**Prompt Tuning**

Prompt Tuning的出发点，是基座模型(Foundation Model)的参数不变，为每个特定任务，训练一个少量参数的小模型，在具体执行特定任务的时候按需调用。其基本原理是在输入序列X之前，增加一些特定长度的特殊Token，以增大生成期望序列的概率。具体来说，就是将X = [x1, x2, ..., xm]变成，X` = [x`1, x`2, ..., x`k; x1, x2, ..., xm], Y = WX`。Prompt Tuning是发生在Embedding这个环节的。如果将大模型比做一个函数：Y=f(X)，那么Prompt Tuning就是在保证函数本身不变的前提下，在X前面加上了一些特定的内容，而这些内容可以影响X生成期望中Y的概率。

[论文：The Power of Scale for Parameter-Efficient Prompt Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.08691.pdf)

**Prefix Tuning**

Prefix Tuning的灵感来源是，基于Prompt Engineering的实践表明，在不改变大模型的前提下，在Prompt上下文中添加适当的条件，可以引导大模型有更加出色的表现。Prefix Tuning的出发点，跟Prompt Tuning的是类似的，只不过它们的具体实现上有一些差异。Prompt Tuning是在Embedding环节，往输入序列X前面加特定的Token。而Prefix Tuning是在Transformer的Encoder和Decoder的网络中都加了一些特定的前缀。具体来说，就是将Y=WX中的W，变成W` = [Wp; W]，Y=W`X。Prefix Tuning也保证了基座模型本身是没有变的，只是在推理的过程中，按需要在W前面拼接一些参数。

[论文：Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2101.00190.pdf)


**LoRA**

LoRA是跟Prompt Tuning和Prefix Tuning完全不相同的另一条技术路线。LoRA背后有一个假设：我们现在看到的这些大语言模型，它们都是被过度参数化的。而过度参数化的大模型背后，都有一个低维的本质模型。通俗讲人话：大模型参数很多，但并不是所有的参数都是发挥同样作用的；大模型中有其中一部分参数，是非常重要的，是影响大模型生成结果的关键参数，这部分关键参数就是上面提到的低维的本质模型。LoRA的基本思路，包括以下几步：

* 首先, 要适配特定的下游任务，要训练一个特定的模型，将Y=WX变成Y=(W+∆W)X，这里面∆W主是我们要微调得到的结果；
* 其次，将∆W进行低维分解∆W=AB (∆W为m * n维，A为m * r维，B为r * n维，r就是上述假设中的低维)；
* 接下来，用特定的训练数据，训练出A和B即可得到∆W，在推理的过程中直接将∆W加到W上去，再没有额外的成本。
* 另外，如果要用LoRA适配不同的场景，切换也非常方便，做简单的矩阵加法即可：(W + ∆W) - ∆W + ∆W`。

[论文：LoRA: Low-Rank Adaptation of Large Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2106.09685.pdf)

**QLoRA**

量化（Quantization），是一种在保证模型效果基本不降低的前提下，通过降低参数的精度，来减少模型对于计算资源的需求的方法。量化的核心目标是降成本，降训练成本，特别是降后期的推理成本。QLoRA就是量化版的LoRA，它是在LoRA的基础上，进行了进一步的量化，将原本用16bit表示的参数，降为用4bit来表示，可以在保证模型效果的同时，极大地降低成本。，65B的LLaMA 的微调要780GB的GPU内存；而用了QLoRA之后，只需要48GB。

[论文：QLoRA: Efficient Finetuning of Quantized LLMs](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2305.14314.pdf)

<img width="937" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/b3d5efc5-414f-4124-bf2c-98c90f24cac8">

<img width="992" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/5047405b-b53b-4305-868c-6d0554a17367">

[其他微调方法：Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2303.15647.pdf)

### 数据处理

<img width="971" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/1400f7c9-773a-4f79-91c6-9d44e88b2747">

<img width="975" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/68c4644c-1a39-4928-bb7d-d493485a98ad">

<img width="924" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/c8859184-93f6-446e-9cf0-c4aaa3e9eb97">

<img width="938" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/92246ffc-de78-4159-99ad-25450c49105c">

<img width="961" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/39eb19d5-e23c-49ee-a875-b91e767e921a">

## XTuner

## 8GB显存玩转LLM

## InternLM2_1.8B_模型

## 多模态LLM微调

## Agent
