[视频地址](https://www.bilibili.com/video/BV1QA4m1F7t4/)|[官网地址](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md)|[参考知乎文章](https://zhuanlan.zhihu.com/p/668082024)

# contents
- [RAG](#RAG)
   - [定义与基础概述](#定义与基础概述)
   - [原理](#原理) 
   
# RAG  

## 定义与基础概述

检索增强生成（Retrieval Augmented Generation），简称 RAG，已经成为当前最火热的LLM应用方案。经历今年年初那一波大模型潮，想必大家对大模型的能力有了一定的了解，但是当我们将大模型应用于实际业务场景时会发现，通用的基础大模型基本无法满足我们的实际业务需求，主要有以下几方面原因：

* 知识的局限性：模型自身的知识完全源于它的训练数据，而现有的主流大模型（ChatGPT、文心一言、通义千问…）的训练集基本都是构建于网络公开的数据，对于一些实时性的、非公开的或离线的数据是无法获取到的，这部分知识也就无从具备。
  
* 幻觉问题：所有的AI模型的底层原理都是基于数学概率，其模型输出实质上是一系列数值运算，大模型也不例外，所以它有时候会一本正经地胡说八道，尤其是在大模型自身不具备某一方面的知识或不擅长的场景。而这种幻觉问题的区分是比较困难的，因为它要求使用者自身具备相应领域的知识。
  
* 数据安全性：对于企业来说，数据安全至关重要，没有企业愿意承担数据泄露的风险，将自身的私域数据上传第三方平台进行训练。这也导致完全依赖通用大模型自身能力的应用方案不得不在数据安全和效果方面进行取舍。

RAG（Retrieval Augmented Generation）是一种结合了检索（Retrieval）和生成（Generation）的技术，前者主要是利用向量数据库的高效存储和检索能力，召回目标知识；后者则是利用大模型和Prompt工程，将召回的知识合理利用，生成目标答案。RAG旨在通过利用外部知识库来增强大型语言模型（LLMs）的性能。它通过检索与用户输入相关的信息片段，并结合这些 信息来生成更准确、更丰富的回答。

<img width="985" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/3536851a-da23-42b5-b235-189b0c0ca9c7">

## 原理

完整的RAG应用流程主要包含两个阶段：

* 数据准备阶段：数据提取——>文本分割——>向量化（embedding）——>数据入库

* 应用阶段：用户提问——>数据检索（召回）——>注入Prompt——>LLM生成答案

<img width="994" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/b27ea74f-02d2-4deb-b0df-6572608b9779">

**数据准备阶段：** 数据准备一般是一个离线的过程，主要是将私域数据向量化后构建索引并存入数据库的过程。主要包括：数据提取、文本分割、向量化、数据入库等环节。

* 数据提取
	* 数据加载：包括多格式数据加载、不同数据源获取等，根据数据自身情况，将数据处理为同一个范式。
	* 数据处理：包括数据过滤、压缩、格式化等。
	* 元数据获取：提取数据中关键信息，例如文件名、Title、时间等 。
* 文本分割：
	* 文本分割主要考虑两个因素：1）embedding模型的Tokens限制情况；2）语义完整性对整体的检索效果的影响。一些常见的文本分割方式如下：
	* 句分割：以”句”的粒度进行切分，保留一个句子的完整语义。常见切分符包括：句号、感叹号、问号、换行符等。
	* 固定长度分割：根据embedding模型的token长度限制，将文本分割为固定长度（例如256/512个tokens），这种切分方式会损失很多语义信息，一般通过在头尾增加一定冗余量来缓解。
* 向量化（embedding）：

向量化是一个将文本数据转化为向量矩阵的过程，该过程会直接影响到后续检索的效果。目前常见的embedding模型如表中所示，这些embedding模型基本能满足大部分需求，但对于特殊场景（例如涉及一些罕见专有词或字等）或者想进一步优化效果，则可以选择开源Embedding模型微调或直接训练适合自己场景的Embedding模型。

* 数据入库：
数据向量化后构建索引，并写入数据库的过程可以概述为数据入库过程，适用于RAG场景的数据库包括：FAISS、Chromadb、ES、milvus等。一般可以根据业务场景、硬件、性能需求等多因素综合考虑，选择合适的数据库。
<img width="986" alt="image" src="https://github.com/kalabiqlx/InternLM2-Tutorial-Assignment/assets/102224466/421f1daf-b573-47f5-9471-13d8d16ceee2">

**Embedding模型：**

* [ChatGPT-Embedding](https://link.zhihu.com/?target=https%3A//platform.openai.com/docs/guides/embeddings/what-are-embeddings)
* [ERNIE-Embedding V1](https://link.zhihu.com/?target=https%3A//cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu)
* [M3E](https://link.zhihu.com/?target=https%3A//huggingface.co/moka-ai/m3e-base)
* [BGE](https://link.zhihu.com/?target=https%3A//huggingface.co/BAAI/bge-base-en-v1.5)

**应用阶段：** 在应用阶段，我们根据用户的提问，通过高效的检索方法，召回与提问最相关的知识，并融入Prompt；大模型参考当前提问和相关知识，生成相应的答案。关键环节包括：数据检索、注入Prompt等。

* 数据检索:
	* 相似性检索：即计算查询向量与所有存储向量的相似性得分，返回得分高的记录。常见的相似性计算方法包括：余弦相似性、欧氏距离、曼哈顿距离等。
    	* 全文检索：全文检索是一种比较经典的检索方式，在数据存入时，通过关键词构建倒排索引；在检索时，通过关键词进行全文检索，找到对应的记录。
 
* 注入Prompt
  
Prompt作为大模型的直接输入，是影响模型输出准确率的关键因素之一。在RAG场景中，Prompt一般包括任务描述、背景知识（检索得到）、任务指令（一般是用户提问）等，根据任务场景和大模型性能，也可以在Prompt中适当加入其他指令优化大模型的输出。Prompt的设计只有方法、没有语法，比较依赖于个人经验，在实际应用过程中，往往需要根据大模型的实际输出进行针对性的Prompt调优。


