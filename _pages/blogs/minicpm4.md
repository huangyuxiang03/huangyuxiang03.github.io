permalink: /blogs_apb_zh
title: ""
layout: default
---

# 是时候在推理引擎中支持稀疏注意力了
2025/06 黄宇翔 @清华大学

原文发布在知乎中。[原问题链接](https://www.zhihu.com/question/1914321160193140575/answer/1914339822979512127)

---

## 当InfLLM-v2遇上推理引擎

我们目前开源出来的vLLM实现（已经合并到官方主分支中）能够支持模型的dense推理、投机采样、量化、量化加投机四个推理模式，但Minicpm4中最大的亮点之一——InfLLM-v2并没有合并到官方版vLLM中。

虽然没有把InfLLM-v2合并到vLLM或SGLang中，我们开源了cpm.cu框架，并且在cpm.cu中支持了InfLLM-v2。采用这个框架，而没有将稀疏attention整合到目前较为流行的推理引擎中，受到很多因素影响。首要因素是Minicpm系列作为端侧模型，目前我们更关心单batch推理的场景——这样看来的话，vLLM与SGLang中的其他部分在这个场景中略显臃肿。另一个因素是，现有的推理引擎中实现稀疏attention管理受到Serving场景的一些制约。

目前的推理引擎（vLLM、SGLang等一众优秀的框架）面向的是Serving场景，在云端部署，考虑大吞吐、大batch size，KV cache切块与重用等场景与特性。以我比较熟悉的vLLM框架为例，往其中去整合InfLLM-v2、NSA、MoBA这类Decoding友好的稀疏Attention实际上是非平凡的一件工作。

- 块表示的缓存：分块稀疏attention大多需要缓存prefill之后输入的块表示，而目前的vLLM似乎并不能通过简单的修改来支持这一点。在vLLM中，由于Block Table在管理的时候实际上不区分K Cache和V cache，而块表示可以看成一种没有V向量的K cache，所以直接把块表示插到块表里是不太现实的。另一种可行的实现思路是，从KV cache的块池划分一部分出来给块表示，并且设置一个新的块表来单独管理这些块表示。


- 非平凡的Offloading：以InfLLM-v2为代表的分块稀疏attention实际上是一种分层attention，除了块表示之外，实际的KV cache块并没有必要常驻显存。事实上，目前很多流行的offloading框架（例如ShadowKV）都是这样实现的。所以在推理引擎支持块表示缓存之后，需要实现设计一个cache scheduler来管理KV cache的offloading。


- Page Attention与稀疏Attention的结合：这两个的结合在原理上并没有任何不可行之处，但是目前社区没有一份非常好的高效实现。这里需要考虑的问题有：稀疏Attention的第一阶段（Q与块表示的attention）是否需要做page attention？第一阶段与第二阶段（实际的稀疏attention）的页大小是否应该相等？怎么安排每个page上的layout能够减少访存次数？能否通过设计统一的块表或者把块表示与对应cache放在同一块上来减少重复访存？由于这里的开放问题太多，我们期待未来出现一篇工作（或者是由我们来完成这样的工作）解释什么样的实现最利于分层attention与Page attention结合的。


在开发过程中，我基于vLLM改了一套内部的框架，能够支持InfLLM-v2。但由于上面的改动太多以及太过于剧烈，对vLLM其他部件的功能影响过大（这样的改动大概率是过不了PR的hhh），我们最后选择仅在内部使用这套框架，没有把它开源出来。**因此，我们在这里呼吁推理引擎的开源社区，尽早将稀疏attention与分层attention的精细支持纳入考虑之中~**

---

## 开始幻想：稀疏attention友好的下一代推理引擎

下面是我的一些关于稀疏attention与推理引擎的思考，思维比较发散，也不见得全部聚焦于单batch的场景。

- 像上文所说，好的推理引擎在未来一定必将会支持稀疏attention。attention的稀疏化已经在非常多工作上被证明有效，能够在几乎无性能损失的情况下大幅加速模型推理。所以为了支持稀疏attention，推理引擎的资源管理（块分配、块表管理等）、调度器、attention后端实现，都需要对稀疏attention做出一些改动。一些可行的想法已经在上面写了，不再赘述。


- Offloading和分块稀疏的结合非常自然。所有Offloading的技术都应该被合并到推理引擎中：计算与访存的overlap，对块是否会被选中的预测等。个人观点是这方面学术界的工作已经足够丰富，我们现在非常需要一个开源框架把它们都用起来。


- 重计算与Offloading的平衡：在RL以及serving场景下，显存大小仍然是一个非常显著的瓶颈，所以丢掉某些KV cache是很自然的做法。那么选择哪些KV cache来丢，就回到了学术界之前很关心的问题：如何做KV cache eviction，例如SnapKV、H2O等。虽然从去年下半年开始，eviction的工作就不太被看好，会被质疑无法做高难度的长文本检索任务，关注的几篇做的很好的工作（以及我自己的Locret）还在辗转于各大会议之间hhh，但我一直认为eviction仍然是一个很重要的话题，我们需要在尽量少信息的条件下决定什么KV块可以丢掉。在过去基于Full Attention做eviction的时代，此类工作会被质疑性能问题，但放在分块稀疏attention的时代，也许丢KV cache这个操作又会活过来。


- 灵活的推理范式：目前的推理引擎的并行策略和混合策略较为有限，但我们看到了近年来一些非常有启发性的思路。比如PowerInfer讨论的混合推理，NVIDIA在Star Attention（以及我最新的工作APB）中讨论的推理时稀疏与序列并行结合，也有可能是一种未来推理引擎的特性。

---

最近在参与团队工作的同时，自己也在思考（以及与同行交流）什么样的推理引擎是现在这个时代真正需要的。虽然理性上认为，一个大而全的框架是不可能的，端侧、serving、RL rollout中需要的特性各不相同，但是感性上我们还是希望会有一个框架能够很好地把他们都做到一起，并且解决掉臃肿的问题。推理框架范式的统一需要模型侧做一些面向系统结构的设计，在今年稀疏attention的成功可以认为是迈出了重要的一步。

近期一篇非常有意思的文章 [Kinetics](https://arxiv.org/pdf/2506.05333) 中通过详细的实验表达这样一个有意思的结论：稀疏attention，尤其是block sparse top-k attention在test time scaling里比dense模型更好。这正好印证了我们以及同行们在今年猛推分块分层稀疏attention这个技术方案的原因。这么说，我们的确需要开始把分层attention加到推理引擎中了。这篇文章其实还揭示了一个评测稀疏attention是否具有足够好性质的机会：我们能够在小模型上跑sparse attention的scaling实验，这何尝不是一种test time scaling的风洞试验？