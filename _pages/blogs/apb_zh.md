---
permalink: /blogs_apb_zh
title: ""
layout: default
---

# APB: 实现10倍无损的长文本推理加速
2025/02 黄宇翔 @清华大学，李明业 @中南大学 [[English](apb.md)]

**简要总结:** 我们介绍了 **APB**，一个整合了稀疏注意力机制的序列并行推理框架。APB能够在无性能损失的前提下达到10x的长文本推理加速比。

<div id="mem_task" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/acc-speed.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 1: APB的速度与性能</figcaption>
</div>

---

## 加速长文本预填充

长文本预填充的效率受到计算的制约。由于注意力机制的计算量与序列长度呈二次方关系，长文本的计算通常是计算-bound的。主流加速长文本预填充的路线有两种：**提升并行度**和**减少计算**。

- **提升并行度**：优化计算-bound的过程的一个常用思路是提升并行度。我们可以将注意力机制的计算分布在不同设备上来提升并行度。当一个GPU的算力被充分的利用时，简单的增加GPU的数量就可以增加有效算力。从这个角度来看，优化长文本预填充的关键在于如何跨GPU高效地提升并行度。幸运的是，这是一个在大模型训练中被广泛研究的话题。我们有各种各样的并行策略，包括张量并行、模型并行、序列并行等。对于长文本推理优化，最好的并行策略当属于序列并行，因为它不受模型架构的制约，具有很好的可扩展性，尤其是输入序列极长的时候。

- **减少计算**：另一个加速长文本预填充的方式是减少计算，也就是使用稀疏注意力。我们可以选择注意力矩阵中计算的位置，并不计算其他位置来减少整体的计算量。这方面的相关工作非常多，例如 MInference, SnapKV, Locret, 等。
不像提升并行度，这类方法通常会带来一定的性能损失。计算时忽略重要的上下文会导致无法处理某些任务。然而，这还是一个非常有效的加速长文本计算的方法。

那么问题来了。这两者有没有可能结合起来？如何在序列并行框架里减少计算？

第一个问题的答案是：“是，但不全是”。在这里，我们介绍两个试图这样做的先驱方法。英伟达提出的Star Attention直接去除了序列并行中的所有通信，并只计算每个GPU上局部上下文的注意力。可以想象到，这样计算会导致很大程度的性能损失，这是因为模型并不是这么训练的。好在StreamingLLM引入了注意力池这一概念，也就是保留序列的前若干token能够极大程度的恢复性能。Star Attention采取了这一方法，在每个GPU上的局部上下文块前拼接了输入开头的一部分token，数目与局部上下文中的token数目相同。通过这样的方法，Star Attention能够在保留95%的性能的前提下实现大幅度的加速。另一个先驱工作是卡内基梅隆大学提出的APE，关注了RAG场景下长文本预填充加速。APB通过调整注意力的softmax温度和增加放缩因子的方式来恢复性能。

---

## 我们的目标

为什么不进一步加速？

让我们构建一个更快、性能更好，且适配通用长文本任务的长文本加速方法！

---

## 方法

<div id="framework" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/framework.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 2: APB框架</figcaption>
</div>

让我们从最基础的序列并行框架开始，输入文档被均分到各个设备上。

- 增加Anchor block：Star Attention中引入的Anchor block（输入序列开始的若干token）能够极大恢复性能。但是我们真的需要这么庞大的anchor block吗？（Star Attention中的anchor block与局部上下文块一样大。）我们减少anchor block的大小，使其和上下文块的1/4或1/8一样大。
- 解决长距离语义依赖问题：Star Attention和APE在某些任务上性能下降的一个原因是它们无法处理长距离语义依赖。如果后面的设备上的上下文需要看到前面设备所持有的上下文，这将在传统方法上无法被处理。我们通过构建passing block的方式来解决这一问题。Passing block由前面设备上的重要KV对组成。每个上下文块先被压缩，然后将被压缩的上下文块通信到后续设备上来构建passing block。
- 压缩上下文块：在序列并行框架中，在不通信的前提下，每个设备只对自己持有的上下文有访问权限。因此，现存的KV Cache压缩算法（例如H2O和SnapKV）都不适用，因为它们依赖全序列的信息。然而，这个特点与[Locret](locret.md)一致，KV Cache重要性分数仅仅与对应KV对的Q, K, V相关。
- 给Locret提供更多查询相关信息：Locret是一个查询无关算法，因为计算重要性分数时无法看到查询。然而，在序列并行框架中，我们并没有Locret场景中那么强的显存限制，因此我们可以让Locret中的保留头看到查询。实现方法为，在anchor block中放入查询，这样当预填充结束时，这些查询可以随着anchor block一同被删除，不会影响整体计算的同时还能让保留头看到查询的内容。通过这种方式，保留头能够更精准的识别出查询相关的KV对，并通过通信机制传给后续设备。


---

APB的推理过程如下。
- 上下文分割：长文本被均匀的分到每个设备上，开头拼接一个anchor block，其中包含了查询问题。
- 上下文压缩：我们用Locret引入的保留头来压缩KV Cache。
- 通信：我们对压缩过的KV Cache施加一个AllGather算子。每个设备会拿到前序设备传来的压缩缓存，并构建passing block。
- 计算：我们使用一个特殊的Flash Attention Kernel来实现这个特殊的注意力机制。我们更改了注意力掩码的形状。Passing block在注意力计算结束后就被删除，不参与后续计算。

---

## 更快、性能更好的长文本推理

<div id="performance" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/varlen.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 3: 不同长度上的APB</figcaption>
</div>

APB相较于所有基线算法实现了更快、性能更好的推理。APB能够在所有长度的RULER基准测试上实现持续更好的速度和性能。注意到，APB的计算量要远低于Star Attention和传统的Flash Attention。


---

## 为什么APB更快？


<div id="breakdown" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/breakdown-tb.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 4: 时间分析</figcaption>
</div>

- 更小的anchor block。Star Attention中的anchor block引入了太多不必要的开销。它在注意力机制计算中占比很庞大，同时也在FFN计算中引入了很大的开销。一个更小的anchor block能够减轻这些开销。
- 只传递最重要的KV cache。我们仅计算了当前上下文块和前序上下文块中较为重要的一些KV对。对于设备上的上下文块而言，前序上下文块被压缩，而当前上下文块的KV cache被完整计算。这有助于减少计算的同时保持良好的性能。



---
## 引用
请引用我们的ArXiV[论文](https://arxiv.org/abs/2502.xxxxx)。

```
@article{huang2025apb,
  title={APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs},
  author={Yuxiang Huang, Mingye Li, Xu Han, Chaojun Xiao, Weilin Zhao, Sun Ao, Hao Zhou, Jie Zhou, Zhiyuan Liu, Maosong Sun},
  journal={arXiv preprint arXiv:xxxx},
  year={2025}
}
```
