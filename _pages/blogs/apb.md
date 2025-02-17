---
permalink: /blogs_apb
title: ""
layout: default
---

# APB: 10x Lossless Speedup for Long-Context Inference
2025/02 Yuxiang Huang @Tsinghua, Mingye Li @CSU [[中文](apb_zh.md)]

**TL;DR:** We introduce **APB**, a sequence parallelism framework integrated with approximate attention mechanism, that is able to achieve 10x lossless speedup for long-context inference, without any performance degredation.

<div id="mem_task" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/acc-speed.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 1: Speed and performance tradeoff of Locret.</figcaption>
</div>

---

## On Accelerating Long-Context Prefill

The speed of long-context prefill of Transformer-based LLMs is bounded by the quadratic compute introduced by the attention mechanism. There are two major ways to enhance the efficiency of long-context prefill: **Enhancing Parallelism** and **Reducing Compute**.

- **Enhancing Parallelism**: One natural way to deal with compute-bound process is to spread the compute to multiple devices and enhance the parallelism. As one GPU's compute power is fully utilized, we can always increase the number of GPUs for more compute power. From this perspective, the key point to optimize long-context inference is to enhance parallelism and distribute the quadratic compute to multiple GPUs. Fortunately, parallelism is a well-studied topic in LLM training. Methods such as tensor parallelism, model parallelism, and sequence parallelism can all distribute the compute burden. For long-context optimization, the best choice would be sequence parallelism, as spliting the compute from a sequence level maintains its scalability towards extremely long inputs.

- **Reducing Compute**: Another way of accelerating long-context prefill is to apply sparsity, i.e. reducing compute. We can simple make a choice on where to compute in the attention score. There are tons of related works of reducing the attention compute, e.g. MInference, SnapKV, Locret, etc. Unlike enhancing parallelism, this type of methods normally introduces certain level of performance degradation. Overlooking the unimportant part in the sequence would result in failure in some tasks. However, it is still a very promising type of approaches to enhance efficiency.

So here comes the question. Is it possible to combine them together? How about reducing compute in a sequence parallelism framework?

The answer is partially yes. Here, we would like to introduce two pioneer attempts of doing this. Star Attention from NVIDIA directly remove all the communication in the sequence parallelism framework, and only local attention is conducted on each GPU. As we can imagine, this way would result in a large performance decay, as the LLMs are not trained this way. Fortunately, StreamingLLM tells everyone that attention sink is a good idea, that we only need to retain the initial tokens to recover most of the performance. Star Attention adopts this by prepending an *anchor block*, which is the initial tokens with length equal to the context block on each GPU, to each context block. By this way, Star Attention is able to achieve large speedups with 95% performance maintained. Another pioneer work is APE by CMU, which focus on parallelizing the process of RAG contexts. APE modifies attention by adjusting softmax temperature and applying scaling factors to recover the performance.

---

## Our Goal

Why don't we push the limits?

Let's build a faster, better performance long-context acceleration framework for general long-context tasks!

---

## Methodology of APB

<div id="framework" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/framework.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 2: Framework of APB.</figcaption>
</div>

Let's start from a basic sequence parallelism framework, where we evenly split the document to each devices (termed as hosts). 

- Adding anchor blocks: The anchor block introduced in Star Attention is a good idea. But do we need such a large anchor block? (The anchor block is as big as the context block in Star Attention.) We reduce the size of anchor block, and set its length to 1/4 or 1/8 of the context block.
- Solving distant semantic dependencies: One reason that Star Attention and APE exhibit performance decay is that they cannot handle distant semantic dependencies. If the context on subsequent hosts need to attend with the context located in previous hosts, that would be impossible for existing methods. We solve this problem by constructing the *passing blocks*, which are the essential KVs of the previous hosts. Each context block is compressed by a compressor, then the compressed block is sent to subsequent hosts. 
- Compressing the context block: Now, we have a context layout that each host only has access to a partial KV cache. Thus, existing KV compressors such as H2O or SnapKV are not compatible. However, this aligns with the design of [Locret](locret.md), that the assessment of KV importance only depends on its Q, K, and V. We pick the retaining heads introduced in Locret as the compressor.
- Providing more query-related information to Locret: As introduced in the paper of Locret, it is a causal method that conducts KV cache compression without the awareness of the query. However, in the sequence parallelism framework, we can provide the query without disturbing the whole sequence. We simply embed the query at the beginning of each anchor block, so they will be discarded with the anchor block once the prefill is finished. By this way, the retaining heads are able to identify the query-related KV cache more accurately.

---

The inference process of APB is described as follows.

- Context Splitting: The long-context document is evenly split to each host. An anchor block with query embedded is prepended at the front.
- Block Compression: We compress the KV cache of each host by the retaining heads of Locret.
- Communication: We apply an AllGather communication operation to the compressed KV cache. For each host, it takes the compressed KV cache sent from the previous hosts as the passing block.
- Computation: We implement the attention with a modified mask by a tailored Flash Attention kernel. The passing block is discarded after the attention calculation. 

---

## Faster, Better Performance on all Input Lengths

<div id="performance" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/varlen.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 3: APB on various input lengths.</figcaption>
</div>

APB is faster and obtains a better performance compared with all the baselines. The speed and performance are consistently outperforming on RULER with all input lengths (from 32K to 512K). Notably, the compute of APB is much lower compared with Star Attention and vanilla Flash Attention.

---

## Why is it Faster?


<div id="breakdown" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/apb/breakdown-tb.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 4: Breakdown Analysis.</figcaption>
</div>

- Reduced anchor block. Anchor block in Star Attention is introducing a lot of overheads. It is heavy in the attention computation (as it is as large as the context block), and it also introduces overheads in the FFN computation. A reduced anchor block can alleviate the burden in these parts.
- Passing the most essential parts. We only calculate attention on the compressed KV cache. For each context block, previous KV cache is compressed but the KV cache of itself is unchanged. Thus, we are able to reduce the compute while preserving accuracy.


---
## Citation
Please refer to our ArXiV [paper](https://arxiv.org/abs/2502.xxxxx).

```
@article{huang2025apb,
  title={APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs},
  author={Yuxiang Huang, Mingye Li, Xu Han, Chaojun Xiao, Weilin Zhao, Sun Ao, Hao Zhou, Jie Zhou, Zhiyuan Liu, Maosong Sun},
  journal={arXiv preprint arXiv:xxxx},
  year={2025}
}
```