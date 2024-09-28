---
permalink: /blogs_locret
title: ""
layout: default
---

# Locret: Enabling Long-Context Inference on Personal Devices
2024/09 Yuxiang Huang @Tsinghua & HKUST

**TL;DR:** We introduce **Locret**, a light-weight training-based KV cache compression method conducted by chunked prefill along with cache eviction. Locret achieves $20\times$ and $8\times$ KV cache compression ratio compared to the full cache for Phi-3-mini-128K and Llama-3.1-8B-instruct, respectively, with only <1 GPU hours training. Locret is robust and can be combined with multiple efficient inference approaches. To the best of our knowledge, Locret is the first framework capable of deploying Llama-3.1-8B or similar models on a single Nvidia 4090 GPU, enabling 128K long-context inference without compromising generation quality, and requiring little additional system optimizations.

<div id="mem_task" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/locret/memory_acc.png" alt="desc" style="width: 100%;">
  <figcaption>Figure 1: Memory Statistics vs. Task Performance of Locret.</figcaption>
</div>

---

## Background

### Consumer-Grade Devices, End-Side LLMs and Long-Context Inference

In recent years, we are observing a blooming trend of the development of Large Language Models (LLMs). Models exhibits a stronger performance on nearly all domains, and practioners start to develop more compact models specially designed for consumer-grade devices.

**Consumer-Grade Devices.** To obtain a better experience on LLM usage, hardware providers are designing and manufacturing cheaper and smaller GPUs, or implementing GPUs and NPUs on a single SoC to reduce the overall cost for AI models. For example, Nvidia 4090 only has 24GB GPU memory and can be installed on personal computers, while the price is controlled under $2000 U.S. dollars. Companies such as Apple and Qualcomm are now providing devices that is specially optimized for AI compute loads, such as matrix multiplication and sparse operations. However, such devices still suffer from severe inadequate GPU memory and limited compute power.

**End-Side LLMs.** To tackle the obstacles of memory and compute, end-side LLMs are designed and trained to serve a delicate AI service on the user's side. Such models usually has a compact size less than 8B, usually around 1B-3B. MiniCPM consist of a series of  1.2B, 2.4B and 4B models, Phi-3-mini has a size around 3B and Llama-3.2 series is equipped with 1B and 3B models. Despite the reduced size, such LLMs exhibits strong ability and are often compared with 7B-8B models. To support more complicated tasks, e.g. multi-hop reasoning, needle-in-a-haystack and AI-driven operating systems, the context lengths of such models are often extended. MiniCPM-3-4B is capable to process 32K context, Phi-3-mini-128K and Llama-3.2-1B/3B even support 128K long-context inference, making it possible for end-side LLMs to obtain long-context abilities.

**Long-Context Inference.** Compared to traditional short-context LLM inference, long-context LLM inference shifts the computing paradigm in two key ways: 

- Increased computational overhead for attention mechanisms
    
    As context length grows, the computation required for obtaining attention scores increases quadratically, which results in a higher ratio of the computational budget in a transformer block;
- Higher memory footprint for key-value (KV) caching

    Longer contexts require larger KV caches, which dramatically increases the peak memory usage.

These shifts demand innovative techniques to mitigate computational costs and manage memory usage effectively for long-context LLM inference. Consumer-grade devices cannot provide such memory space for KV cache, making the usage of end-side LLM obeying its original design. From such consideration, developing a KV cache compression algorithm for long-context inference on consumer-grade devices is vital for the democratization of LLMs.

### Existing Efficient Inference Approaches

KV cache is always a bottleneck of inference throughput, for which a bunch of KV cache-centric efficient inference algorithms are proposed. We categorize them into *algorithm optimization* and *system optimization*

Algorithm Optimization:

- Quantization-based methods:

    The KV cache is stored in low-bit representations, e.g. 2-bits or 4-bits. Quantization can be applyed to each token indivisually or by channel. 

- Sparsity-based methods:

    No KV cache reduction is carried out. When computing the attention matrix, find some patterns according to heads or layers, and approach the complete matrix by only calculating limited entries.

- Token-dropping:
    - Eviction-based. A scoring function (usually designed manually) is designed to assess the importance of each token (or each cache unit), then evict the units with low importance score.
    - Token-merging (Attention Pool-based). A merging function is designed to mix multiple adjacent cache units into one single unit, e.g. StreamingLLM use a addition function to pool the cache units. 

System Optimizations:

- Offloading-base:

    Divide the full cache into chunks, and offload most of them to CPU or disk memory. Only the most related ones are retrieved to GPU at each chunk of chunked prefill.

- Hardaware-aware algorithms:

    Flash-attention and Page-attention utilize the architecture of modern GPUs to implement a memory efficient attention kernel, reducing the runtime peak GPU memory.

- Designing better infrastructures:

    More efficient programming languages, disaggregated inference frameworks can also enhance the efficiency of LLM long-context inference. 

We list the pros and cons of each way of implementing efficient long-context inference in the [Appendix](#appendix).




### Inference Spatial Complexity

From our perspective, existing technics of inference can be categorized according to their spatial complexity. Denote $n$ as the length of context and $c\geq 1$ is a constant.

- $O(n^2)$: Quadratic complexity, e.g. vanilla full KV cache inference. This is extremely heavy for all kind of devices in long-context inference scenarios.

- $O(c\times n)$: Linear complexity, e.g. full KV cache inference with chunked prefill. This complexity is very heavy for memory constrained scenarios, as the KV cache size increases when the context is longer.

- $O(n/c)$: Linear complexity with constant reduction, e.g. quantization, sparse attention, and most system optimizations. This complexity can solve the long-context inference problem to some extent, as the size of KV cache can be significantly reduced when the constant $c$ is large enough. However, this complexity is unacceptable when the context length grows even longer, to 128K or even 1M tokens. 

- $O(1)$: Constant complexity. This complexity can be implemented by two means: token dropping and RNN. Token dropping with a static budget size is $O(1)$, and RNNs such as Mamba and RWKV also have constant complexity during inference.

In order to tackle the long-context inference problem, we would like to find an algorithm that is $O(1)$. **Thus, designing a better scoring function to resolve the inaccuracy in existing eviction-based algorithms is our target.** Apart from designing the scoring function by hand, we introduce a training paradigm to learn an accurate scoring function.

---

## Locret

Here is the overall framework design of Locret, where we first find the importance scoring function by training, then we conduct eviction along with chunked prefill.

<div id="framework" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/locret/pattern.png" alt="desc" style="width: 60%;">
  <figcaption>Figure 2: The framework of Locret.</figcaption>
</div>

### Training to Evict

#### Retaining Head and Causal Importance Score

As drawn in Figure [2](#framework), we inject additional parameters, named **retaining head** (denote as $\mathbf{R}$ for simplicity), to each attention module. The retaining head is an FFN consist of two matrixes and one non-linear activation, i.e. 
$$\mathbf{R}(\mathbf{x}) = \sigma(\mathbf{xW_1})\mathbf{W_2}.$$

The input of the retaining head is the concatenation $[\mathbf{Q}, \mathbf{K}, \mathbf{V}]$, and output the number of KV heads values representing the importance, which we name **causal importance score (CIS)**. It is implemented as the following code (pytorch style).
```python
cis = self.retaining_head_w2(
    self.act(
        self.retaining_head_w1(
            torch.cat([h_q, h_k, h_v], dim=-1)
        )
    )
)
```

Formally, it is written as the following equation, where $\tilde S[k]_j^{(i)}$ is the CIS score of the $k$-th token at layer $i$ head $j$, i.e. `cis[:, k, j]` at layer $i$.
$$\tilde S[k]_j^{(i)} = \sigma([\mathbf{Q}, \mathbf{K}, \mathbf{V}]\mathbf{W}_1)\mathbf{W_2}$$

#### Training Object

We generate the labels of CIS as follows. The retaining heads are trained on a small Question-Answer SFT dataset, where each entry consists of a single prompt and one answer. The CIS label of the $k$-th token at layer $i$ head $j$ is 
$$\mathbf{S}[k]_j^{(i)} := \max_{n_q(d) \leq p \leq n_q(d) + n_a(d)}\left(\mathbf{Q}_j^{(i)}\mathbf{K}_{j}^{(i)T}\right)_{p, k}, $$
where $n_q(d)$ and $n_a(d)$ represent the lengths of the prompt and answer in data $d$.

Note that the number of heads between Q and KV is not same in GQA models, thus we aggregate the maximum value among all the heads in the same kv group as the CIS label.

By denoting the number of layers as $L$, number of heads as $h$, the training object is 
$$\argmin_{\mathbf{W_1}^{(i)}, \mathbf{W_2}^{(i)}, i=1, 2\cdots, L} \mathbb{E}_{d\in \mathcal{D}}\left[\sum_{i=1}^{L}\sum_{j=1}^{h}\sum_{k=1}^{n_q(d)}\mathcal{L}\left(\mathbf{\tilde S}[k]_j^{(i)}, 
    \mathbf{S}[k]_j^{(i)}
    \right)\right]$$

and the loss function $\mathcal{L}$ is 
$$\mathcal{L}\left(\mathbf{\tilde S}[k]_j^{(i)}, \mathbf{S}[k]_j^{(i)}\right) = \text{Smooth-}\mathcal{L}_1\left(\mathbf{\tilde S}[k]_j^{(i)}, \mathbf{S}[k]_j^{(i)}\right) + \alpha \mathcal{L}_2\left(\mathbf{\tilde S}[k]_j^{(i)}, \mathbf{\tilde S}[k-1]_j^{(i)}\right),$$
where Smooth-$\mathcal{L}_1$ is the smooth 1-norm and $\mathcal{L}_2$ represents the 2-norm.

Following the recipe above, we train the retaining heads on **LongAlpaca** for **3000 steps** only. **The training cost is less than 1 GPU hours.**

### Inference with Retaining Heads

Now we have an accurate scoring function that can predict the CIS. We adopt chunked prefill and perform cache eviction based on the predicted CIS.

As shown in Figure [2](#framework), we leave the last $n_s$ cache units at every head and every layer, named **stabilizers**, to obtain a better performance. We maintain a cache set with a static budget size $b$, and conduct chunked prefill. When the next chunk is process, we first calculate the CIS, then we assign $+\infty$ to the stabilizers. Then, we concatenate the cache provided in the current chunk with the cache set, and retain cache units with the highest $b-n_s$ CIS. By this way, the spatial complexity can be bounded to a constant, as the cache set has a constant size. The retaining heads are able to provide an accurate scoring funtion and retain the most important cache units towards latter operations. The pseudocode of Locret Inference is described in Algorithm [1](inference).

<div id="inference" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/locret/inference.png" alt="desc" style="width: 100%;">
</div>


---

## Budget-Constrainted Long-Context Inference


---

## Acknowlegement

We gratefully appreciate the following indivisuals for there contribution. Without them, this project would be impossible to conduct.

- [Binhang Yuan](https://binhangyuan.github.io/site/) (Prof. @HKUST) The advisor of this project during Yuxiang's internship at HKUST.
- [Xu Han](https://thucsthanxu13.github.io/) (Research Prof. @Tsinghua) and [Zhiyuan Liu](https://nlp.csai.tsinghua.edu.cn/~lzy/) (Prof. @Tsinghua) The advisors from THUNLP lab, who provided so much useful assists and valuable opinions.
- [Chaojun Xiao](https://xcjthu.github.io/) (PhD Stu. @Tsinghua) The author of InfLLM, who provided valuable advices and proofread the paper. 
- [Ruisi Cai](https://cairuisi.github.io/) (PhD Stu. @UT Austin) The author of LoCoCo, who offered aid to training LoCoCo and provided suggestions on extending LoCoCo to Llama-3.1 series.
- [Xinrong Zhang](https://scholar.google.com/citations?hl=en&user=IvTrgR0AAAAJ) (PhD Stu. @Tsinghua) The author of InfiniteBench, who provided insights of the original design of the benchmark InfiniteBench.
- [Weilin Zhao](https://achazwl.github.io/), [Chenyang Song](https://scholar.google.com/citations?user=4L39cy0AAAAJ&hl=en&oi=ao), [Shuo Wang](https://scholar.google.com/citations?user=5vm5yAMAAAAJ&hl=en&oi=ao) and [Yuan Yao](https://yaoyuanthu.github.io/) for valuable discussions.

---

## Citation

Please refer to our ArXiV [paper](TODO).

```
@article{huang2024locret,
  title={Locret: Accelerating Long-Context LLM Inference with Retaining Heads},
  author={Yuxiang Huang, Binhang Yuan, Xu Han, Chaojun Xiao, Zhiyuan Liu},
  journal={arXiv preprint arXiv:TODO},
  year={2024}
}
```

---

## Appendix

| Category | Type | Pros | Cons | Examples |
|-|-|-|-|-|
| Algorithm | Quantization | Barely no performance loss for >4-bits quantization. Easy to implement. | Severe performance loss at 2-bits. Slow inference speed. Need special hardware support. Constant size reduction of KV cache.| KIVI, KVQuant|
| Algorithm | Sparsification | Very fast inference speed. Low runtime GPU memory requirement for internal variables. | Cannot reduce the size of KV cache at all. Observable performance drop for denser models, e.g. MLA and GQA models. | MInference, FastGen|
| Algorithm | Token dropping - eviction | Fast inference speed and simple implementation. Memory usage can be bounded. | Severe performance degredation due to the inaccuracy of scoring functions. | H2O, SirLLM|
| Algorithm | Token dropping - merging | Memory usage can be bounded. | Additional training is required for some algorithms. Severe performance loss if the post training is inadequate. | StreamingLLM, LoCoCo |
| System | Offloading | Barely no performance degradation. | Very slow inference due to the limited I/O bandwidth. Delicate optimization of offloading is required. | InfLLM, FlexGen |
| System | Hardware-aware algorithms | High utility of hardware architecture, fast inference speed and no accuracy drop at all. | Cannot reduce the size of KV cache at all. Need to be specially adapted to each hardware architecture. | Flash-Attention, Page-Attention | 
| System | Better Infrastructures | Allows enterprise-level applications. | Extremely hard to develop. Low universality towards different scenarios. | KTransformers, HexGen |