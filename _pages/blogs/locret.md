---
permalink: /blogs_locret
title: ""
layout: default
---

# Locret: Enabling Long-Context Inference on Personal Devices
2024/09 Yuxiang Huang @Tsinghua & HKUST

**TL;DR:** We introduce **Locret**, a light-weight training-based KV cache compression method conducted by chunked prefill along with cache eviction. Locret achieves $20\times$ and $8\times$ KV cache compression ratio compared to the full cache for Phi-3-mini-128K and Llama-3.1-8B-instruct, respectively, with only <1 GPU hours training. Locret is robust and can be combined with multiple efficient inference approaches. To the best of our knowledge, Locret is the first framework capable of deploying Llama-3.1-8B or similar models on a single Nvidia 4090 GPU, enabling 128K long-context inference without compromising generation quality, and requiring little additional system optimizations.

<div id="framework" style="text-align: center;">
  <img src="https://raw.githubusercontent.com/huangyuxiang03/huangyuxiang03.github.io/refs/heads/main/_pages/blogs/assets/locret/pattern.png" alt="desc" style="width: 65%;">
  <figcaption>Figure 1: The framework of Locret.</figcaption>
</div>

---

## Background

### Consumer-Grade Devices, End-Side LLMs and Long-Context Inference

In recent years, we are observing a blooming trend of the development of Large Language Models (LLMs). Models exhibits a stronger performance on nearly all domains, and practioners start to develop more compact models designed for consumer-grade devices.

**Consumer-Grade Devices.** To obtain a better experience on LLM usage, hardware providers are designing and manufacturing cheaper and smaller GPUs, or implementing GPU and NPUs on a single SoC to reduce the overall cost for AI models. For example, Nvidia 4090 only has 24GB GPU memory and can be install on personal computers, while the price is controlled under $2000 U.S. dollars. Companies such as Apple and Qualcomm are now providing devices that is specially optimized for AI compute loads, such as matrix multiplication and sparse operations. However, such devices still suffer from severe inadequate GPU memory and limited compute power.

**End-Side LLMs.** To tackle the obstacles of memory and compute, end-side LLMs are designed and trained to serve a delicate AI service on the user's side. Such models usually has a compact size less than 8B, usually around 1B-3B. MiniCPM consist of a series of  1.2B, 2.4B and 4B models, Phi-3-mini has a size around 3B and Llama-3.2 series is equipped with 1B and 3B models. Despite the reduced size, such LLMs exhibits strong ability and are often compared with 7B-8B models. To support more complicated tasks, e.g. multi-hop reasoning, needle-in-a-haystack and AI-driven operating systems, the context lengths of such models are often extended. MiniCPM-3-4B is capable to process 32K context, Phi-3-mini-128K and Llama-3.2-1B/3B even support 128K long-context inference, making it possible for end-side LLMs to obtain long-context abilities.

**Long-Context Inference.** Compared to traditional short-context LLM inference, long-context LLM inference shifts the computing paradigm in two key ways: 

- Increased computational overhead for attention mechanisms
    
    As context length grows, the computation required for obtaining attention scores increases quadratically, which results in a higher ratio of the computational budget in a transformer block;
- Higher memory footprint for key-value (KV) caching

    Longer contexts require larger KV caches, which dramatically increases the peak memory usage.

These shifts demand innovative techniques to mitigate computational costs and manage memory usage effectively for long-context LLM inference. Consumer-grade devices cannot provide such memory space for KV cache, making the usage of end-side LLM obeying its original design. From such consideration, developing a KV cache compression algorithm for long-context inference on consumer-grade devices is vital for the democratization of LLMs.

### Existing Efficient Inference Approaches

### Inference Complexity

## Locret

### Training to Evict

### Inference with Retaining Heads


## Budget-Constrainted Long-Context Inference

## Acknowlegement

We gratefully appreciate the following indivisuals for there contribution. Without them, this project would be impossible to conduct.

- [Binhang Yuan](https://binhangyuan.github.io/site/) (Prof. @HKUST) The advisor of this project during Yuxiang's internship at HKUST.
- [Xu Han](https://thucsthanxu13.github.io/) (Research Prof. @Tsinghua) and [Zhiyuan Liu](https://nlp.csai.tsinghua.edu.cn/~lzy/) (Prof. @Tsinghua) The advisors from THUNLP lab, who provided so much useful assists and valuable opinions.
- [Chaojun Xiao](https://xcjthu.github.io/) (PhD Stu. @Tsinghua) The author of InfLLM, who provided valuable advices and proofread the paper. 
- [Ruisi Cai](https://cairuisi.github.io/) (PhD Stu. @UT Austin) The author of LoCoCo, who offered aid to training LoCoCo and provided suggestions on extending LoCoCo to Llama-3.1 series.
- [Xinrong Zhang](https://scholar.google.com/citations?hl=en&user=IvTrgR0AAAAJ) (PhD Stu. @Tsinghua) The author of InfiniteBench, who provided insights of the original design of the benchmark InfiniteBench.
- [Weilin Zhao](https://achazwl.github.io/), [Chenyang Song](https://scholar.google.com/citations?user=4L39cy0AAAAJ&hl=en&oi=ao), [Shuo Wang](https://scholar.google.com/citations?user=5vm5yAMAAAAJ&hl=en&oi=ao) and [Yuan Yao](https://yaoyuanthu.github.io/) for valuable discussions.

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