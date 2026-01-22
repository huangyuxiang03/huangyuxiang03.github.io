---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}
{% assign c_url = gsDataBaseUrl | append: "canteen/breakfast.json" %}

<span class='anchor' id='about-me'></span>


I am a 1st year Ph.D. Student in the [TsinghuaNLP Group](https://nlp.csai.tsinghua.edu.cn/) at Tsinghua University, Beijing, under the supervision of [Prof. Zhiyuan Liu](https://nlp.csai.tsinghua.edu.cn/~lzy/). Prior to this, I obtained a B.Eng. degree from the Department of Computer Science and Technology at Tsinghua University. My research focuses on efficient AI and machine learning systems, particularly in the area of LLM inference systems. Currently, I am working on developing efficient algorithms and system frameworks for long-context processing to enhance LLM inference speed. My research spans model compression, speculative decoding, and long-context inference acceleration, which I believe are critical to improving the efficiency of LLM systems. In the future, I aim to explore long-CoT inference acceleration as a promising direction.

I am a strong advocate of the idea that the scaling law, especially test-time scaling is a pathway to AGI. I believe efficiency is the key to scaling. I have published papers at ACL, EMNLP and COLM, with a citation count of <a href='https://scholar.google.com/citations?user=nvCXW78AAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>.

- News: APB has been selected as **ACL Oral Paper**.

- News: My bachelor thesis has been selected as **outstanding bachelor thesis** of Tsinghua University. 

- News: Graduate with honor (outstanding graduate of Dept. of CST, Tsinghua University).

- News: Two papers (APB, FR-Spec) accepted by ACL 2025 main.

- News: We are releasing our new long-context inference acceleration method APB. 10x speedup without any performance degradation!

# Selected Publications

**Huang, Y.<sup>*</sup>**, Wang, P.<sup>*</sup>, Han, J., Zhao, W., Su, Z., Sun, A., Lyu, H., Zhao, H., Wang, Y., Xiao, C., Han, X., & Liu, Z. (2025). [NOSA: Native and Offloadable Sparse Attention.](https://arxiv.org/pdf/2510.13602) arXiv preprint arXiv:2510.13602.

**Huang, Y.**, Li, M., Han, X., Xiao, C., Zhao, W., Sun, A., Yuan, Z., Zhou, H., Meng, F., Liu, Z. (2026). Spava: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention. (In Submission)

**Huang, Y.<sup>*</sup>**, Li, M.<sup>*</sup>, Han, X., Xiao, C., Zhao, W., Sun, A., Zhou, J., Zhou, H., Liu, Z., & Sun, M. (2025). [APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs.](https://arxiv.org/pdf/2502.12085) arXiv preprint arXiv:2502.12085 (**ACL** 2025 main <span style="color: red;"><strong>Oral</strong></span>).

**Huang, Y.**, Yuan, B., Han, X., Xiao, C., & Liu, Z. (2025). [Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads.](https://arxiv.org/pdf/2410.01805) Transactions on Machine Learning Research (**TMLR** 2025).

Zhao, W.<sup>*</sup>, **Huang, Y.<sup>*</sup>**, Han, X., Xu, W., Xiao, C., Zhang, X., Fang, Y., Zhang, K., Liu, Z., & Sun, M. (2024). [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting.](https://aclanthology.org/2024.emnlp-main.742.pdf) Main Conference of Empirical Methods in Natural Language Processing (**EMNLP** 2024 main).

Zhao, W.<sup>*</sup>, **Huang, Y.<sup>*</sup>**, Han, X., Liu, Z., Zhang, Z., Li, K., Chen, C., Yang, T., & Sun, M. (2024). [CA-LoRA: Adapting Existing LoRA for Compressed LLMs to
Enable Efficient Multi-Tasking on Personal Devices.](https://openreview.net/pdf?id=kpf7UbnSAm) Conference on Language Modeling (**COLM** 2024).

Xiao, J., **Huang, Y.**, Hu, C., Song, S., Huang, X., & Wang, J. (2022). [Time series data encoding for efficient storage: a comparative analysis in Apache IoTDB.](https://www.vldb.org/pvldb/vol15/p2148-song.pdf) Proceedings of the **VLDB** Endowment, 15(10), 2148-2160 (**VLDB** 2023).

(Note: <sup>*</sup> indicates equal contribution.)

# Theses

**Huang, Y.** (2025). [On Accelerating Long-Context Inference via Sparse Self-Attention.](https://newetds.lib.tsinghua.edu.cn/qh/paper/summary?dbCode=UNDERGRADUATE&sysId=3194) B.Eng dissertation, Tsinghua University.


# Research Experiences
- *2022.07-now*: Working in THUNLP, dept. of CST. Topiced *efficient LLMs*.
- *2024.07-2024.09*: Research Internship at HKUST, topiced *LLM long-context inference*, advised by Prof. [Binhang Yuan](https://binhangyuan.github.io/site/).
- *2021.10-2022.07*, SRT (Student Research Training): Worked at School of Software, topiced *compression algorithms in big data database*, advised by Prof. Shaoxu Song.

# Honors and Awards
- Outstanding Bachelor Thesis of Tsinghua University, 2025.06.
- Outstanding Graduate of Dept. of CST, Tsinghua University, 2025.06.
- Academic Excellence Award of Dept. of CST, 2023.09-2024.07
- Academic Excellence in Research Award of Dept. of CST, 2022.09-2023.07
- Comprehensive Scholarship (Scholarship from Prof. Zesheng Tang) of Dept. of CST, 2021.09-2022.07 
- The third prize, the 40th Tsinghua Challenge Cup
- High School Graduation with Honor, Beijing No.9 Middle School, 2021.07

# Educations
- *2025.09-now*, Tsinghua University, Beijing, China. Ph.D. Student.
- *2021.09-2025.07*, Tsinghua University, Beijing, China. B.Eng.
- *2024.07-2024.09*, The Hong Kong University of Science and Technology, Sai Kung, Hong Kong S.A.R., China. Research Internship.
- *2023.09-2023.12*, University of Washington, Seattle, U.S.A. Exchange Student at School of Arts and Sciences.
- *2018.09-2021.07*, Beijing No.9 Middle School, Beijing, China. High school Student.

# Teaching and Service

- Teaching assistant: 
  - Towards Artifitial General Intelligence (00240401/00240411): 2025 Spring, 2026 Spring
  - Towards Artifitial General Intelligence Practice (00240421): 2025 Fall, 2026 Spring

- Reviewer: 
  - ICLR: 2025, 2026
  - COLM: 2025, 2026
  - AAAI: 2025
  - ACL ARR: Dec. 2023

- Supporting education for Qinghai University (2022 autumn - 2023 spring), involved in *The foundation of Programming (higher level)* teaching. Lecture 1: [Search](https://cloud.tsinghua.edu.cn/f/a32ef2f86127456abb43/?dl=1) (In Chinese). Lecture 2: [Graphs and Trees](https://cloud.tsinghua.edu.cn/f/a8a5b591cb6649a78936/?dl=1) (In Chinese).

# Collaborators

I work closely with the MLSys guys ([Xu Han](https://thucsthanxu13.github.io/) (Research Assist. Prof.), [Weilin Zhao](https://achazwl.github.io/) (PhD. Candidate) and [Ao Sun](https://maydomine.github.io) (M.Sc. Student)) in my lab, TsinghuaNLP Lab. If you want to work with me as a collaborator, please feel free to reach out. I'm also welcoming to all kinds of discussion, e.g. MLSys, AI, academic choice, daily life, etc. Here, I'd like to list the people I have collaborated with.

- Mingye Li @Central South Univ. (2024.Summer - 2025.Summer)
- Jicheng Han @Tsinghua Univ. (2025.Summer - Now)
- Pengjie Wang @Tsinghua Univ. (2025.Fall - Now)
