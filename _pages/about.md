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

<!-- I am a 4th year undergraduate student of Dept. of Computer Science and Technology of Tsinghua University, Beijing, PRC, with a 3.92/4.00 overall GPA.  -->
I am an incoming Ph.D. student (starting Fall 2025) in the [TsinghuaNLP Group](https://nlp.csai.tsinghua.edu.cn/) at Tsinghua University, Beijing, under the supervision of [Prof. Zhiyuan Liu](https://nlp.csai.tsinghua.edu.cn/~lzy/). Prior to this, I obtained a B.Eng. degree from the Department of Computer Science and Technology at Tsinghua University. My research focuses on efficient AI and machine learning systems, particularly in the area of LLM inference systems. Currently, I am working on developing efficient algorithms and system frameworks for long-context processing to enhance LLM inference speed. My research spans model compression, speculative decoding, and long-context inference acceleration, which I believe are critical to improving the efficiency of LLM systems. In the future, I aim to explore long-CoT inference acceleration as a promising direction.

I am a strong advocate of the idea that the scaling law, especially test-time scaling is a pathway to AGI. I believe efficiency is the key to scaling. I have published papers at ACL, EMNLP and COLM, with a citation count of <a href='https://scholar.google.com/citations?user=nvCXW78AAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>.

- News: APB has been selected as **ACL Oral Paper**.

- News: My bachelor thesis has been selected as **outstanding bachelor thesis** of Tsinghua University. 

- News: Graduate with honor (outstanding graduate of Dept. of CST, Tsinghua University).

- News: Two papers (APB, FR-Spec) accepted by ACL 2025 main.

- News: We are releasing our new long-context inference acceleration method APB. 10x speedup without any performance degradation!

<!-- - News: One paper (Ouroboros) accepted by EMNLP 2024 main. -->

<!-- - News: One paper (CA-LoRA) accepted by COLM. The first COLM was super great! -->

<!-- - News: Recently I have been working on efficient decoding algorithms. We have released  "Ouroboros", a new Speculative Decoding algorithm with Large Model Enhanced Drafting. Please refer to [Paper](https://arxiv.org/pdf/2402.13720.pdf) and [Code](https://github.com/thunlp/Ouroboros). It achieves speedups of up to $1.9\times$ and $2.8\times$ compared to lookahead decoding and speculative decoding, without any training. -->

<!-- - News: I am involved in the [MiniCPM](https://github.com/OpenBMB/MiniCPM) project of ModelBest Inc., OpenBMB and THUNLP. It is an end-side LLM outperforms Llama2-13B. I am responsible to model inference.  -->

# Publications and Preprints 

**Huang, Y.** (2025). On Accelerating Long-Context Inference via Sparse Self-Attention. B.Eng dissertation, Tsinghua University.

MiniCPM Team. (2025). [MiniCPM4: Ultra-Efficient LLMs on End Devices](https://arxiv.org/pdf/2506.07900) arXiv preprint arXiv:2506:07900.

**Huang, Y.<sup>*</sup>**, Li, M.<sup>*</sup>, Han, X., Xiao, C., Zhao, W., Sun, A., Zhou, J., Zhou, H., Liu, Z., & Sun, M. (2025). [APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs.](https://arxiv.org/pdf/2502.12085) arXiv preprint arXiv:2502.12085 (**ACL** 2025 main <span style="color: red;"><strong>Oral</strong></span>).

Zhao, W.<sup>*</sup>, Pan, T.<sup>*</sup>, Han, X., Zhang, Y., Sun, A., **Huang, Y.**, Zhang, K., Zhao, W., Li, Y., Wang, J. & Liu, Z. (2025). [FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling.](https://arxiv.org/pdf/2502.14856) arXiv preprint arXiv:2502.14856 (**ACL** 2025 main).

Yuan, Z., Li, J., Li, Y., **Huang, Y.**, Chen, C., Wang, S., & Gou, Z. (2025). CITR: Efficient Long Video Understanding Needs Causal Importance. (**ACM MM**).

**Huang, Y.**, Yuan, B., Han, X., Xiao, C., & Liu, Z. (2024). [Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads.](https://arxiv.org/pdf/2410.01805) arXiv preprint arXiv:2410.01805 (In submission).

Zhao, W.<sup>*</sup>, **Huang, Y.<sup>*</sup>**, Han, X., Xu, W., Xiao, C., Zhang, X., Fang, Y., Zhang, K., Liu, Z., & Sun, M. (2024). [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting.](https://aclanthology.org/2024.emnlp-main.742.pdf) Main Conference of Empirical Methods in Natural Language Processing (**EMNLP** 2024 main).

Zhao, W.<sup>*</sup>, **Huang, Y.<sup>*</sup>**, Han, X., Liu, Z., Zhang, Z., Li, K., Chen, C., Yang, T., & Sun, M. (2024). [CA-LoRA: Adapting Existing LoRA for Compressed LLMs to
Enable Efficient Multi-Tasking on Personal Devices.](https://openreview.net/pdf?id=kpf7UbnSAm) Conference on Language Modeling (**COLM** 2024).

Hu, S., Tu, Y., Han, X., Cui, G., He, C., Zhao, W., ... & Sun, M. (2024). [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://openreview.net/pdf?id=3X2L2TFr0f) Conference on Language Modeling (**COLM** 2024).

Qin, Y., Hu, S., Lin, Y., Chen, W., Ding, N., Cui, G., ... & Sun, M. (2023). [Tool Learning with Foundation Models.](https://arxiv.org/pdf/2304.08354.pdf) **ACM Computing Surveys**.

Xiao, J., **Huang, Y.**, Hu, C., Song, S., Huang, X., & Wang, J. (2022). [Time series data encoding for efficient storage: a comparative analysis in Apache IoTDB.](https://www.vldb.org/pvldb/vol15/p2148-song.pdf) Proceedings of the **VLDB** Endowment, 15(10), 2148-2160 (**VLDB** 2023).

(Note: <sup>*</sup> indicates equal contribution.)

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

# Service and Voluntary Work

- Reviewer: COLM 2025, ICLR 2025, ACL ARR 2023 December Cycle.

- Maintainer: [MiniCPM](https://github.com/OpenBMB/MiniCPM) github repository

- Maintainer: [Ouroboros](https://github.com/thunlp/Ouroboros) github repository

- *2022 autumn - 2023 spring*: Supporting education for Qinghai University, involved in *The foundation of Programming (higher level)* teaching. Lecture 1: [Search](https://cloud.tsinghua.edu.cn/f/a32ef2f86127456abb43/?dl=1) (In Chinese). Lecture 2: [Graphs and Trees](https://cloud.tsinghua.edu.cn/f/a8a5b591cb6649a78936/?dl=1) (In Chinese).

# Collaborators

I work closely with the MLSys guys ([Xu Han](https://thucsthanxu13.github.io/) (Research Assist. Prof.), [Weilin Zhao](https://achazwl.github.io/) (PhD. Candidate) and [Ao Sun](https://maydomine.github.io) (M.Sc. Student)) in my lab, TsinghuaNLP Lab. If you want to work with me as a collaborator, please feel free to reach out. I'm also welcoming to all kinds of discussion, e.g. MLSys, AI, academic choice, daily life, etc. Here, I'd like to list the people I have collaborated with.

- Mingye Li @Central South Univ. (2024.Summer - Now)
- Jicheng Han @Tsinghua Univ. (2025.Summer - Now)

# More

- Recently I find taking notes with LaTeX is fun on maths or math-related cs courses, so I created this repository: [CourseNotes](https://github.com/huangyuxiang03/CourseNotes). If you are looking for some learning materials of THU CST courses, please reach to the repository. If you are also taking notes with LaTeX, just contact me!

- I was an exchange student at University of Washington in 2023 Autumn. The experience was amazing of being an oversea exchange student. If you want to exchange at UW or Tsinghua and want to talk to someone, I am always pleasure to chat. (TL;DR: If you want to exchange at UW, you must be nominated by your home institution; for Tsinghua Univerisity, exchange students cannot be Chinese citizens. For other things you are not sure, just ask me!)

- I speak Chinese and English and I am recently learning German (yes, I want to write Deutsch at first then I realized I'm writing English). You can contact me freely within Chinese or English. German... probably several years after then I could verstehe was du geschrieben :)