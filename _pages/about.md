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

<span class='anchor' id='about-me'></span>

I am a 3rd year undergraduate student of Dept. of Computer Science and Technology of Tsinghua University, Beijing, PRC, with a 3.91/4 overall GPA. My research interest is efficient AI and machine learning systems, recently working on efficient large language models and parameter efficient tuning. I am interested in inference speed enhancement and model compression methods. Currently working in THUNLP with Weilin Zhao (PhD. Student), Xu Han (Postdoc) and Zhiyuan Liu (Professor).

- News: Recently I have been working on efficient decoding algorithms. We have released  "Ouroboros", a new Speculative Decoding algorithm with Large Model Enhanced Drafting. Please refer to [Paper](https://arxiv.org/pdf/2402.13720.pdf) and [Code](https://github.com/thunlp/Ouroboros). It achieves speedups of up to $1.9\times$ and $2.8\times$ compared to lookahead decoding and speculative decoding, without any training.

- News: I am involved in the [MiniCPM](https://github.com/OpenBMB/MiniCPM) project of ModelBest Inc., OpenBMB and THUNLP. It is an end-side LLM outperforms Llama2-13B. I am responsible to model inference. 

# Publications and Preprints 

Zhao, W.$^*$, **Huang, Y.$^*$**, Han, X., Xiao, C., Liu, Z., & Sun, M. (2024). [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting.](https://arxiv.org/pdf/2402.13720.pdf) arXiv preprint arXiv:2402.13720

Hu, S., Tu, Y., Han, X., Cui, G., He, C., Zhao, W., ... & Sun, M. (2024). [MiniCPM: Unveiling the Potential of End-side Large Language Models.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20) OpenBMB Blog.

Zhao, W.$^*$, **Huang, Y.$^*$**, Han, X., Liu, Z., Zhang, Z., & Sun, M. (2023). [CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models.](https://arxiv.org/pdf/2307.07705.pdf) arXiv preprint arXiv:2307.07705.

Qin, Y., Hu, S., Lin, Y., Chen, W., Ding, N., Cui, G., ... & Sun, M. (2023). [Tool Learning with Foundation Models.](https://arxiv.org/pdf/2304.08354.pdf) arXiv preprint arXiv:2304.08354.


Xiao, J., **Huang, Y.**, Hu, C., Song, S., Huang, X., & Wang, J. (2022). [Time series data encoding for efficient storage: a comparative analysis in Apache IoTDB.](https://www.vldb.org/pvldb/vol15/p2148-song.pdf) Proceedings of the VLDB Endowment, 15(10), 2148-2160.

(Note: $^*$ indicates equal contribution.)

# Research Experiences
- *2022.07-now*: Working in THUNLP, dept. of CST. Topiced *efficient LLMs*.
- *2021.10-2022.07*, SRT (Student Research Training): Worked at School of Software, topiced *compression algorithms in big data database*, advised by Prof. Shaoxu Song.

# Honors and Awards
- Academic Excellence in Research Award of Dept. of CST, 2022.09-2023.07
- Comprehensive Scholarship (Scholarship from Prof. Zesheng Tang) of Dept. of CST, 2021.09-2022.07 
- The third prize, the 40th Tsinghua Challenge Cup

# Educations
- *2021.09-now*, Tsinghua University, Beijing, China. Undergraduate Student.
- *2023.09-2023.12*, University of Washington, Seattle, U.S.A. Exchange Student at School of Arts and Sciences.
- *2018.09-2021.07*, Beijing No.9 Middle School, Beijing, China. High school Student.

# Service and Voluntary Work

- Maintainer: [Ouroboros](https://github.com/thunlp/Ouroboros) github repository

- Maintainer: [MiniCPM](https://github.com/OpenBMB/MiniCPM) github repository

- *2022 autumn - 2023 spring*: Supporting education for Qinghai University, involved in *The foundation of Programming (higher level)* teaching. Lecture 1: [Search](https://cloud.tsinghua.edu.cn/f/a32ef2f86127456abb43/?dl=1) (In Chinese). Lecture 2: [Graphs and Trees](https://cloud.tsinghua.edu.cn/f/a8a5b591cb6649a78936/?dl=1) (In Chinese).

# More

- Recently I find taking notes with LaTeX is fun on maths or math-related cs courses, so I created this repository: [CourseNotes](https://github.com/huangyuxiang03/CourseNotes). If you are looking for some learning materials of THU CST courses, please reach to the repository. If you are also taking notes with LaTeX, just contact me!

- I was an exchange student at University of Washington in 2023 Autumn. The experience was amazing of being an oversea exchange student. If you want to exchange at UW or Tsinghua and want to talk to someone, I am always pleasure to chat. (TL;DR: If you want to exchange at UW, you must be nominated by your home institution; for Tsinghua Univerisity, exchange students cannot be Chinese citizens. For other things you are not sure, just ask me!)

- I speak Chinese and English and I am recently learning German (yes, I want to write Deutsch at first then I realized I'm writing English). You can contact me freely within Chinese or English. German... probably several years after then I could verstehe was du geschrieben :)