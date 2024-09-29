---
permalink: /food
title: ""
excerpt: ""
author_profile: false
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}
{% assign c_url = gsDataBaseUrl | append: "canteen/breakfast.json" %}
{% assign d_url = gsDataBaseUrl | append: "canteen/lunch.json" %}
{% assign e_url = gsDataBaseUrl | append: "canteen/dinner.json" %}
{% assign f_url = gsDataBaseUrl | append: "canteen/lns.json" %}

# 今日餐厅推荐

<a href=''><img src="https://img.shields.io/endpoint?url={{ c_url | url_encode }}&labelColor=f6f6f6&color=9cf&style=flat&label=早餐"></a>

<a href=''><img src="https://img.shields.io/endpoint?url={{ d_url | url_encode }}&labelColor=f6f6f6&color=9cf&style=flat&label=午餐"></a>

<a href=''><img src="https://img.shields.io/endpoint?url={{ e_url | url_encode }}&labelColor=f6f6f6&color=9cf&style=flat&label=晚餐"></a>

<a href=''><img src="https://img.shields.io/endpoint?url={{ f_url | url_encode }}&labelColor=f6f6f6&color=9cf&style=flat&label=夜宵"></a>
