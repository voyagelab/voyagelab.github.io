---
layout: page
title: Links
description: 共同打造属于我们的博客
comments: false
menu: 链接
permalink: /links/
---

> 数据改变企业决策，数据改善人类生活

{% for link in site.data.links %}
* [{{ link.name }}]({{ link.url }})
{% endfor %}
