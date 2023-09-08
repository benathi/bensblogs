---
layout: distill
title: A Note on Efficient Einstein Summation with CUBLAS GEMM
date:   2023-03-18
description: 
tags: cuda
# categories: 
published: false
social: true
giscus_comments: true

authors:
  - name: Ben Athiwaratkun 
    url: https://benathi.github.io



bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
#toc:
#  - name: 


_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
---

## Intro

- why we need GEMM / cublas. low latency and etc
- look at how cublas work from a baby step perspective




## Practical
- Applying it to optimize latency in a real-world scenario with DeepSpeed Inference
- 