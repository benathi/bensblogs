---
layout: distill
title: OpenAI Still Makes 2X Profits on ChatGPT at 0.2 Cents Per 1K Tokens
date:   2023-03-03
description: Estimating ChatGPT inference cost based on FLOPs.
tags: chatgpt inference
categories: chatgpt transformers
published: true
giscus_comments: true

authors:
  - name: Ben Athiwaratkun 
    url: https://benathi.github.io
#    affiliations:
#      name: AWS AI Labs


bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
#  - name: Notation
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
#  - name: Tensor Operations
#  - name: Multi-Head Attention
#  - subsections:
#    - name: Context Computation
#    - name: Incremental Decoding



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



## GPT3.5 Turbo
As of March 1 2023, OpenAI announced GPT3.5 Turbo API, with $0.002 per 1k tokens, a drastic price drop from 0.02 per 1k tokens for Davinci. Let's analyze how much it really costs to host such a model and if this price is still profitable for OpenAI.

<!--
{% twitter https://twitter.com/AlphaSignalAI/status/1630994231887101958?s=20 %}
{% twitter https://twitter.com/AlphaSignalAI/status/1630997137805770769?s=20 %}
-->

{% twitter https://twitter.com/theLionary/status/1631024073324253184?s=20 %}


## What's the price at cost?

In this blog we plan to analyze how low the cost per 1k tokens can be based on the GPU FLOPs requirement of GPT models.


* According to the scaling laws paper, each forward pass for the context computation requires `2DN` tensor operations where `D` is the number of tokens, and `N` is the number of parameters.
  * Reference: https://arxiv.org/abs/2001.08361
  * This amounts to `700` Tera TOPs (tensor operations). The reason I do not use FLOPs is because we do not assume that OpenAI uses floating point operations, since they likely will use quantization with int8, at least for some operations.
* Assume that OpenAI uses 1 node with 8 A100 GPUs to perform such inference with tensor parallel. However, the analysis also applies if the TP degree is higher.
  * The cost of 1 node per hour is `~$12` according to the lowest price I found. With a long-term deal with Microsoft, this cost can be lower.
* 8 GPUs has a throughput of `8 * T` where $$T = 313$$ for floating point operations (bf16 or fp16), or $$T = 624$$ with int8. These numbers can be doubled with sparsity.
  * Reference: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
  * Note: It's unclear what sparsity means in this case.
* To perform `2DN` TOPs, the time required is `2DN/8T`
  * With `D = 1k` tokens, `N = 175B` parameters, and `T = 312 TFLOP/sec`, the time required per 1k tokens with maximum flops is `0.14` seconds.
* With `$12` per hour, the cost of processing `1k` tokens is `$12/3600 * 0.14` which is `0.046` cents.
* In practice, attaining the 100% efficiency is impractical. Let `e` be the efficiency, which is realistically around `e ~ 30%`. 
* Also let `p` be the efficiency multiplier in case of int8, or future int8, or sparsity. For instance, `p=2` for int8.
* The final cost becomes

$$ \frac{0.046}{e p} \text{ cents / 1k token} $$


* Example: If `e = 30%` and `p=2` due to int8, the cost becomes `0.076` cents per 1k tokens.
* At the current price of `0.2` cents per 1k tokens, it is 3X more expensive that the cost. This translates to 2 times profit, or `66%` margin.
* **Caveat** This takes into account only the context encoding time. The incremental decoding time is generally less efficient, but if things are batched well across users, it is possible to attain the same efficiency as the context computation cases. In fact, I would not put it past OpenAI to optimize such efficiency given their engineering prowess. 
* **Batching across users is key**. This is likely possible for ChatGPT given how popular. If OpenAI needs to aggregate ~10 requests, they can likely do that without having to wait long. Moreover, the requests can be grouped by how long the inputs are to avoid padding with the batch inference.
<!--
it is possible that the above analysis based on FLOPs apply. (I will have the check if the forward for incremental decoding is actually `2DN`)
-->
