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

## TL;DR

- The estimated cost assuming sufficient batching across users request is around `0.062` cents per 1k, which is `~3.2` times lower the API price at `0.2` cents. 
- Without batching across users, the cost can be as high as `0.7` cents, which is 3.5 higher than the offering price. We can see that batching is key.
- ChatGPT needs high volume of requests to offer such price, which is likely possible due to the popularity. Other competitors will require such economies of scale to provide such low-cost offering.



## GPT3.5 Turbo Pricing
As of March 1 2023, OpenAI announced GPT3.5 Turbo API, with $0.002 per 1k tokens, a drastic price drop from 0.02 per 1k tokens for Davinci. Let's analyze how much it really costs to host such a model and if this price is still profitable for OpenAI. We can also see why economies of scale is so important for LLM inference.

<!--
{% twitter https://twitter.com/AlphaSignalAI/status/1630994231887101958?s=20 %}
{% twitter https://twitter.com/AlphaSignalAI/status/1630997137805770769?s=20 %}
-->

{% twitter https://twitter.com/gdb/status/1630991925984755714?s=20 %} 

{% twitter https://twitter.com/theLionary/status/1631024073324253184?s=20 %}


## How Much Does it Cost to Process 1k Tokens?
In this blog we plan to analyze how low the cost per 1k tokens can be based on the GPU FLOPs requirement of GPT models.

### Estimating Cost via Tensor Operations


* According to the scaling laws paper, each forward pass for the context computation requires `2DN` tensor operations where `D` is the number of tokens, and `N` is the number of parameters.
  * Reference: https://arxiv.org/abs/2001.08361
  * This amounts to `700 Tera TOPs` (tensor operations). The reason I do not say FLOPs is because we do not assume that OpenAI uses floating point operations, since they likely will use quantization with int8. (or even int4?)
* For simplicity, assume that OpenAI uses 1 node with 8 A100 GPUs to perform such inference with tensor parallel. However, the analysis still applies if the TP degree is higher.
  * The cost of 1 node per hour is `~$12` according to the lowest price I found. With a long-term deal between OpenAI and Microsoft, this cost can be lower.
* 8 GPUs has a throughput of `8 * T` where `T = 313 TFLOPs` (bf16 or fp16), or `T = 624 TOPs` with int8. 
  * Reference: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
  * This number does not take into account sparsity, which can make the TOPs higher.
* To perform `2DN` TOPs, the time required is `2DN/8T`
  * With `D = 1k` tokens, `N = 175B` parameters, and `T = 624 TOPs/sec`, the time required per 1k tokens with maximum flops is `0.07` seconds.
* With `$12` per hour, the cost of processing `1k` tokens is `$12/3600 * 0.07` which is `0.023` cents.
* In practice, attaining the 100% efficiency is impractical. Let `e` be the efficiency, which is realistically around `e ~ 30%`.
* The final cost becomes

$$ \frac{0.023}{e} \text{ cents / 1k token} $$


* Example: If `e = 40%`, the cost becomes `0.057` cents per 1k tokens.
* **At the current price of `0.2` cents per 1k tokens, it is 3.5X more expensive that the cost of `0.057` cents. This translates to ~2.5 times profit.** 
* **Caveat** This takes into account only the context encoding time. The incremental decoding time is generally less efficient, but if things are batched well across users, it is possible to attain the same efficiency as the context computation cases. 

<!--
it is possible that the above analysis based on FLOPs apply. (I will have the check if the forward for incremental decoding is actually `2DN`)
-->



### Taking into Account Incremental Decoding


First, let's do a realistic estimate of the incremental decoding flop efficiency.
According to BLOOM inference, the throughput is `0.71 ms` per token per batch (batch size 128 which gives the highest efficiency before OOM). This is the best case scenario since the input is only a few tokens, meaning that we don't have to load large previous $$K$$ and $$V$$ tensors. The amounts to doing `2N*1` TOPs, which means that the per GPU TOPs/second is `350 / (0.71 * 8) = 61.6 Tera Flops / GPU`, which equals to around 20% efficiency. In reality when then $$K,V$$ tensors are high, this efficiency will be lower due to memory IO cost. Here, we can see that if the batch is high, then it is possible to attain efficiency ~20%, which is quite high. The cost for incremental decoding is almost the same as the context computation with sufficient batching (in the best case of short context). Overall, since we use `e=30%` for context computation, the efficiency of `20%` for incremental decoding likely does not affect the efficiency much. That is, at `e=20%`, the cost to generate 1000 tokens would be

$$ \frac{0.023}{e} \text{ cents / 1k token} = 0.115 $$

Overall, the cost to process input with context length C and generate T steps (C and T are in the unit of 1k tokens) is 

$$ \text{total process cost per 1k tokens} =  (0.057 C + 0.115 T)/(C+T) $$

Using $$C=1$$ and $$T=0.1$$ (generate 100 tokens based on 1k context length), then the cost to process 1k tokens on average is `0.062`, still pretty close to `0.57`. The profit would be 3.2x.




### Inefficient Case with Batch Size = 1

With batch size = 1, however, the efficiency would be much lower. For instance, it takes `44 ms` per token for batch size 1, which is 62 times less efficient compared to `0.71 ms` per token processing time in batch. This would equate to tensor operation efficiency of `e = 20% / 62 ~ 0.32%`.

The (inefficient) cost to generate 1000 tokens with batch size 1 would be 

$$ \frac{0.023}{0.32 \% } = 7.2 \text{ cents} >> 0.076 \text{ cents}$$

which is much higher than in the context computation case, almost 100 times higher. 


$$ \text{total process cost per 1k tokens (batch 1)} =  (0.057 C + 7.2 T)/(C+T) $$


* Again, let's assume C=1, T=0.1, then the amortized cost is `0.71` cents, **much higher than 0.2 cents**.


* Note: My previous blog on multi-query outlines the comparative inefficiency of incremental decoding versus context computation as a function of batch size 
[link]({% post_url 2023-02-01-multi-query-attention %}).


## Summary

* **Batching across users is key**. 
With economies of scale which OpenAI likely has due to ChatGPT's popularity, sufficient batching by aggregating multiple users request makes the inference becomes quite cheap at ~`0.076` cents per 1k tokens. Moreover, the requests can be grouped by how long the inputs are to avoid any inefficiency due to padding with the batch inference.
* Other competitor's inference cost likely lies between `7` cents per 1k tokens and `0.076` cents, depending on how many users at a given time which dictates the batch size used.