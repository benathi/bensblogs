---
layout: distill
title: Parallelism in Model Training
date:   2023-09-08
description: 
tags: training
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

## Scaling Model Training

Outline
- Large model training needs some parallelism to make it viable
- 


## Data Parallelism: DeepSpeed versus FSDP



## sources
- https://github.com/microsoft/DeepSpeed/discussions/1911
- https://github.com/microsoft/DeepSpeed/discussions/3476
- https://engineering.fb.com/2021/07/15/open-source/fsdp/ 
    -- Myte Ott is the author
- https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f

- https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f
    -- very interesting article. Need some more time to read and think about what perspective I might have to offer.

- There's so much advance for Zero. Zero++ can use quantized weight communication now? how nice!
https://arxiv.org/pdf/2306.10209.pdf
-- Try using Zero ++ with Chat Model

-- https://arxiv.org/pdf/2306.10209.pdf
- https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/

-- Summary
https://docs.google.com/document/d/1TsYkDYtV6BKiCN9PAOirRAy3TrNDu2XncUZ5UZfaAKA/edit#heading=h.xl5ihve181vc


-- maybe cover RLHF training pipeline 
https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat

-- 