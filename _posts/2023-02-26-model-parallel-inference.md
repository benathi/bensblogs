---
layout: distill
title: Tensor Parallel Inference
date:   2022-11-17
description:  
tags: llm attention transformers gpt
categories: transformers
published: true


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
  - name: Overview
  - subsections:
    - name All-Reduce
      
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Attention Parallel
  - subsections:
    - name Output Parallel
    - name Input Parallel
    
  - name: MLP Parallel
  #- subsections:
    #  - name: Context Computation
    #  - name: Incremental Decoding


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


## Overview

### All-Reduce



<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/tensor-parallel.svg"
  class="img-fluid rounded z-depth-1"
%}
</div>

## Attention Parallel

<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/attention-with-tp.svg"
  class="img-fluid rounded z-depth-1"
%}
</div>

### Output Parallel

### Input Parallel



## MLP Parallel
<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/feedforward-tp.svg"
  class="img-fluid rounded z-depth-1"
%}
</div>
