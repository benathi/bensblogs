---
layout: distill
title: The Illustrated Attention via Einstein Summation
date:   2022-11-15
description: Introduction to einsum with attention operations.
tags: llm attention transformers gpt
categories: transformers
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
  - name: Notation
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Tensor Operations
  - name: Multi-Head Attention
  - subsections:
    - name: Context Computation
    - name: Incremental Decoding



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



This blog aims to lay the groundwork for a series of deep-dive articles on transformers. We briefly introduce the notion of Einstein Summation (einsum) or generalized tensor product, which provides a convenient framework for thinking about how tensors interact. With the einsum notation, we will be able to see what each operation does without having to worry about technical implementation details such as which axes to transpose or permute. If you have not heard of it before, it may take some time to develop an understanding and become comfortable with it, but it can change your life in terms of how you think about tensor operations and make things much easier to understand in the long run. For a more detailed blog on einsum, you can check out [Einsum Is All You Need](https://rockt.github.io/2018/04/30/einsum).

## Notation
This section explains the notation that will be used in the following discussion.

* $$b$$: batch size.
* $$h$$: number of heads.
* $$k,v$$: dimension of value and key head. $$k=v$$ for transformers attention, but we use the different symbols for clarity.
* $$d$$: hidden dimension of the model where $$d=hk=hv$$.
* $$m$$: context length or key length.
* $$n$$: query length. For context computation, $$n=m$$. For incremental decoding, $$n=1$$. In the general form of attention, $$n$$ and $$m$$ can be any values (such as in the case of inference with deep prompt tuning where there's an initial key and value tensors before the context encoding step).
* $$x$$: input tensor to the attention layer. If the input to the query and key/value projections are different, we may denote them as $$X_Q$$ for the query input and $$X_K$$ for the key and value input.
* $$Q, K, V, O$$: query, key, value, and output tensors.
* $$P_Q, P_K, P_V, P_O$$: the projection matrices to produce $$Q, K, V, O$$ from input $x$.


## Tensor Operations

In this section, we seek to develop an intuition about the meaning of different einsum operations. This will help develop a deep understanding of the attention mechanism in the future. We will see that many familiar operations such as matrix multiplication or dot products can be described neatly with einsum.

### Einsum
We will use the notation $$ C= \langle A,B\rangle: \langle \text{shape}_A,\text{shape}_B \rangle \to \text{shape}_C$$ as the Einstein sum  between $$A$$ and $$B$$.
* Here, $$A$$ and $$B$$ are the input tensors, and this einsum specifies the way that the tensor $$C$$ is computed from $$A$$ and $$B$$, based on the given input and output shapes.
* Each shape is a symbolic representation of the tensor's indices and dimensions. For example, a tensor $$A$$ can be $$\text{shape}_A=bd$$ where $$b$$ describes the batch index, and $$d$$ describes the feature dimension index.
* In most cases, it is often clear what the inputs and the output are, so we may use the abridged notation $$\langle \text{shape}_A,\text{shape}_B \rangle \to \text{shape}_C$$ to represent the einsum.

### Einsum Examples

* `Dot product`
  * $$\langle a,b \rangle : \langle d,d \rangle \to 1 $$  
  * This operation specifies that we have two inputs of sizes d each with a scalar as output. This is a **vector dot product** which is the sum over each element along the axis that d represents. Note that d occurs in both the inputs, but not in the output. Therefore, d is the dimension is summed over (hence, the term einsum) and is reduced away. We also call d the **summation** or **reduction** axis.
  * If we are to write this operation in terms of vectors, it can be written as $$a^Tb$$ where $$a^T$$ is the transpose of $$a$$. Note that for einsum, we do not need to specify explicit transpose, since the shapes of the input tensors and the output tensor completely specify the necessary operation. 
* `Matrix-vector multiplication`
  * $$\langle A,b : \langle md,d \rangle \to m$$  
  * This operation specifies that we have a matrix $$A$$ and a vector b as inputs and we want an output vector of size $$m$$, with the axis $$d$$ reduced away since it does not appear in the output. That is, this operation is a usual multiplication of a matrix and a vector.
  * There are $$m$$ rows in the matrix, each of which has dimension $$d$$. Each row is dotted with $$b$$, which gives a scalar. This happens $$m$$ times for the $$m$$ rows of $$A$$.
* `Attention`
  * $$\langle K,q \rangle = \langle hmk,hk \rangle \to hm $$  
  * In this case, $$h$$ is the common index that is not reduced away (we have h in both inputs as well as the output). This einsum operation is equivalent to performing $$\langle mk,k \rangle \to m$$ for h times where $$\langle mk,k \rangle \to m$$ is a matrix multiplication.
  * In fact, this is the tensor operation that specifies the interaction between the query tensor $$q$$ and the key tensor $$K$$ in Transformer’s attention block during incremental decoding, with batch size 1.
* `Batched Attention`
  * $$ W= \langle K,Q \rangle : \langle bhmk,bhnk \rangle \to bhmn $$  
  * This is similar to doing $$\langle mk,nk \rangle \to mn$$ for bh times.
  * Here, $$\langle mk,nk \rangle \to mn$$ is a multiplication of two matrices, or more precisely, $$AB^T$$ where $$A,B$$ are of shapes $$mk,nk$$ respectively. Again, we can see that for einsum, that we do not need to worry about transpose or orders of shapes.
  * This operation is precisely the batch **key-query attention**.
* `Linear Projection`
  * $$ K = \langle X,P_K \rangle : \langle bmd,dhk \rangle \to bhmk $$  
  * Here, d is reduced away. This is the linear layer to obtain the key tensor from the input.




## Multi-Head Attention


For a detailed understanding of the GPT architecture, I recommend [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), [The GPT Architecture on a Napkin](https://dugas.ch/artificial_curiosity/GPT_architecture.html), and [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).


We describe the attention in two stages. Given inputs with batch size b and m tokens, we first perform the **context computation** to obtain the key and value tensors that will be needed later for incremental decoding. 



<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/attention-refined.svg"
  class="img-fluid rounded z-depth-1"
  padding="10px"
  caption="Figure 1: Attention via Einsum"
%}
</div>

### Context Computation

* Each attention operation starts from the input $$x$$. For normal transformers setup, the key and query inputs are the same. However, in `Figure 1`, we described the generalized version where the query input $$X_Q$$ and key/value input $$X_K$$ can be different. The distinct shapes of these two inputs also provide clarity for the einsum operations in this figure.
* For each batch index in $$b$$ and length index in $$m$$ or $$n$$, we have a feature of dimension $$d$$. 
  * Again, that we distinguish the key length $$m$$ and query length $$n$$ even though the numeric value can be the same for context encoding.
* Intuition for projection $$Q= \langle x,P_Q \rangle : \langle bnd,dhk \rangle \to bhnk $$
    * for each batch and query length, we project the feature dimension of $$x$$ (index $$d$$) with the parameterized feature mapping $$P_Q$$ (linear layer). The generic input $$x$$ is transformed to be a tensor that will later act as a query.
    * Note that the reduction axis is along dimension $$d$$ (boldface), meaning that each output gather the information over the input feature dimension and the projection input dimension $$d$$.
    * The same logic applies for $$K$$ and $$V$$.
* Intuition for the score computation $$ W = \langle K,Q \rangle : \langle bhmk,bhnk \rangle \to bhmn $$ 
    * The reduction index is $$k$$, the key head dimension. Again, this can be seen as computing $$\langle mk, nk \rangle \to mn$$ for $$bh$$ times. For each key length index $$m$$ and query length index $$n$$, we obtain the *score* which is the sum over all the feature in axis $$k$$ for each head $$h$$. This is precisely the logic behind *multi-head attention*.
* Intuition for $$ O= \langle W,V \rangle : \langle bhmn,bhmv \rangle \to bhnv $$
    * In this case, we reduce over the key length. That is, for each query length index, we aggregate all the scores or attention from all key positions (all context tokens). This is the weighted sum of the value where each key position contributes differently to the value tensor to produce the output.
    * For decoder models, note that the step to produce $$W$$ is a causal mask which zeros out the signals from the key position that follows each query position. Without such causal mask, this would constitute a bi-directional attention.


### Incremental Decoding


* After the context computation is done, for each **incremental** **decoding** step, the attention is computed the same way, except that the incoming input corresponds to length 1. The same notation in `Figure 1` applies with the input $$x = bd$$.
* We also perform concatenation with the previous key $$K'$$ and previous value $$V'$$ respectively before each attention operation.
* In such step, each input token is projected into a query, key, and value tensors. The attention gathers the information along the head feature $$k$$ and the output is the aggregated value with weight average over all key positions $$m$$.



<!--
  <script src="https://giscus.app/client.js"
  data-repo="benathi/blogs"
  data-repo-id="R_kgDOI_5r3w"
  data-category="Ideas"
  data-category-id="DIC_kwDOI_5r384CUfWs"
  data-mapping="pathname"
  data-strict="0"
  data-reactions-enabled="1"
  data-emit-metadata="0"
  data-input-position="top"
  data-theme="preferred_color_scheme"
  data-lang="en"
  crossorigin="anonymous"
  async>
  </script>
-->
