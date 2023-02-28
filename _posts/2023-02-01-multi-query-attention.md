---
layout: distill
title: Analysis of Inference Efficiency of Multi-Query Attention
date:   2023-02-01
description: Memory IO complexity
tags: llm 
categories: transformers
published: true
giscus_comments: true
enable_math: true

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




## Multi-Query Attention at a Glance

The key difference of multi-query attention is to collapse all the heads of the projection matrices P_K and P_V to only 1 head. All other projection matrices (P_Q and P_O) still have sizes ‘hdk’. P_K and P_V have the size reduction described below. 

* P_K: ‘hdk’ [multi attention] → ‘dk’ [multi query]
* P_V: ‘hdv’ [multi attention] → ‘dv’ [multi query]



## Background

**Need a diagram here showing a projection, then the use of key and value tensors during attention.**

Note that given an input x of hidden dimension d, during incremental decoding, x is still projected to many heads during querying (since the query has h heads). Since the query has many heads, the fact that the projection matrix P_K and P_V has 1 head still leads to multiple head-interactions during logits and output computation. This will be evident later in section ...



### Operation and Memory Access Counting (short version)

At a high level, the number operations and memory access for the computation “A,B → C” are (when A,B are expressed in terms of dimension like in Einstein sum)

* Number of operations: $$\small O( \text{product}(\text{distinct dimensions in A and B})))$$
* Number of memory access: $$\small \mathcal{O}(\vert A \vert) + O( \vert B \vert )$$ where $$\small \vert A \vert$$ is the size of the tensor A (product of all dimensions).
* For example, $$\small bhnv, hdv \to bhnd$$ requires
    * $$\small \mathcal{O}(bhndv) = \mathcal{O}(bnd^2)$$ number of operations
    * and $$\small \mathcal{O}(bhnv + hdv + bhnd)$$ memory access for both of the inputs as well as the output.

### Details — Operation and Memory Access Counting (can be skipped)

* The number of memory access for operation “A,B → C” is O(A) + O(B) because we need to access both input A and B. For instance, 
    * “bhnv, hdv → bnd” requires the number of memory access = O(bhnv) + O(hdv) = O(bnd) + O(d^2) 
* The number of operations for “A,B - > C” is the number of duplicates * the number of base operations.
    * Example 1: “bhnk, bhmk → bhnm“ has bh number of duplicates where the base operation is ”nk,mk→ nm“ since ”bh“ are the dimensions that are shared across all inputs and output. This matrix multiplication ”nk,mk→ nm“ requires nmk operations. Therefore, total number of operations is O(bh * nmk ) = O(bn^2d)
        * Note. for ”nk,mk→ nm“, n and m are the non interacting dimensions and k is the interacting dimension (getting summed over). The number of operations in general = product(set(non-interacting dimensions)) * interacting dimension = nm * k.
    * Example 2: “bhnv, hdv → bnd”. In this case, there’s no ‘duplicate’ dimensions. Since this can be framed as “bn * hv, d * hv → bnd”, we see that bn and d are the non-interacting dimensions and hv are the interacting one. Therefore, the number of operations is O(bnd * hv ) = O(bnd^2)
    * In general, this is equivalent to O(product(set(A, B)))




## Memory IO Cost



<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/attention-multiquery.svg"
  class="img-fluid rounded z-depth-1"
  padding="10px"
    caption="Figure 2: Attention Parallel"
%}
</div>



### Incremental Decoding

**Main Takeaway**
The calculations that incur the highest amount of memory access for normal multi-head attention are the logits and output calculations in [Table A] which involves the following tensor operation (for logits)
Multi Head        <q,K>: “bhk, bhmk → bhm”
There are O(bhmk) number of operations but it requires O(bhmk) memory access (1-1). This calculation is memory-bound and is inefficient. Instead, the multi query variant requires only O(bhk + bmk) memory access.
Multi Query      <q,K>: “bhk, bmk → bhm”


**Details**
The following table provides analysis for number of operations and memory access cost (in terms of tight complexity bounds) for both the traditional multi-head attention versus multi-query attention.

* The color red denote the change due to multi-query attention. Other operations are the same across multi-attention and multi-head if the difference is not stated explicitly.
* Note: The number of operations are the same for multi-query and multi-attention

<br>

**Table 1**: Memory Access and Computation Complexities for Incremental Decoding with Multi-Head and Multi-Query Attention.

$$
\scriptsize{
\begin{array}{l|l|c|c}
\textbf{Operation} & \textbf{Einsum} & \textbf{Memory Access} & \textbf{Computation} \\\hline
\text{Input (x) : bd} & & \\
\rule{0pt}{2em}
q = \langle x, P_q \rangle & bd,hdk \rightarrow bhk & bd + hdk = bd + d^2 & bdhk = bd^2 \\
\rule{0pt}{1.5em}
 K = \langle x, P_k \rangle \ (+ K_{prev}) & [MH] \ bd,{\color{red}{h}} dk \rightarrow b{\color{red}{h}}k \ (+ bm{\color{red}{h}}k) & bd + {\color{red}{d^2}} & bdhk = bd^2 \\
 & [MQ] \ bd,dk \rightarrow bk \ (+ bmk) & bd + {\color{red}{dk}} & \\
\rule{0pt}{2em}
V = \langle x, P_v \rangle \ (+ V_{prev}) & [MH] \ bd,{\color{red}{h}}dv \rightarrow bhv \ (+ bm{\color{red}{h}}v) & bd + {\color{red}{d^2}} & bdhv = bd^2 \\
 & [MQ] \ bd,dv \rightarrow bv \ (+ bmv) &  bd + {\color{red}{dv}} & \\
\rule{0pt}{2em}
\text{logits} = \langle q, K \rangle & [MH] \ bhk,b{\color{red}{h}}mk \rightarrow bhm & bhk + bhmk = bd + bm{\color{red}{d}} & bhmk = bmd \\
 & [MQ] \ bhk,bmk \rightarrow bhm &  bhk + bmk = bmd + bm{\color{red}{k}} & \\
\rule{0pt}{2em}
\text{weights: softmax} & & bhm & bhm \\
\rule{0pt}{2em}
\text{out(O)} = \langle \text{weights}, V \rangle & [MH] \ bhm,b{\color{red}{h}}mv \rightarrow bhv & bhm + bhmv = bhm + bm{\color{red}{d}} & bhmv = d \\
 & [MQ] \ bhm,bmv \rightarrow bhv & bhm + bm{\color{red}{v}} & \\
\rule{0pt}{2em}
y=\langle O, P_O \rangle & bhv,hdv \rightarrow bd & bd + d^2 & bdhv = bd^2  \\
\rule{0pt}{2em}
\text{Total}\text{: Multi Head} &  & bd + bmd + d^2 & bhm + bm{\color{red}{d}} + bd^2 \approx bd^2 \\
\text{Total}\text{: Multi Query} & &  bd + bm{\color{red}{k}} + d^2 & \\
\hline
\rule{0pt}{1em} 
r: \text{Multi Head} & & 1/d + m/{\color{red}{d}} + 1/b & \\
r: \text{Multi Query} &  & 1/d + m/({\color{red}{dh}}) + 1/b & \\
\end{array}
}
$$

Note: $$r$$ is the ratio of memory access complexity versus computation complexity. A ratio close to 1 would indicate that there are 1-to-1 memory access per computation, which would be very inefficient. An unfused softmax or dropout is such examples of IO inefficienct operations.




**Observations**

* for b ~ 1 or m ~ d, the number of memory access is high compared to the number of operations
* For multi-query, the offending term m/d is reduced by h to m/(dh)



### Batch Computation Cost for Multi-Head Attention (can be skipped)

Batch computation in this case refers to when we compute attentions corresponding to ‘n’ tokens. The analysis below shows that the number of memory access per operation is << 1 in which makes it quite efficient.

The table below shows the analysis per each operation. The memory access complexity are the same for both multi-head and multi-query. In practice, the multi-query setting is slightly faster due to lower constants. (In MQ, some d^2 terms are reduced to dk for example, but the total complexity is still bounded by d^2)


<br>
**Table 2**: Memory Access and Computation Complexities for Batch Computation with Multi-Head and Multi-Query Attention.

$$
\scriptsize{
\begin{array}{l|l|c|c}
\textbf{Operation} & \textbf{Einsum} & \textbf{Memory Access} & \textbf{Computation} \\\hline
\text{Input M, N : bmd, bnd} & & \\
\rule{0pt}{2em}
q = \langle N, P_q \rangle & bnd,dhk \rightarrow bhnk & bnd + dhk = bnd + d^2 & bndhk = bnd^2 \\
\rule{0pt}{1.5em}
 K = \langle M, P_k \rangle  & [MH] \ bmd,d{\color{red}{h}}k \rightarrow b{\color{red}{h}}mk  & bmd + {\color{red}{d^2}} & bmdhk = bmd^2 \\
 & [MQ] \ bmd,dk \rightarrow bmk  & bmd + {\color{red}{dk}} & \\
\rule{0pt}{2em}
V = \langle M, P_v \rangle  & [MH] \ bmd,d{\color{red}{h}}v \rightarrow b{\color{red}{h}}mv  & bmd + {\color{red}{d^2}} & bmdhv = bd^2 \\
 & [MQ] \ bmd,dv \rightarrow bmv &  bmd + {\color{red}{dv}} & \\
\rule{0pt}{2em}
\text{logits} = \langle Q, K \rangle & [MH] \ bhnk,b{\color{red}{h}}mk \rightarrow bhnm & bhnk + bhmk = bnd + bm{\color{red}{d}} & bhmnk = bmnd = bn^2d \\
 & [MQ] \ bhnk,bmk \rightarrow bhnm &  bhnk + bmk = bnd + bm{\color{red}{k}} & \\
\rule{0pt}{2em}
\text{weights: softmax} & & bhnm & bhnm \\
\rule{0pt}{2em}
\text{out(O)} = \langle \text{weights}, V \rangle & [MH] \ bhnm,b{\color{red}{h}}mv \rightarrow bhnv & bhnm + bhmv = bhnm + bm{\color{red}{d}} & bhnmv = bmnd = bn^2d \\
 & [MQ] \ bhnm,bmv \rightarrow bhnv & bhnm + bm{\color{red}{v}} & \\
\rule{0pt}{2em}
y=\langle O, P_O \rangle & bhnv,hvd \rightarrow bnd & bnd + d^2 & bndhv = bnd^2  \\
\rule{0pt}{2em}
\text{Total}\text{: Multi Head} &  & bnd + bhn^2 + d^2 & bnd^2 + bn^2d \approx bnd^2 \\
\text{Total}\text{: Multi Query} & & bnd + bhn^2 + d^2 & \\
\hline
\rule{0pt}{1em} 
r: \text{Multi Head} & & 1/d + 1/k + 1/(bn) << 1 & \\
r: \text{Multi Query} &  & 1/d + 1/k + 1/(bn) << 1 & \\
\end{array}
}
$$

<br>
* Both MQ and MH have the same memory access complexity in the batch case, leading to the same efficiency.
* We estimate the complexity with $$n=m$$ for the usual context encoding case (where the query and key inputs are the same)
* The memory complexity for MQ is roughly the same as MH since (1) dk < d^2 and (2) bmk < bnd??
* Observation. The batched version is really the ideal case where all n queries and m keys interact all at once. We will see in the inference latency benchmark that, even for 13B model, the amortized latency cost per token is ~ 0.2 ms instead of 30+ ms / step for incremental decoding. We can see clearly that in terms of computation capacity, current GPUs are already quite fast when computation can be done in batch to reduce memory I/O. The main bottleneck for incremental decoding is memory access.





### Notes


The dimensionality reduction of P_K and P_V leads to lower number of parameters (for example, 12.8B multi-attention model becomes 10.5B multi-query model, fixing all other configurations constant). In order to scale up the multi-query attention model to be of similar size, we can increase other configurations such as h or d/h.

