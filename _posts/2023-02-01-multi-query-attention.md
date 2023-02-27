---
layout: distill
title: Analysis of Inference Efficiency of Multi-Query Attention
date:   2023-02-01
description: Memory IO complexity
tags: llm 
categories: 
published: false
---

# Blog: Memory IO Efficiency of Multi-Query Attention 



### Multi-Query Architecture at a Glance

The key difference of multi-query attention is to collapse all the heads of the projection matrices P_K and P_V to only 1 head. All other projection matrices (P_Q and P_O) still have sizes ‘hdk’. P_K and P_V have the size reduction described below. 

* P_K: ‘hdk’ [multi attention] → ‘dk’ [multi query]
* P_V: ‘hdv’ [multi attention] → ‘dv’ [multi query]



**Need a diagram here showing a projection, then the use of key and value tensors during attention.**

Note that given an input x of hidden dimension d, during incremental decoding, x is still projected to many heads during querying (since the query has h heads). Since the query has many heads, the fact that the projection matrix P_K and P_V has 1 head still leads to multiple head-interactions during logits and output computation. This will be evident later in section [Incremental Decoding Cost: Multi Query Architecture for Memory Access Bottleneck Reduction — Latency Benchmark and Literature Review](https://quip.com/ERHxAblxndub#temp:C:cJDc8467b94424e4362a59476323).



### Operation and Memory Access Counting (short version)

At a high level, the number operations and memory access for the computation “A,B → C” are (when A,B are expressed in terms of dimension like in Einstein sum)

* Number of operations: O(product(distinct dimensions in A and B)))
* Number of memory access: O(|A|) + O(|B|) where |A| is the size of the tensor A (product of all dimensions).
* For example, “bhnv, hdv → bnd” requires
    * O(bhndv) = O(bnd^2) number of operations
    * and O(bhnv + hdv) = O(bnd + d^2) memory access
* Long version with explanations: [Details — Operation and Memory Access Counting (can be skipped): Multi Query Architecture for Memory Access Bottleneck Reduction — Latency Benchmark and Literature Review](https://quip.com/ERHxAblxndub#temp:C:cJD1790d72e0231402a912f9c6c6)



### Details — Operation and Memory Access Counting (can be skipped)

* The number of memory access for operation “A,B → C” is O(A) + O(B) because we need to access both input A and B. For instance, 
    * “bhnv, hdv → bnd” requires the number of memory access = O(bhnv) + O(hdv) = O(bnd) + O(d^2) 
* The number of operations for “A,B - > C” is the number of duplicates * the number of base operations.
    * Example 1: “bhnk, bhmk → bhnm“ has bh number of duplicates where the base operation is ”nk,mk→ nm“ since ”bh“ are the dimensions that are shared across all inputs and output. This matrix multiplication ”nk,mk→ nm“ requires nmk operations. Therefore, total number of operations is O(bh * nmk ) = O(bn^2d)
        * Note. for ”nk,mk→ nm“, n and m are the non interacting dimensions and k is the interacting dimension (getting summed over). The number of operations in general = product(set(non-interacting dimensions)) * interacting dimension = nm * k.
    * Example 2: “bhnv, hdv → bnd”. In this case, there’s no ‘duplicate’ dimensions. Since this can be framed as “bn * hv, d * hv → bnd”, we see that bn and d are the non-interacting dimensions and hv are the interacting one. Therefore, the number of operations is O(bnd * hv ) = O(bnd^2)
    * In general, this is equivalent to O(product(set(A, B)))



### Incremental Decoding Cost

**Main Takeaway**
The calculations that incur the highest amount of memory access for normal multi-head attention are the logits and output calculations in [Table A](https://quip-amazon.com/ERHxAblxndub/Multi-Query-Architecture-for-Memory-Access-Bottleneck-Reduction-Literature-Review-and-Latency-Benchmark#temp:C:cJD5f1ee57d0f60445cb0d3203c3) which involves the following tensor operation (for logits)
Multi Head        <q,K>: “bhk, bhmk → bhm”
There are O(bhmk) number of operations but it requires O(bhmk) memory access (1-1). This calculation is memory-bound and is inefficient. Instead, the multi query variant requires only O(bhk + bmk) memory access.
Multi Query      <q,K>: “bhk, bmk → bhm”


**Details**
The following table provides analysis for number of operations and memory access cost (in terms of tight complexity bounds) for both the traditional multi-head attention versus multi-query attention.

* The color red denote the change due to multi-query attention. Other operations are the same across multi-attention and multi-head if the difference is not stated explicitly.
* Note: The number of operations are the same for multi-query and multi-attention



| Tensor Operation                                                                              | Memory Access | Memory Access | Computation complexity |
|-----------------------------------------------------------------------------------------------|------------------------------|------------------------|
|                                                                                               | Multi-Head                   | Multi-Query            |
| Input (x) bd                                                                                  |                              |                        | bhk*d = bd^2           |
| q = <x, P_q>                                                                                  | bd + hdk = bd + d^2          | bd + d^2               | bhk*d = bd^2           |
| K = <x, P_k> + (append previous) <br> [MH] bd,hdk → bhk (+ bmhk) <br> [MQ] bd,dk → bk (+ bmk) | bd + d^2                     | bd + dk | bhv*d = bd^2 |
| V = <x, P_v> +[append previous] <br> [MH] bd,hdv → bhv [ bmhv] <br> [MQ] bd,dk → bv [ bmv]              | bd + d^2                     | bd + dv | bhm*k = bmd |
| logits = <q, K> <br> [MH] bhk,bhmk → bhm <br> [MQ] bhk,bmk → bhm                                        | bhk + bhmk = bd + bmd        | bhk + bmk = bd + bmk | bhm |
| weights: softmax                                                                              | bhm                          | bhm                    | bhv*m = bmd            |
| out(O) = <weights, V> <br> [MH] bhm,bhmv → bhv <br> [MQ] bhm,bmv → bhv                                  | bhm + bhmv = bhm + bmd       | bhm + bmv | bdhv = bd^2 |
| y=<O, P_O> bhv,hdv → bd                                                                       | bd + d^2                     | bd + d^2               | bmd + bd^2 ~ bd^2     |
| Total                                                                                         | bd + bmd + d^2               | bd + bmk + d^2         |                        |
| Ratio of Memory Access Per Computation Operatios                                              | 1/d + m/d + 1/b              | 1/d + m/(dh) + 1/b |                        |


|Tensor Operation	|Memory Access Complexity	|Computation complexity	|
|---	|---	|---	|
|Input (x)
     bd	|-	|-	|bhk*d = bd^2	|
|q = <x, P_q>	|bd,hdk → bhk	|bd + hdk = bd + d^2	|bhk*d = bd^2	|
|K = <x, P_k> + (append previous)	|Multi-Head
 bd,**h**dk → b**h**k (+ bm**h**k)	|bd + d^2	|bhv*d = bd^2	|
|Multi-Query
bd,**1**dk → b**1**k (+ bm**1**k)	|bd + dk	|
|	|	|	|	|





**Observations**

* for b ~ 1 or m ~ d, the number of memory access is high compared to the number of operations
* For multi-query, the offending term m/d is reduced by h to m/(dh)



### Batch Computation Cost for Multi-Head Attention (can be skipped)

Batch computation in this case refers to when we compute attentions corresponding to ‘n’ tokens. The analysis below shows that the number of memory access per operation is << 1 in which makes it quite efficient.

The table below shows the analysis per each operation. The memory access complexity are the same for both multi-head and multi-query. In practice, the multi-query setting is slightly faster due to lower constants. (In MQ, some d^2 terms are reduced to dk for example, but the total complexity is still bounded by d^2)

|Tensor Operation	|Memory access complexity	|Computation complexity	|
|---	|---	|---	|
|Multi-Head	|Multi-Query	|
|Input X
    bnd for query 
    bmd for key	|	|	|	|
|Q = <X, P_q>:
     bnd,hdk → bhnk	|bnd + hdk = bnd + d^2	|bnd + d^2	|bn*hk*d = bnd^2	|
|K = <X, P_k>:
     [MH] bmd,hdk → bhmk
     [MQ] bmd,dk → bmk	|bmd + hdk = bmd + d^2	|bmd + dk	|bm*hk*d = bmd^2	|
|V = <X, P_v>:
     [MH] bmd,hdv → bhmv
     [MQ] bmd,dv → bmv	|bmd + d^2	|bmd + dv	|bm*hv*d = bmd^2	|
|logits = <Q, K>:
     [MH] bhnk,bhmk → bhnm
     [MQ] bhnk,bmk → bhnm	|bhnk + bhmk = bnd	|bhnk + bmk = bnd	|bh*n*m*k = bn^2d	|
|weights: softmax	|bhnm = bhn^2	|bhnm = bhn^2	|bhnm = bhn^2	|
|O=<weights, V>:
     [MH] bhnm,bhmv → bhnv
     [MQ] bhnm,bmv → bhnv	|bhnm + bhmv = bn^2h + bnd	|bhnm + bmv = bn^2h + bnv	|bh*n*v*m = bn^2d	|
|y = <O, P_O>:
     bhnv,hdv → bnd	|bhnv + hdv = bnd + d^2	|bhnv + hdv = bnd + d^2	|b*n*hvd = bnd^2	|
|Total	|bnd + bhn^2 + d^2	|bnd + bhn^2 + d^2	|bnd^2 + bn^2d ~ bnd^2	|
|Ratio of Memory Access Per Operations	|1/d + 1/k + 1/(bn) << 1	|1/d + 1/k + 1/(bn) << 1	|-	|

* The memory complexity for MQ is roughly the same as MH since (1) dk < d^2 and (2) bmk < bnd??
* Observation. The batched version is really the ideal case where all n queries and m keys interact all at once. We will see in the inference latency benchmark that, even for 13B model, the amortized latency cost per token is ~ 0.2 ms instead of 30+ ms / step for incremental decoding. We can see clearly that in terms of computation capacity, current GPUs are already quite fast when computation can be done in batch to reduce memory I/O. The main bottleneck for incremental decoding is memory access.





### Notes


The dimensionality reduction of P_K and P_V leads to lower number of parameters (for example, 12.8B multi-attention model becomes 10.5B multi-query model, fixing all other configurations constant). In order to scale up the multi-query attention model to be of similar size, we can increase other configurations such as h or d/h.
