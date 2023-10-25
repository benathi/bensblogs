---
layout: distill
title: "The Illustrated Long Convolution Models"
date:   2023-10-15
description: 
tags: operators convolution s4 h3 gss hyena 
categories: 
published: true
social: true
giscus_comments: true


authors:
  - name: Ben Athiwaratkun 
    url: https://benathi.github.io

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Background
  - subsections:
      - name: A. Convolution and Fast Fourier Transform
      - name: B. Orthonormal Basis in Function Space
      - name: C. State Space Models
      - name: D. Einstein Summation for General Tensor Operations 
      - name: E. Attention and Linear Attention
  - name: Long Convolution Models
  - subsections:
      - name: Hippo Matrix for History Representation via State Space Models
      - name: "S4: Structured State Space"
      - name: "H3: Hungry Hungry Hippos"
      - name: "Hyena Hierarchy"
      - name: "A Unified Perspective on Long Convolution Models"

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





There are recent exciting developments of AI models for sub-quadratic long context modeling involving the use of convolutions, ranging from Structured State Space Models (S4) {% cite s4 %} {% cite s4d %}, Gated State Spaces (GSS) {% cite gss %}, H3 {% cite h3 %}, and most recently, Hyena model {% cite hyena %}. In this blog, we will seek to develop intuition behind these models with some illustrations. 

<!-- We will aim for this blog post to be as accessible as possible, which means we will trade mathematical rigor for intuitive explanation and may omit some details related to the convergence conditions etc. (after all, if you want mathematical regor, the corresponding original papers also do a great job).
-->




# Background 

The goal for this post is to be as self-contained as possible, meaning that we will cover a lot of necessary background to deeply understand related convolution models and how they compare and connect with transformer's attention. For instance, we cover the core idea and derivation of convolution theorem and fast fourier transform (FFT), which is at the heart of being able to do long range model with only sub-quadratic computation complexity. We provide background section on orthonomal basis in function space, which will be important to understand the construction of Hippo matrix for long range modeling with state space models. We will also cover einstein summation, as it provides a convenient way to tie different operations together under general (linear) tensor operations and is used throughout the blog post. 

If you are already familiar with certain topics, feel free to skip. Below outlines the recommended background for each of the convolution models so that you can decide what to focus on.


* Hippo Matrix: requires background [A](#a-convolution-and-fast-fourier-transform), [B](#b-orthonormal-basis-in-function-space), [C](#c-state-space-models)
* S4: requires background [A](#a-convolution-and-fast-fourier-transform) and [C](#c-state-space-models) and understanding of [Hippo matrix](#hippo-matrix-for-history-representation-via-state-space-models)
* H3: requires background [A](#a-convolution-and-fast-fourier-transform), C and understanding of [Hippo matrix](#hippo-matrix-for-history-representation-via-state-space-models) and [S4](#s4-structured-state-space)
* Hyena: requires background [A](#a-convolution-and-fast-fourier-transform). All background topics recommended.
* Unified perspective: requires background [A](#a-convolution-and-fast-fourier-transform), [D] [E]. All background topics recommended.

<!--

## A: Convolution and Fast Fourier Transform
## B: Orthonormal Basis in Function Space
## C: State Space Models
## D: Einstein Summation for General Tensor Operations
## E: Attention and Linear Attention

-->

<!--
<details>
  <summary></summary>
  using the collapsible is not really working well with the TOC
  and also the rendering of title is broken
  -->

## A. Convolution and Fast Fourier Transform

- introduce the concept of convolution and show the illustration.
- tie to the probability distribution interpretation: the density of X + Y is the convolution of the density of X and Y.
- Give intuition in terms of getting contribution from both the signal and the convolution kernel.


### What is a Convolution?
<!-- ##  and $$\mathbf{b} \in \mathbb{R}^M$$ -->

Let us consider two N-dimensional vectors $$\mathbf{a}, \mathbf{b} \in \mathbb{R}^N$$. The convolution of the two vectors, denoted as $$\mathbf{a} * \mathbf{b}$$ is defined as:

$$
\begin{align*}
\mathbf{c}_i = (\mathbf{a} * \mathbf{b})_i &= \sum_{n=0}^{N-1} a_n b_{i-n} \\
\end{align*}
$$


<!-- &= \sum_{n=0}^{N-1} a_{i-n} b_n -->
To make it more concrete, let's expand this out with $$N=4$$. 

$$
\begin{align*}
c_0 &= a_0 b_0 \\
c_1 &= a_0 b_1 + a_1 b_0 \\
c_2 &= a_0 b_2 + a_1 b_1 + a_2 b_0 \\
c_3 &= a_0 b_3 + a_1 b_2 + a_2 b_1 + a_3 b_0 \\
c_4 &= a_1 b_3 + a_2 b_2 + a_3 b_1 \\
c_5 &= a_2 b_3 + a_3 b_2 \\
c_6 &= a_3 b_3 \\
\end{align*}
$$

where we note that $$a_j$$ or $$b_j$$ is undefined outside $$j \in \{0, \dots, N-1 \}$$, so we can think of such invalid terms as zero.


We can also write $$\mathbf{c}$$ as a matrix multiplication:

$$
\mathbf{c} = 
\begin{bmatrix}
c_0 \\
c_1 \\
c_2 \\
c_3 \\
c_4 \\
c_5 \\
c_6 \\
\end{bmatrix}
=
\begin{bmatrix}
a_0 & 0 & 0 & 0 \\
a_1 & a_0 & 0 & 0 \\
a_2 & a_1 & a_0 & 0 \\
a_3 & a_2 & a_1 & a_0 \\
0 & a_3 & a_2 & a_1 \\
0 & 0 & a_3 & a_2 \\
0 & 0 & 0 & a_3 \\
\end{bmatrix}
\begin{bmatrix}
b_0 \\
b_1 \\
b_2 \\
b_3 \\
\end{bmatrix}
\tag{Convolution}

$$

The convolution of two vectors can be then written in matrix multiplication term as

$$
\mathbf{c} = S_a \mathbf{b}
$$

where $$S_a$$ is the matrix representation of vector $$\mathbf{a}$$. 
We observe that $$S_a$$ is Toeplitz[^toeplitz], meaning that it each diagonal values from left to right are constant. 
The convolution operator is also commutative, meaning that $$\mathbf{a} * \mathbf{b} = \mathbf{b} * \mathbf{a}$$, or in matrix term, $$\mathbf{c} = S_a \mathbf{b} = S_b \mathbf{a}$$ where $$S_b$$ is the convolution matrix representation of vector $$\mathbf{b}$$.


<!--
In many cases, we use convolutions as a way to process signals that depends on time or positions, so it is common that we use the **causal** convolution[^causal-convolution] where the resulting signal $$c_i$$ can never depend on future time steps $$ n > i$$. That is, the causal convolution becomes:

$$
\begin{align*}
\mathbf{c}_i = (\mathbf{a} * \mathbf{b})_i &= \sum_{n=0}^{i} a_n b_{i-n} \\
\end{align*}
$$

For $$N=4$$, 

$$
\begin{align*}
c_0 &= a_0 b_0 \\
c_1 &= a_0 b_1 + a_1 b_0 \\
c_2 &= a_0 b_2 + a_1 b_1 + a_2 b_0 \\
c_3 &= a_0 b_3 + a_1 b_2 + a_2 b_1 + a_3 b_0 \\
\end{align*}
$$

In the matrix form, this corresponds to a truncated or causal version of the Toeplitz matrix representation:

$$
\mathbf{c} =
\begin{bmatrix}
c_0 \\
c_1 \\
c_2 \\
c_3 \\
\end{bmatrix}
=
\begin{bmatrix}
a_0 & 0 & 0 & 0 \\
a_1 & a_0 & 0 & 0 \\
a_2 & a_1 & a_0 & 0 \\
a_3 & a_2 & a_1 & a_0 \\
\end{bmatrix}
\begin{bmatrix}
b_0 \\
b_1 \\
b_2 \\
b_3 \\
\end{bmatrix}
\tag{Causal Convolution}
$$


BenA: if we assume that outside the defined indices, the values of a_n and b_n are zero, then any convolution is already a causal convolution since the signal from a_n where n > i that contributes to c_i will multiply with b_{i-n} which is zero.

-->



[^toeplitz]: See [Wikipedia - Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix) for more details on Toeplitz matrix.

[^causal-convolution]: See [Causal Convolution](https://paperswithcode.com/method/causal-convolution#:~:text=Causal%20convolutions%20are%20a%20type,2%20%2C%20%E2%80%A6%20%2C%20x%20T%20.)



#### High-Level Intuition

We can see convolution as a way to combine two signals. The first signal is the convolution kernel $$\mathbf{a}$$ and the second signal is the input signal $$\mathbf{b}$$. The combination is such that for the output $$\mathbb{c}_i$$ gather signals from indicies that add up to $$i$$ exactly. For instance, $$c_2$$ is the sum of $$a_0 b_2$$ and $$a_1 b_1$$ and $$a_2 b_0$$, where the indicies add up to $$2$$ exactly. 


Observe that due to how we index the inputs where $$a_{n} = 0 $$ for $$ n < 0$$ results in the convolution being **causal**, which means that the signal $$c_i$$ can only depend on input at time $$i$$ or before. This is because if there is a term $$b_{i+m}$$ for $$m>0$$ that contributes to $$c_i$$, the corresponding term from $$\mathbb{a}$$ is $$a_{-m}$$ which is zero. However, if we do not index the inputs this way, then the convolution may not be causal and the output signal can depend on future inputs unless explicitly handled.

In many cases, we are interested in mapping an input signal $$b(t)$$ to and output $$c(t)$$ on the same time domain $$t = 0, \dots, T-1$$, in which case we can use the truncated version where the corresponding Toeplitz matrix is square and is lower diagonal.


$$
\mathbf{c} =
\begin{bmatrix}
c_0 \\
c_1 \\
c_2 \\
c_3 \\
\end{bmatrix}
=
\begin{bmatrix}
a_0 & 0 & 0 & 0 \\
a_1 & a_0 & 0 & 0 \\
a_2 & a_1 & a_0 & 0 \\
a_3 & a_2 & a_1 & a_0 \\
\end{bmatrix}
\begin{bmatrix}
b_0 \\
b_1 \\
b_2 \\
b_3 \\
\end{bmatrix}
$$


#### Continuous Case of Convolution

The convolution of two functions $$f$$ and $$g$$ is defined as:

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
$$

Let's try to develop an understanding to see what this really means. To do this, we start with an example from probability where we have two random variables $$X$$ and $$Y$$ with probability density functions $$f$$ and $$g$$ respectively. The convolution of $$f$$ and $$g$$ is the probability density function of the sum of $$X$$ and $$Y$$, that is, $$f * g$$ is the probability density function of $$X + Y$$.



To do this, we start with a simple example of a convolution of two functions, one of which is a Dirac delta function. The Dirac delta function is defined as:

$$
\delta(t) =
\begin{cases}
\infty & t = 0 \\
0 & t \neq 0
\end{cases}
$$

and

$$
\int_{-\infty}^{\infty} \delta(t) dt = 1
$$

The Dirac delta function is a special function that is zero everywhere except at $$t=0$$, where it is infinite. However, the integral of the function is 1. This is a special function that is useful in many areas of mathematics, including probability theory and signal processing. In fact, the Dirac delta function can be thought of as a probability distribution that is concentrated at $$t=0$$.

Now, let's consider the convolution of the Dirac delta function with another function $$f$$:

$$
\begin{align*}
(f * \delta)(t) &= \int_{-\infty}^{\infty} f(\tau) \delta(t - \tau) d\tau \\
&= f(t)
\end{align*}
$$

The convolution of $$f$$ with the Dirac delta function is simply $$f$$ itself! This is because the Dirac delta function is zero everywhere except at $$t=0$$, where it is infinite. Therefore, the only contribution to the integral comes from $$f(0)$$, which is simply $$f$$ itself.

Now, let's consider the convolution of $$f$$ with a shifted Dirac delta function:

$$
\begin{align*}
(f * \delta(t - \tau))(t) &= \int_{-\infty}^{\infty} f(\tau') \delta(t - \tau - \tau') d\tau' \\
&= f(t - \tau)
\end{align*}
$$


The convolution of $$f$$ with a shifted Dirac delta function is simply $$f$$ shifted by $$\tau$$! This is because the shifted Dirac delta function is zero everywhere except at $$t=\tau$$, where it is infinite. Therefore, the only contribution to the integral comes from $$f(\tau)$$, which is simply $$f$$ shifted by $$\tau$$.



#### Fourier Transform
- Illustrate the Fourier transform
- Show in both continuous (uncountable basis) and also the countable Fourier basis
- Show in the discrete case as well

As illustrated in [Fourier Basis](#example-2-fourier-basis-for-periodic-functions), any periodic function can be written as a linear combination of sine and cosine, or more compactly, as a linear combination of complex exponentials. A general function on the real line can be expressed in terms of Fourier Transform, which is a continuous version of the Fourier series. The Fourier transform of a function $$f$$ is defined as:

$$
\begin{align*}
\mathcal{F}[f](\omega) &= \int_{-\infty}^{\infty} f(t) e^{-i \omega t} dt \\
&= \int_{-\infty}^{\infty} f(t) \cos(\omega t) dt - i \int_{-\infty}^{\infty} f(t) \sin(\omega t) dt
\end{align*}
$$

where $$\omega$$ is the frequency. Let's look at an example of the Fourier transform of a box function:

$$
\begin{align*}
f(t) &=
\begin{cases}
1 & |t| < L \\
0 & \text{otherwise}
\end{cases}
\end{align*}
$$

It can be shown that the Fourier transform of $$f$$ is:

$$
\begin{align*}
\mathcal{F}[f](\omega) &= 2L \frac{\sin(\omega L)}{\omega L} = 2L \ \text{sinc} (\omega L)
\end{align*}
$$


That is, for a function not is no longer periodic and is defined on the whole real line, to represent it in the frequency basis, we need a continuum of frequencies. This is in contrast to the Fourier series where we only need a countable number of frequencies. In contrast, for a periodic function $$f(x) = \sin(\omega_0 x)$$, the Fourier transform is a Dirac delta function at $$\omega_0$$. That is, we only need a single frequency to represent such function.


<!--
Let's consider another example where $$f$$ is a truncated sine wave.

$$
\begin{align*}
f(t) &=
\begin{cases}
\sin(\omega_0 t) & 0 \leq t \leq T = \frac{2 \pi}{\omega_0} \\
0 & \text{otherwise}
\end{cases}
\end{align*}
$$

Then,

$$
\begin{align*}
\mathcal{F}[f](\omega) &= \frac{1}{2i} \left[ \delta(\omega - \omega_0) - \delta(\omega + \omega_0) \right]
\end{align*}
$$
-->


#### Discrete Fourier Transform

<!-- explain the Nuquist frequency as well -->

The Discrete Fourier Transform (DFT) is a mathematical operation used to analyze the frequency components of a discrete signal. Given a discrete sequence of values $$x[n]$$ for $$n = 0, 1, 2, \ldots, N-1$$, the DFT computes a set of complex coefficients $$X[k]$$ for $$k = 0, 1, 2, \ldots, N-1$$ that represent the frequency content of the signal. The DFT is defined as:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i\frac{2\pi}{N}kn}
$$

Here:
- $$X[k]$$ is the DFT coefficient at frequency $$k$$.
- $$x[n]$$ is the input signal at time index $$n$$.
- $$N$$ is the total number of samples in the input signal.
- The coefficients are complex and can be interpreted as the coefficients of the sine and cosine components (see [Fourier Basis](#example-2-fourier-basis-for-periodic-functions)).


The original series can be recovered from $$X[k]$$ via the inverse DFT (IDFT):

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{i\frac{2\pi}{N}kn}
$$

Both DFT and iDFT involves multiplying $$N$$ numbers together for $$N$$ entries, thus incurs a computational complexity of $$O(N^2)$$. Later, we will discuss a way to perform these operations efficiently using the Fast Fourier Transform (FFT) algorithm that reduces the complexity to $$O(N \log N)$$.



#### Fast Fourier Transform

<!--
We show that DFT can be done without approximation with $$O(N \log N)$$ complexity.
-->

The Fast Fourier Transform (FFT) is an algorithm that computes the DFT efficiently. The FFT algorithm is based on the divide-and-conquer strategy, and is able to compute the DFT in $$O(N \log N)$$ time, which is much faster than the naive $$O(N^2)$$ algorithm. We sketch a proof below.

Let's start with the DFT definition:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i\frac{2\pi}{N}kn}
$$

We can rewrite this by splitting into the odd and even terms as:

$$
X[k] = \sum_{n=0}^{N/2-1} x[2n] \cdot e^{-i\frac{2\pi}{N}k(2n)} + \sum_{n=0}^{N/2-1} x[2n+1] \cdot e^{-i\frac{2\pi}{N}k(2n+1)}
$$

We can further rewrite this as:

$$
\begin{align*}
X[k]    &= \sum_{n=0}^{N/2-1} x[2n] \cdot e^{-i\frac{2\pi}{N/2}kn} + e^{-i\frac{2\pi}{N}k} \sum_{n=0}^{N/2-1} x[2n+1] \cdot e^{-i\frac{2\pi}{N/2}kn} \\
&= E_k + e^{-i\frac{2\pi}{N}k} O_k
\end{align*}
$$

We can see that the first term is the DFT of the even terms of $$x$$, and the second term is the DFT of the odd terms of $$x$$, multiplied by a complex exponential. The key part is that we also obtain $$X[k + \frac{N}{2}]$$ for free as:

$$
X[k + \frac{N}{2}] = E_k - e^{-i\frac{2\pi}{N}k} O_k
$$

Therefore, the the complexity of DFT to obtain $$X[k]$$ consists of the complexity of two DFT of $$N/2$$ elements plus $$O(1)$$ operations, amortized by two, since for each $$X[k]$$, we also get $$X[k+ \frac{N}{2}]$$. This gives us the following recurrence relation:

$$
T(N) = \frac{1}{2} \left( 2 T(N/2) + O(1) \right)
$$

which yields $$T(N) = O(\log N)$$. For all $$k$$, this results in an $$O(N \log N)$$ algorithm for DFT.



#### The Convolution Theorem

<!--
A sketch of proof to show the idea behind convolution theorem. Make things very clear.

- related to sinc function and how it converges to Dirac delta
- causal convolution
- in discrete case, show that convolution can be written as a matrix multiplication with a Toeplitz matrix
- need not be square. but show an example when it's square for causal convolution
-->

In this section, we will show that a convolution of two vectors can be seen as a multiplication of their Fourier transforms. This is known as the convolution theorem. We will first show this in the continuous case, and then show how it can be extended to the discrete case.

Let's start with the continuous case. The convolution of two functions $$f$$ and $$g$$ is defined as:

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
$$

or in the discrete case

$$
(f * g)(t) = \sum_{\tau=-\infty}^{\infty} f(\tau) g(t - \tau)
$$

<!-- Let $$F$$ and $$G$$ be the corresponding Fourier Transform. -->

We will sketch the proof of the convolution theorem in the continuous case

$$
\begin{align*}
\mathcal{F}[f * g](\omega) &= \int_{-\infty}^{\infty} (f * g)(t) e^{-i \omega t} dt \\
&= \int_{-\infty}^{\infty} \left( \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau \right) e^{-i \omega t} dt \\
&= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau) g(t - \tau) e^{-i \omega t} d\tau dt \\
&= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(\tau) g(t - \tau) e^{-i \omega t} dt d\tau \\
&= \int_{-\infty}^{\infty} f(\tau) \left( \int_{-\infty}^{\infty} g(t - \tau) e^{-i \omega t} dt \right) d\tau \\
&= \int_{-\infty}^{\infty} f(\tau) \left( \int_{-\infty}^{\infty} g(t) e^{-i \omega (t + \tau)} dt \right) d\tau \\
&= \int_{-\infty}^{\infty} f(\tau) \left( \int_{-\infty}^{\infty} g(t) e^{-i \omega t} e^{-i \omega \tau} dt \right) d\tau \\
&= \int_{-\infty}^{\infty} f(\tau) \left( \int_{-\infty}^{\infty} g(t) e^{-i \omega t} dt \right) e^{-i \omega \tau} d\tau \\
&= \int_{-\infty}^{\infty} f(\tau) \mathcal{F}[g](\omega) e^{-i \omega \tau} d\tau \\
&= \mathcal{F}[f](\omega) \cdot \mathcal{F}[g](\omega)
\end{align*}
$$

Or in other words,

$$
(f*g)(w) = \mathcal{F}^{-1}[\mathcal{F}[f](\omega) \cdot \mathcal{F}[g](\omega)](w)
$$

which also holds in the discrete convolution case as well.


#### Long Convolution in $$O(N \log N)$$

 The implication of the convolution theorem is that if we want to perform a convolution of long signals $$f(t)$$ and $$g(t)$$, which naively would incur $$O(N^2)$$ computational complexity, we can reduce it to $$O(N \log N)$$ without any approximation. This can be done by

- Computing the Fourier transform of $$f$$ and $$g$$, which incures $$O(N \log N)$$ via Fast Fourier Transform. We obtain the frequency components of length $$N$$.
- Then multiply the Fourier transforms, incurring $$O(N)$$, and finally compute the inverse Fourier transform of the product, which incurs another $$O(N \log N)$$.
- In total, the convolution can be done in $$O(N \log N)$$ via Fast Fourier Transform and the Convolution Theorem, instead of the usual $$O(N^2)$$. This is quite neat!!
- **This sub-quadratic behavior allows fast long range modeling, and is the foundation of convolution models such as S4, S4d, GSS, H3, and Hyena.**

 <!-- </details> A. Convolution and Fast Fourier Transform  -->



## B. Orthonormal Basis in Function Space


We turn our attention to a specialized function space known as a Hilbert space[^hilbert-space] $$\mathcal{H}$$, equipped with a countable and dense orthonormal basis[^schauder-basis] $$\{ g_n \}_{n=0}^\infty$$. In essence, a Hilbert space provides a notion of projections and distance via its inner product $$\langle f, h \rangle$$ and the induced norm $$\| f \| = \sqrt{\langle f, f \rangle}$$. Further, a Hilbert space has a completeness property, meaning that any sequence of elements in $$\mathcal{H}$$ that draws progressively closer together actually converge within the space. As example of such a Hilbert space is the space of square integrable functions on an interval $$L^2[a,b]$$, with a corresponding inner product 

$$\langle f, g \rangle = \int_a^b f(x)^* g(x) \cdot w(x) dx$$

where $$f(x)^*$$ denotes the complex conjugate of $$f$$. Optionally, there can be a function $$w(x)$$ that corresponds to the weight of different points in the interval which reflects the measure that the inner product is defined on.[^measure-theoretic-innerproduct] For instance, we may want to weight points far away with smaller weights -- this will give rise to a different inner product compared to the case where all points are weighted equally.

[^measure-theoretic-innerproduct]: The measure theoretic way is to define the inner product as $$\langle f, g \rangle = \int f(x)^* g(x) d\mu$$ where $$\mu$$ is the measure. In the case of $$L^2[a,b]$$, the measure is $$d\mu = w(x) dx$$ where $$w(x)$$ is the weight function and is zero outside the interval $$[a,b]$$.



The power of the dense orthonormal basis $$\mathcal{G}$$ is that it allows us to represent any function in the space using a linear combination of the basis elements. That is, for any function $$u(t) \in \mathcal{H}$$, there exists coefficients $$\{ c_n \}_{n=1}^\infty$$ such that:

$$ u(t) = \sum_{n=1}^{\infty} c_n g_n(t) $$

That is, in the orthonormal basis $$\mathcal{G}$$, a function $$u$$'s representation is simply  a (infinite) vector $$\{c_n\}_{n=1}^\infty$$. This is incredibly powerful since we reduce a possibly very complicated function to a sequence of numbers.

We can also think of the partial sum $$ u_m(t) = \sum_{n=1}^{m} c_n g_n(t) $$ as an approximation of the function $$u(t)$$, where the approximation gets better as $$m \to \infty$$ (where the convergence is in the norm[^convergence-in-norm]). Therefore, a finite vector $$ (c_1, c_2, .., c_m)$$ can also be used to approximately represent an entire function!! This is a profound concept used in many areas of mathematics, including functional analysis, harmonic analysis, and signal processing.


While this representation ensures approximation, it does not directly offer a method to find $$c_n$$ for a given $$u(t)$$. The key lies in the **orthogonality** of the basis. The set $$\{ g_n \}$$ is orthonormal, implying $$\langle g_m, g_n \rangle = \delta_{m,n}$$, where $$\delta_{m,n}$$ is the Kronecker delta, a function that returns 1 when $$m = n$$ and 0 otherwise. This orthogonality simplifies our task of finding coefficients to taking inner products:

$$
\begin{align*}
\langle u(t), g_n(t) \rangle &= \langle \sum_{m=1}^{\infty} c_m g_m(t), g_n(t) \rangle \\
&= \sum_{m=1}^{\infty} c_m \langle g_m(t), g_n(t) \rangle \\
&= \sum_{m=1}^{\infty} c_m \delta_{m,n} \\
&= c_n
\end{align*}
$$

This "filtering" property, intrinsic to orthonormal bases, ensures that we isolate each coefficient efficiently. 


Based on the weight function $$w(x)$$ and the subspace of functions we operate on, the orthonormal basis can be different. For instance, for uniform weight $$w$$ on an interval, the Legendre polynomials form an orthonormal basis. For periodic functions with uniform weight $$w$$ on an interval, the Fourier basis is an orthonormal basis. For an exponentially decaying weight $$w$$, the Laguerre polynomials form an orthonormal basis. We will discuss these three examples below.


#### Example 1: Legendre Polynomials as Orthonormal Basis on Uniform Measure

The space of square-integrable functions over the interval $$[-1, 1]$$, denoted by $$L^2[-1,1]$$, is an example of a Hilbert space. The inner product is defined as $$\langle f,g \rangle = \int f(x) \cdot g(x) dx $$. In this space, the Legendre polynomials form a countable orthonormal basis.

The Legendre Polynomials offer a rich tapestry of function spaces that play a pivotal role across mathematics and physics. Originating from the studies of celestial mechanics by Adrien-Marie Legendre, these polynomials have since been embraced in various applications spanning from quantum mechanics to approximation theory.

**Definition**:
The $$n^{th}$$ Legendre polynomial, denoted $$P_n(x)$$, is given by Rodrigues' formula:

$$ P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left[ (x^2 - 1)^n \right] $$

More concretely,

$$
\begin{align*}
P_0(x) &= 1 \\
P_1(x) &= x \\
P_2(x) &= \frac{1}{2} (3x^2 - 1) \\
P_3(x) &= \frac{1}{2} (5x^3 - 3x) \\
&\ ..
\end{align*}
$$

These polynomials are orthogonal on the interval $$[-1,1]$$ with respect to the weight function $$w(x) = 1$$. In other words:

$$ \int_{-1}^{1} P_m(x) P_n(x) dx = \frac{2}{2n+1} \delta_{m,n} $$

where $$\delta_{m,n}$$ is the Kronecker delta function. 


The example above can be extended to a Hilbert space defined on $$L^2[a,b]$$, where the orthonormal basis functions can be derived as follows.

We introduce a linear change of variables to adapt $$[-1,1]$$ to the interval $$[a,b]$$. By defining a new variable $$y$$ such that $$y = \frac{2(x - a)}{b - a} - 1$$, we transform the interval $$[a,b]$$ to $$[-1,1]$$. The Legendre polynomials on the interval $$[a,b]$$ are then expressed as $$P_n\left( \frac{2(x - a)}{b - a} - 1 \right)$$. Define $$ \tilde{P}_n(x) $$ as a normalized Legendre polynomial, together with $$dy = \frac{2}{b-a} dx$$ and $$ \int P_m P_n dx = \delta_{m,n} $$, we have:

$$ \tilde{P}_n(x) = \sqrt{\frac{2n+1}{b-a}} P_n\left( \frac{2(x - a)}{b - a} - 1 \right) $$



#### Example 2: Fourier Basis for Periodic Functions

On $$L^2[-L,L]$$ with the inner product $$ \langle f,g \rangle = \frac{1}{2L} \int_{-L}^{L} f(x)^* g(x) dx $$ where $$^*$$ denotes complex conjugate, the Fourier basis is an orthonormal basis when applied to periodic functions.[^periodic] 



The Fourier series representation of a periodic function $$f(x)$$ over the interval $$[-L,L]$$ is given by:

$$
f(x) = \sum_{n=-\infty}^{\infty} c_n e^{i\frac{2\pi n}{2L}x}
$$

where $$c_n$$ are the Fourier coefficients of the basis elements $$f_n = e^{i\frac{2\pi n}{2L}x}$$, computed as:

$$
c_n = \langle e^{i\frac{2\pi n}{2L}x}, f(x)  \rangle =   \frac{1}{2L} \int_{-L}^{L} e^{-i\frac{2\pi n}{2L}x} f(x) dx
$$

We can extract out the coefficient $$c_n$$ from the inner product with $$f_n$$ because for distinct integers $$m$$ and $$n$$, the basis functions $$f_n$$ and $$f_m$$ are orthogonal on the interval $$[-L,L]$$. Their inner product is:

$$
\begin{align*}
\langle f_n , f_m \rangle &= 
\frac{1}{2L} \int_{-L}^{L} e^{- i\frac{2\pi m}{2L}x} e^{i\frac{2\pi n}{2L}x} dx  \\
&= \frac{1}{2L} \int_{-L}^{L} e^{i\frac{2\pi (n-m)}{2L}x} dx \\
&= \frac{1}{2L} \left[ \frac{1}{i\frac{2\pi (n-m)}{2L}} e^{i\frac{2\pi (n-m)}{2L}x} \right]_{-L}^{L} \\
&= \frac{1}{2L} \frac{1}{i\pi (n-m)} \left[  e^{i\pi (n-m)} - e^{-i\pi (n-m)} \right] \\
&= \frac{1}{2L} \frac{1}{i\pi (n-m)} \left[  (-1)^{n-m} - (-1)^{n-m} \right] \\
&= 0.
\end{align*}
$$

For $$n=m$$, we have $$ \frac{1}{2L} \int_{-L}^{L} e^{- i\frac{2\pi m}{2L}x} e^{i\frac{2\pi m}{2L}x} dx = \frac{1}{2L} \int_{-L}^{L} 1 dx = 1 $$, which is indeed normalized.


The complex representation here is a convenient way to express the Fourier basis, but we can also express it in terms of cosine and sine functions which gives us nice physical interpretations. This integer multiples of the base frequency (or simply called harmonics), with arbitrary phase in that frequency controlled by the relative coefficient of the cosine and sine functions. 


Expanding $$c_n = a_n + i b_n$$ and $$e^{i \omega n x } = \cos(\omega n x) + i \sin(\omega n x)$$ where $$\omega = \frac{2 \pi}{2L}$$, we have

$$
\begin{align*}
f(x) &= \sum_{n=-\infty}^{\infty} c_n e^{i \omega n x} \\
&= \sum_{n=-\infty}^{\infty} (a_n + i b_n) (\cos(\omega n x) + i \sin(\omega n x)) \\
&= \sum_{n=-\infty}^{\infty} (a_n \cos(\omega n x) - b_n \sin(\omega n x)) + i \sum_{n=-\infty}^{\infty} (b_n \cos(\omega n x) + a_n \sin(\omega n x))
\end{align*}
$$

Now, let's simplify it doing a sum over non-negative $$n$$, from $$0$$ to $$\infty$$. We'll start with the real part of the function decomposition of $$f$$, which is given by:

$$
\begin{align*}
\text{Re}[f(x)] &= \sum_{n=-\infty}^{\infty} (a_n \cos(\omega n x) - b_n \sin(\omega n x)) \\
&= a_0 + \sum_{n=1}^{\infty} (a_n \cos(\omega n x) - b_n \sin(\omega n x) 
 + \sum_{n=1}^{\infty} (a_{-n} \cos(-\omega n x) - b_{-n} \sin(- \omega n x) \\
&= a_0 + \sum_{n=1}^{\infty} (a_n + a_{-n}) \cos(\omega n x) + (- b_n + b_{-n}) \sin(\omega n x)  \\
\end{align*}
$$

The complex part of the function decomposition of $$f$$ is given by:

$$
\begin{align*}
\text{Im}[f(x)] &= \sum_{n=-\infty}^{\infty} (b_n \cos(\omega n x) + a_n \sin(\omega n x)) \\
&= b_0 + \sum_{n=1}^{\infty} (b_n \cos(\omega n x) + a_n \sin(\omega n x))  + \sum_{n=1}^{\infty} (b_{-n} \cos(-\omega n x) + a_{-n} \sin(- \omega n x) \\
&= b_0 + \sum_{n=1}^{\infty} (b_n + b_{-n}) \cos(\omega n x) + (a_n - a_{-n}) \sin(\omega n x)  \\
\end{align*}
$$

If the function $$f(x)$$ is real, then $$\text{Im}[f(x)] = 0$$, which implies that $$b_0 = 0$$ and $$a_n = a_{-n}$$ and $$b_n = -b_{-n}$$. That is, for real $$f$$, we do not need to compute the coefficients with respect to negative frequencies due the symmetry.

The simplified components for a real $$f$$ becomes:

$$
\begin{align*}
f(x) &= a_0 + \sum_{n=1}^{\infty} (a_n + a_{-n}) \cos(\omega n x) + (- b_n + b_{-n}) \sin(\omega n x)  \\
&= a_0 + 2\sum_{n=1}^{\infty}  a_n \cos(\omega n x) -  b_n \sin(\omega n x)  \\
&= a_0 + 2 \sum_{n=1}^{\infty}  \sqrt{a_n^2 + b_n^2} \cos(\omega n x - \phi_n) \\
\end{align*}
$$

where $$ \phi_n = \tan^{-1} \left( \frac{b_n}{a_n} \right) $$. In this interpretation, any periodic function is a linear combinarion of cosine waves with integer multiples of the base frequency (or simply called harmonics), with arbitrary phase of that frequency controlled by the relative coefficient of the cosine and sine functions. This is the physical interpretation for a real signal $$f$$ we alluded to earlier.

In practice, the complex representation is more convenient to use and has wide adoption. It is also a generalization of the sine and cosine representation that is applicable for both real and complex functions.



#### Example 3: Laguerre Polynomials as Orthonormal Basis on Expontential Decay Measure

In the context of Hilbert spaces, another example of an orthonormal basis is provided by the Laguerre polynomials. The space of square-integrable functions over the interval $$[0, \infty)$$, denoted as $$L^2[0, \infty)$$, serves as an example where these polynomials form a countable orthonormal basis.

**Definition**:
The $$n^{th}$$ Laguerre polynomial, denoted as $$L_n(x)$$, can be defined through Rodrigues' formula:

$$
L_n(x) = \frac{e^x}{n!} \frac{d^n}{dx^n} \left(e^{-x}x^n\right)
$$

In more concrete terms, the first few Laguerre polynomials are as follows:

$$
\begin{align*}
L_0(x) &= 1 \\
L_1(x) &= 1 - x \\
L_2(x) &= \frac{1}{2}(x^2 - 4x + 2) \\
L_3(x) &= \frac{1}{6}(-x^3 + 9x^2 - 18x + 6) \\
&\vdots
\end{align*}
$$

These polynomials are orthogonal over the interval $$[0, \infty)$$ with respect to the weight function $$w(x) = e^{-x}$$. In other words, they satisfy the following orthogonality condition:

$$
\int_{0}^{\infty} L_m(x) L_n(x) e^{-x} dx = \delta_{m,n}
$$

Here, $$\delta_{m,n}$$ is the Kronecker delta function.


We can see that a different weight function, or a different measure, would give rise to a different set of orthonormal basis functions. This is a powerful concept that we will revisit later in the context of convolution models, especially in the construct of the Hippo matrix.

<!--
In summary, Laguerre polynomials form an orthonormal basis in the Hilbert space $$L^2[0, \infty)$$ with the inner product defined using the weight function[^exp-measure] $$w(x) = e^{-x}$$. They provide a powerful mathematical tool for representing and analyzing functions in these spaces, particularly those that exhibit exponential or decay behavior. 
-->


#### Approximating Real-World Functions with Legendre vs. Fourier Basis

Here, we show a Jupyter notebook where we take example functions and approximate them using the Legendre and Fourier basis.

TODO -- add notebook here





<br>
<br>
<br>
<br>
<br>
<br>
<br>


------


## C. State Space Models

<!--
Give intuition behind state spaces. State spaces can be used to model complex dynamics of a state vector. Given an example for a Harmonic oscillator from mechanics, and vary the input function (force) and see if the state behaves as expected.
-->

State space models are a class of models used to describe the time evolution of systems. They are widely used in many fields, including control theory, signal processing, and machine learning. In this section, we will give a brief introduction to state space models, and show how they can be used to model complex dynamics.

In the continuous case, state space models can be written as:

$$
\begin{align*}
\frac{d}{dt} \mathbf{x}(t) &= A \mathbf{x}(t) + B \mathbf{u}(t) \\
\mathbf{y}(t) &= C \mathbf{x}(t) + D \mathbf{u}(t) =  C \mathbf{x}(t)
\end{align*}
$$

$$\mathbf{x}$$ is a N-dimensional vector, $$A$$ is an NxN matrix, B is Nx1, and C is 1xN. $$\mathbf{u}$$ is usually called the input vector, and $$\mathbf{y}$$ is the output vector.
In most cases $$D$$ is assumed to be $$0$$.

For numerical purposes, we can discretize the state space model where we have $$u_k$$ and $$x_k$$. In this discrete case, we can approximate $$x_k$$ by using the derivative at $$k$$, or also the average of the derivative at $$k$$ and $$k-1$$. That is , 

$$x_{k} \approx x_{k-1} + \frac{\Delta}{2} (x'_{k} + x'_{k-1}) + O(\Delta^2)$$

Together with the state space equations, we can show that 

$$
\begin{align*}
x_{k} &= (I - \frac{\Delta}{2} A)^{-1} (I + \frac{\Delta}{2} A) x_{k-1} + \frac{\Delta}{2} B (I - \frac{\Delta}{2} A)^{-1} u_k \\
& \  \text{or more succinctly}\\
x_{k} &= \bar{A} x_{k-1} + \bar{B} u_k \\
\end{align*}
$$

where $$\Delta$$ is the step size and $$y_k = C x_k $$.



#### Example: Harmonic Oscillator

Let's develop some intuition for what state space models can do. We will consider a simple example of a spring attached to a mass $$m$$. The spring has a spring constant $$k$$, and the mass is attached to a wall. The mass is also subject to a force $$u(t)$$.

The dynamics of the system is described by the following differential equation:

$$
m  y''(t) = -k y(t) + u(t)
$$

where $$y(t)$$ is the displacement from equilibrium of the mass at time $$t$$. We know that in the case of $$u(t) = 0$$, this should be a simple harmonic oscillator where the solutions are pure sine wave with certain phase (depending on the initial position and velocity). Let's see how well we can model this system using a state space model. 

While state space models looks linear at first glance, we will see that it can describe non-linear dynamics. The key is how we define the state $$\mathbf{x}$$. Let $$v(t) = \frac{dy}{dt}$$ denote the velocity. In this case, we can see that the differential equation above can be written as:

$$
v'(t) = - \frac{k}{m} y(t) + \frac{1}{m} u(t)
$$

If we define the state as 

$$x = 
\begin{bmatrix}
y \\
v 
\end{bmatrix}
$$

then we can describe the differential equation with state space model as:

$$
\mathbf{x}'(t) = 
\begin{bmatrix}
y'(t) \\
v'(t)
\end{bmatrix}
= \begin{bmatrix}
0 & 1 \\
-\frac{k}{m} & 0 \\
\end{bmatrix} 
\begin{bmatrix}
y(t) \\
v(t)
\end{bmatrix}
+ \begin{bmatrix}
0 \\
\frac{1}{m} \\
\end{bmatrix} u(t)
$$

The upper row simply says $$y'(t) = v(t)$$, the definition of velocity. The lower row says $$v'(t) = -\frac{k}{m} y(t) + \frac{1}{m} u(t)$$, exactly the equation for the acceleration. Then, we extract out the position by 

$$
y(t) = \begin{bmatrix}
1 & 0 \\
\end{bmatrix} \mathbf{x}(t) = C \mathbf{x}(t)
$$


Below, we adapt the code given in [The Annotated S4](https://srush.github.io/annotated-s4/) for state space models.


<!-- show the notebook -->


In this case we use the initial condition such as 

$$
\mathbf{x}[0]
= \begin{bmatrix}
1 \\
0 \\
\end{bmatrix}
$$

which corresponds to initial position at $1$ without velocity. We also use $$u(t) = 0$$, meaning that no additional force is applied. We can see that the position calculated via the discretized discrete state space follows the sine equation quite perfectly. 


Just for fun, let's also consider another case where we start from initial position and velocity $$0$$, but with $$u[t]$$ being something like a Dirac Delta function which injects a fixed momentum at time $$t=0$$. In this case, $$u[0] = 1$$ and $$u[t] = 0$$ for all $$t > 0$$. As shown below, the state space models are able to capture the dynamics quite nicely. Let's see I'm quite convinced that state space models are quite powerful and can capture complex dynamics. 



### Convolution Models

Now we are almost ready to talk about a family of state-space inspired models that are able to model long range dependencies such as S4 and H3. First, we discuss the Hippo matrix and what it represents.


#### Convolution View of State Space Models

<!-- discuss how state spaces can be unrolled -- see the matrix view
-->
Now let's think about how we can process a vector input $$\mathbf{u} = \{ u_k \}_{k=0}^L$$ to obtain an output $$\{y_k\}$$ in batch. Due to the recurrence relations, we can write the output $$y_k$$ as 

$$
\begin{align*}
y_k &=  \sum_{i=0}^{k} \bar{C} \bar{A}^{k-i} \bar{B} u_i  + \bar{C} \bar{A}^k x_0 \\
y_k &=  \sum_{i=0}^{k} \bar{K}_{k-i} u_i + 0 \\
\end{align*}
$$

where $$\bar{K}_n = \bar{C} \bar{A}^n \bar{B}$$ and $$x_0$$ is the initial state, which is often taken to be zero. In this form, we see that $$y_k$$ is a convolution between the L-dimensional vector $$\mathbf{\bar{K}}$$ and the input vector $$\mathbf{u}$$. It means that we can think of the state space model as a convolution model, where the convolution kernel is $$\mathbf{\bar{K}}$$. This unrolled view is useful to process the entire input sequence, either during inference or training. 

That is, if we have a sequence of length $$L$$ as an input, once if we the kernel $$\mathbf{\bar{K}}$$, we can compute the entire output $$\mathbf{y}$$ in $$O(L \log L)$$ time. Such $$\mathbf{y}$$ can be used to predict the next step. In such next step, we can then predict subsequent step in constant time due to the recurrence property!


Now, you may wonder what would be the matrices $$A$$, $$B$$, $$C$$ that we should use? Remember from our earlier examples for the harmonic oscillator that $$A$$ controls the evolution of the dynamical system. How should we construct or interpret such a system to model something such as language modeling or time series forecasting? Next, we discuss the Hippo matrix, which defines a type of matrix $$A,B$$ that can be used to model long range dependencies.



#### Hippo Matrix

The Hippo matrices {% cite hippo %} are the $$A,B$$ matrices associated with state space models, obtained for `input memorization` problem where we want the state $$x_{t}$$ to capture the entire function $$u_{t' \le t}$$, which is the input function up to time $$t$$. This can be very useful for long range modeling where we want the **output feature at a given time to represent the entire past!** There are a few key points before getting into the details on how to do this.

First, the state vector $$x_{t}$$ is $$N-$$ dimensional. How do we use a finite vector to represent an entire function? (Hint: think about orthonormal basis in function space). What if we want to weigh different points $$u(t')$$ differently, for example, if we want to emphasize on the recent past more than the distant past?


Remember that for a given dynamics that we want to model via state space (e.g. harmonic oscillator, or input memorization), we are free to choose the definition of such state vector $$x$$ and the matrices $$A,B$$. Below, we outline the core idea of the Hippo framework for the input memorization problem.

* We need to define how we want to weigh different points. For the Hippo matrix that can perform long range modeling, a sensible choice would be to use uniform weighting so that we can take signals from points far away. In a more technical term, we choose the appropriate measure that defines the integral, which defines the inner product for the Hilbert space. For uniform weighting, we can use the Lebesgue measure, scaled such that the total measure is 1 from time $$0$$ to $$t$$ when we want to memorize input up to time $$t$$. Let's call this measure $$\mu_{t}$$, which can depend on $$t$$.

* Based on inner product (which incorporates the weight), we can then choose an orthonormal basis. For uniform weighting, the (scaled) Legendre polynomials form an orthonormal basis $$\{ g_n \}_{n=0}^{\infty}$$ with respect to the inner product $$\langle \cdot, \cdot \rangle_{\mu_{t}}$$.

* We choose the $$N-$$dimensional state vector $$x(t)$$ to be exactly the coefficients $$c_0(t), c_1(t), \dots, c_{N-1}(t)$$ where $$c_n(t) = \langle g_n , u_{t'\le t} \rangle_{\mu_{t}} $$. The function representation $$g_{t}$$ via the Legendre polynomials is $$g_{\le t}(t') = \sum_{n=0}^{N-1} c_n(t) g_n(t')$$.


* Since these coefficients correspond to the projection of $$u_{t'\le t}$$ onto the basis $$\{ g_n \}_{n=0}^{\infty}$$, the function representation minimizes the distance between $$g_{\le t}$$ and the true input $$u_{\le t}$$. In other words, the function represented by the coefficients $$c(t)$$ is the best N-dimensional approximation of $$u_{\le t}$$ in the Hilbert space $$\mathcal{H}$$ with respect to the inner product $$\langle \cdot, \cdot \rangle_{\mu_{t}}$$. As $$N$$ becomes larger, the distance becomes arbitrarily small.


* With this definition $$c_n(t)$$, we can show that $$\frac{d}{dt} c_n(t) = \frac{d}{dt} \langle g_n, u_{\le t} \rangle $$ can be expressed a linear combination of $$c_m(t)$$ and $$u(t)$$, which means it is a state space model! The details can be found in {% cite hippo %}, Appendix D. There are derivations for different measures as well.

* That is, in the vectorized form, the derivative of the state vector $$x$$ can be written as


$$
\frac{d}{dt} x(t) = A(t) x(t) + B(t) u(t),
$$

where

$$
\begin{align*}
A(t) &= - \frac{1}{t} A \\
B(t) &= \frac{1}{t} B \\
\end{align*}
$$

and $$A,B$$ are the associated Hippo matrices obtained from the derivation outlined above.

$$
\begin{align*}
A_{ij} &= \begin{cases}
(2i+1)^{1/2} (2j+1)^{1/2}   & \text{if   } \ i > j \\
i+1                         & \text{if   } \ i = j  \\
0                           & \text{if   } \ i < j
\end{cases}
\\
B_i &= (2i+1)^{1/2}
\end{align*}
$$

* This is a **time-varying** state space model since the $$A(t)$$ matrix depends on $$t$$! Due to this time dependence, it no longer be expressed as a convolution.

* However, there is a remedy! Dropping $$\frac{1}{t}$$ actually works in practice and enables long range modeling {% cite s4 lmu %}, and according to {% cite train_hippo %}, we can show that this time-invariant version is a valid state space model that corresponds to using exponentially warped Legendre polynomials. Very cool!




<!--
You might be thinking, how do we represent an entire function? The answer is to use a special state space model setup where the state space dynamics is such that (1) the state represent the coefficients of a dense basis in function space and (2) the dynamics of the state space is such that the function induced by the coefficients is close to the input function. Let's see this in equations and also in diagrams.
-->



* The next challenge, to be addressed in S4 {% cite s4 %} and diagonal S4 {% cite s4d %}, is how to compute the convolution kernel $$\mathbf{\bar{K}}$$ efficiently. Observe that the convolution kernel requires obtaining $$A^\ell$$ for all $$\ell = 0, \dots, L$$. Since $$A$$ is an $$N \times N$$ matrix, and we need to do it $$L$$ times, the computational complexity with naive matrix multiplication is $$O(N^2L)$$. The paper S4 {% cite s4 %} describes how to do this efficiently in $$O(N+L)$$ and S4d {% cite s4d %} describes how to do this even more efficiently by using an approximate diagonal matrix $$A$$. We will omit the proofs in this blog post.

** TODO -- see how efficient for the diagonal case. and what big O tilde really means


## D. Einstein Summation for General Tensor Operations


We will describe both attention and linear attention in terms of einstein sum which will provide a concise notation on what dimension is being reduced (summed), and can help us understand how the different tensors ``flow`` together to compose the output.

We denote $$\sum_{d}$$, for example, to be a summation along the $$d$$ axis. Let $$\sigma$$ be a non linear operator where we may denote $$\sigma_L$$ to emphasize that the operation $$\sigma$$ is non linear over dimension $$L$$ in this case. In short, attention operation can be written as 

$$
\begin{align*}
O_{hm} &= \sum_m V_{hmv} \ \sigma_m \left( \sum_{k} Q_{hnk} K_{hmk} \right) \\
&= \sum_m V_{hmv} \  \sigma_m \left( W_{hmn} \right) \\
&= \sum_m V_{hmv} \ W'_{hmn}
\end{align*}
$$

If $$\sigma_m$$ is identity, we can rearrange things where K and V  equivalently interact with each other first. That is,

$$
\begin{align*}
O_{hm} &= \sum_m V_{hmv}  \left( \sum_{k} Q_{hnk} K_{hmk} \right) \\ 
&= \sum_k Q_{hnk} \left( \sum_m V_{hmv} K_{hmk} \right)
\end{align*}
$$

This is a fully linear system given $$Q, K,V$$. We can introduce some non linearity over the key length back by replacing $$K$$ and $$Q$$ with $$K'=\phi(K)$$ and $$Q' = \phi(Q)$$, where $$\phi$$ is a non linear operator over axis $$m$$. This is the idea behind linear attention {% cite linear-attention %}.

$$
\begin{align*}
O^{\text{Linear Attention}}_{hm} &= \sum_m V_{hmv}  \left( \sum_{k} Q'_{hnk} K'_{hmk} \right) \\
&= \sum_k Q'_{hnk} \left( \sum_m V_{hmv} K'_{hmk} \right)
\end{align*}
$$


## E. Attention and Linear Attention



--------
--------
--------




# Long Convolution Models


## Hippo Matrix for History Representation via State Space Models



## S4: Structured State Space



## H3: Hungry Hungry Hippos



## GSS: Gated State Space Models



## Hyena Hierarchy: 

* Implicit convolution means that given an input $$x$$, the convolution filter $$h_\theta(x)$$ depends on $$x$$ and also potentially some parameters $$\thetea$$.

* The position encodings used is illustrated in Figure ().

* 



## A Unified Perspective on Long Convolution Models

Key ideas to highlight

* Need to show attention and convolution as data-dependent operators very clearly. 




### Observations

Interesting observations
* We can see the attention in transformers as a data-dependent operator, controlled by query and key. Such data-dependent operator acts on then value vector V, which then produces the output O.
* Convolution models can also be seen as data-dependent operators, where the operator is controlled by the input function. The key difference here is that due to the convolution theorem and Fast Fourier Transform, the operator can be computed in $O(L log L)$ instead of $O(L^2)$. 
* 



<!--
Observations that we should incorporate.

Discrete SSM breaks down at Nyquist frequency (or sampling frequency).
- should analyze transformers and see how it behaves as well probably
- this may shed some light on positional embeddings


- maybe add in FAQ: how do we make things causal? do we need additional masking? this might be confusing at first. State space models are causal by construction!


- in the perspective of building neural network, maybe draw some analogy that u is the input from the previous layer!!!

- for SSM, how do we extend it to vector instead of operating on numbers for each time (u is a vector of numbers)? is there a mixing benefit at the end? [think hard about this]

- Questions that come to mind that users might appreciate as well: are there any limits where the state space models cannot capture the dynamics faithfully? Seems like we can model quite arbitrary dynamics since many differential equations can be expressed via state space? 


- show an example that state space can model n'th order diff equations

- 


-->



<!-------
                Bibliography and Footnote 
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
###########################################################
-------->



#### References
{% bibliography --cited %}


#### Notes

[^hilbert-space]: A Hilbert space $$\mathcal{H}$$ is a vector space equipped with an inner product $$\langle \cdot, \cdot \rangle$$ that is complete with respect to the norm induced by the inner product $$ \lVert \cdot  \rVert = \sqrt{ \langle \cdot, \cdot \rangle } $$. Completeness means that any Cauchy sequence in $$ \mathcal{H} $$ converges to an element in within $$\mathcal{H}$$ itself. The convergence is with respect to the norm, which provides a notion of distance. To elaborate on completeness further, a Cauchy sequence is a sequence such that elements $$v_n, v_m$$ are getting closer and closer together as $$n,m \to \infty$$. Intuitively, a complete space means that there are no unexpected "holes" in the space where any sequence that is supposed to converge (a Cauchy sequence) actually converges inside the space itself.


<!-- For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Hilbert_space). -->

[^schauder-basis]: A Schauder basis is a *countable* basis that is dense in a complete normed space (a Banach space). To say that a basis is 'dense' means that any element of the space can be arbitrarily closely approximated by a finite linear combination of basis elements. In other words, for every element in the space and any given small positive distance ($$\epsilon$$), there exists a finite sum of basis elements that is within that distance of the given element. This ensures that the basis "spans" the entire space in a limiting sense, even if any specific finite subset of the basis does not span the space. <br>&nbsp;<br> A crucial distinction to note is that a Schauder basis doesn't need to be mutually orthogonal. This is because it's typically defined in the context of a Banach space, where angles or orthogonality might not be relevant concepts. Yet, in a Hilbert space, it's entirely possible to orthogonalize a Schauder basis using the Gram-Schmidt process. 

<!-- 
discuss the significance of completeness .. it might be a distraction
-->


[^convergence-in-norm]: The convergence here is in the norm defined by the inner product of the Hilbert space.

[^periodic]: $$f$$ is periodic in the interval $$[-L,L]$$ if $$f(-L) = f(L)$$. In practice, any function can be made to be periodic via padding. For instance, if a function is defined over $$[-L, L]$$, then we can pad with zeros from $$-L-1$$ to $$-L$$ and also from $$L$$ to $$L+1$$. This would result a function that is periodic over $$[-L-1, L+1]$$.

[^exp-measure]: Or more formally, we say that the inner product is defined via the measure $$d \mu = e^{-x} dx$$. In the Fourier case, we can also think of the inner product as being defined via the measure $$d \mu = \frac{1}{2L} 1_{x \in [-L,L]} dx$$.


<!-- L2 is isomorphic to ell2 -->

<!-- For more details, see [Wikipedia](https://en.wikipedia.org/wiki/Schauder_basis). -->

<!--
in the Hilbert space, meaning that any element in such Hilbert space can be approximated arbitrarily close by a linear combination of the basis elements. The notion of a Schauder basis makes sense given the completeness of such space (a Hilbert space is complete by definition) where completeness gaurantees that there are no ``holes`` in the space and a sequence that should converge actually converges (formally, any Cauchy sequence converges). The importance of being countable is that we can represent the basis elements as a sequence, rather than an continuous integral.
-->


<!--
# Post Appendix -- Temporary Scratchpad


The Hippo matrix is quite special in that we consider the case where the measure is uniform -- that is, the weight of the integral anywhere within the interval does not change. There are cases where this is not true, for example, if we operate in the case where the measure has exponential decay. This would give rise to a different set of orthonormal basis functions, such as the Laguerre polynomials. However, the authors argue that for long range modeling, the uniform measure is more appropriate. For more details, see [...].


$$
\begin{align*}
c_0 &= a_0 b_0 \\
c_1 &= a_0 b_1 + a_1 b_0 \\
c_2 &= a_0 b_2 + a_1 b_1 + a_2 b_0 \\
&\vdots \\
c_{N-1} &= a_0 b_{N-1} + a_1 b_{N-2} + \dots + a_{N-1} b_0 \\
c_N &=  a_2 b_{N-2} + \dots + a_{N-1} b_1 \\
c_{N+1} &= a_2 b_{N-1} + \dots + a_{N-1} b_2 \\
&\vdots \\
\end{align*}
$$



or equivalently:

$$
\mathbf{c} = 
\begin{bmatrix}
c_0 \\
c_1 \\
c_2 \\
c_3 \\
c_4 \\
c_5 \\
c_6 \\
\end{bmatrix}
=
\begin{bmatrix}
b_0 & 0 & 0 & 0 \\
b_1 & b_0 & 0 & 0 \\
b_2 & b_1 & b_0 & 0 \\
b_3 & b_2 & b_1 & b_0 \\
0 & b_3 & b_2 & b_1 \\
0 & 0 & b_3 & b_2 \\
0 & 0 & 0 & b_3 \\
\end{bmatrix}
\begin{bmatrix}
a_0 \\
a_1 \\
a_2 \\
a_3 \\
\end{bmatrix}
$$


from LaTeX It

\begin{pmatrix}
y_0 \\
y_1 \\
y_2 \\
y_3 \\
\vdots \\
y_{L-1} 
\end{pmatrix} 
=
% S_h \mathbf{h}
%S_h = 
\begin{pmatrix}
h_0 & 0   & 0   & 0  & \cdots & 0 \\
h_1 & h_0 & 0   & 0  & \cdots & 0 \\
h_2 & h_1 & h_0 & 0  & \cdots & 0\\
h_3 & h_2 & h_1 & h_0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \cdots & \vdots\\
h_{L-1} & h_{L-2} & h_{L-3} & h{L-4} & \cdots & h_0 
\end{pmatrix}
\begin{pmatrix}
u_0 \\
u_1 \\
u_2 \\
u_3 \\
\vdots \\
u_{L-1} 
\end{pmatrix} 

-->