---
layout: distill
title: Periodicity - The Real Hero of Fourier Transform
date:   2023-09-13
description: A personal note on Fast Fourier Transform.
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

## TL;DR

- Fourier transform shows up everywhere -- in AI, Fourier transform is used in  neural operator for function to function mapping, Hyena model for long range modeling, etc. 
- Fourier transform as a way to represented a function relies on the fact that the Fourier series forms an orthonomal basis. The orthonormal part means that we can extract out the coefficients of each basis term via integral. The basis part means that any function can be represented as a linear combination of the basis terms arbitrarily well.
- However, the Fourier series is not the only orthonormal basis -- there are many others such as Legendre polynomials, Bessel functions, Laguerre polynomials, etc.
- One key feature of Fourier basis is its periodicity which allows for Fast Fourier Transform, reducing the complexity from $$O(N^2)$$ to $$O(N \log N)$$ where N is the number of discrete datapoints. Other orthonormal bases do not have a general approach that has as low as $$O(N \log N)$$ complexity!
- In this blog, we expand on each of these points with some theory and examples.




------

## Usage of Fourier Transform in AI

## A World of Orthonormal Bases In Function Space

- There are many orthonormal bases in function space, such as Fourier series, Legendre polynomials, Bessel functions, Laguerre polynomials, etc. But let's look at what it means to be an orthonormal basis.
- An orthonormal basis is a set of functions that are orthogonal to each other and have unit norm. Here, we operate in the Hilbert space of functions where the inner product of two functions is defined via the integration of the product of the two functions. That is,

$$
\langle f, g \rangle = \int f(x) g(x) dx
$$

We mention without proof that the Fourier series forms an orthonormal basis. The same applies for Legendre polynomials, Bessel functions, Laguerre polynomials, etc. Let's look at each of these in turn.

### Fourier Series

- The Fourier series is a set of sine and cosine functions, conveniently expressed in the complex form $$e^{inx}$$. They form a Schauder basis where any function $$f$$ can be represented as a linear combination of the countable basis elements. That is,

$$
f(x) = \sum_{n=0}^{\infty} a_n e^{i n x}
$$

where $$a_n$$ are the Fourier coefficients. The Fourier basis functions are orthogonal to each other and have unit norm. That is,

$$
\langle e^{i n x}, e^{i m x} \rangle = \int e^{i n x} e^{i m x} dx = \delta_{n,m}
$$

where $$\delta_{n,m}$$ is the Kronecker delta function. The Kronecker delta function is $$1$$ if $$n=m$$ and $$0$$ otherwise.

The coefficients $$a_n$$ can be extracted out via the inner product:

$$
a_n = \langle f(x), e^{i n x} \rangle = \int f(x) e^{i n x} dx
$$

where this process is called the Discrete Fourier transform. (There's some subtlety about periodicity here, but we'll ignore it for now.)





-----

Dive into any function space, and you'll soon be swimming with a variety of orthonormal bases. These are sets of functions that, in a way, act like the building blocks of the space. From Fourier to Legendre, from Bessel to Hermite, these bases have their unique properties and quirks.

In the realm of Hilbert spaces (think of them as geometric spaces of functions where you can measure angles and distances), any function can be approximated — or even represented perfectly — using a combination of these building blocks. It's a bit like expressing a melody using a combination of musical notes.

### Examples --

- this will be taken from a Jupyter notebook including code and plots. Make the code collapsible perhaps.


## So, Why the Spotlight on Fourier?
The Fourier series, and its more general sibling, the Fourier Transform, have an ace up their sleeve: the ability to harness regularities in their basis vectors. This translates to computational efficiency. Enter the Fast Fourier Transform (FFT), a lightning-speed algorithm that has transformed (pun intended) various fields, from audio processing to quantum physics.

This doesn't mean other bases are inferior. Legendre polynomials, for instance, are fantastic for non-periodic functions in a confined interval. But Fourier's basis vectors exhibit a consistent and regular pattern that makes certain computations exceptionally efficient.

Stay with me as we embark on a journey to unpack the magic behind the Fourier Transform and understand why it's become such a staple in the mathematical toolkit.

### time taken to compute Fourier transform vs. Legendre transform


### O(N^2) to O(N log N) Efficiency via Regularity in the Fourier Basis

- Outline: Show the identity that leads to Fast Fourier transform being an exact computation for discrete Fourier transform





## Comparison via Other Convergence Properties

We outline how Fourier series versus Legendre's polynomials compare in terms of other convergence definitions below. Overall, the convergence behavior depends on the function we want to estimate. However, the Fourier series has a big advantage in terms of speed of computation via Fast Fourier Transform (FFT).





## Appendix

### Bits and Pieces
- A Banach space is a normed vector space that is complete with respect to the norm. That is, every Cauchy sequence in the space converges to a limit that is also in the space. It's a more general version of a Hilbert space, which is a Banach space with an inner product.

- A countable sequence of vectors $$ \{ x_n \}$$ is a countable basis of a Banach space $$X$$ if any element $$f \in X$$ can be uniquely represented as a linear combination of the basis elements. That is,

$$
f(x) = \sum_{n=0}^{\infty} \alpha_n x_n ;  \ \ \ \ \ \ \ \alpha_n \in \mathbb{C}.
$$
In the infinite sum case, the required convergence is in the norm. Note: This is different from pointwise convergence -- in fact there exists sequences of functions that converge in the norm but not pointwise, or vice versa.


- If $$X$$ is a space that is not complete, this would pose some challenges since the limit of a Cauchy sequence may not be in $$X$$.



### Convergence Properties of Fourier Series

The Fourier series provides an expansion of functions in terms of sines and cosines, effectively capturing the periodic nature of functions over the interval $$[- \pi, \pi]$$ or any other full period.

#### Pointwise Convergence:

According to Dirichlet's Theorem, the Fourier series of a function $$f$$ converges pointwise to $$f(x)$$ at a point $$x$$ if:
1. $$f$$ is periodic with period $$2\pi$$.
2. $$f$$ is piecewise continuous on any closed interval $$[a, b]$$ of length $$2\pi$$.
3. $$f$$ has a left-hand and right-hand limit at $$x$$ and is of bounded variation in any neighborhood of $$x$$.

Under these conditions, the Fourier series converges to the average of the left-hand and right-hand limits of $$f$$ at $$x$$.

#### Uniform Convergence:

The Fourier series of a function converges uniformly if:
1. $$f$$ is periodic with period $$2\pi$$.
2. $$f$$ is continuous on its period and is of bounded variation on any closed interval $$[a, b]$$ of length $$2\pi$$.

When these conditions are met, the Fourier series converges uniformly to $$f$$ over its period.

#### Convergence in $$L^2$$ Norm:

For the $$L^2$$ norm convergence, Parseval's Theorem provides the conditions. For a function $$f$$ that is square-integrable over one period (i.e., $$f \in L^2([- \pi, \pi])$$), its Fourier series will converge to $$f$$ in the $$L^2$$ norm. This is expressed as:

$$\int_{-\pi}^{\pi} |f(x)|^2 \, dx = \sum_{n=-\infty}^{\infty} |a_n|^2$$

where $$a_n$$ are the Fourier coefficients of $$f$$.

#### Fourier Series in Practice:

The ubiquity of the Fourier series in fields ranging from physics to engineering to music is testament to its practical utility and mathematical elegance. While it's particularly suited for periodic functions, clever extensions like the Fourier transform have broadened its applicability to a wider class of functions.


### Convergence Properties of Legendre Polynomial Series

The Legendre polynomials, denoted as $$P_n(x)$$, form a complete orthogonal basis for functions defined on the interval $$[-1,1]$$ with respect to the weight function $$w(x) = 1$$.

#### Convergence in $$L^2$$ Norm:

For any function $$f$$ that belongs to $$L^2([-1, 1])$$, its expansion in terms of Legendre polynomials can be written as:

$$f(x) \approx \sum_{n=0}^{\infty} a_n P_n(x)$$

where $$a_n$$ represents the projection of $$f$$ onto $$P_n$$. The series converges to $$f$$ in the $$L^2$$ norm. Formally:

$$\lim_{N \to \infty} \int_{-1}^{1} \left| f(x) - \sum_{n=0}^{N} a_n P_n(x) \right|^2 \, dx = 0$$

#### Pointwise and Uniform Convergence:

For smoother functions, the convergence of the Legendre series can be stronger:

- If $$f$$ is continuous on $$[-1,1]$$, then its Legendre series converges pointwise to $$f(x)$$ for every $$x$$ in that interval.
  
- If $$f$$ has derivatives that are continuous and adhere to specific boundary conditions, the Legendre series will converge uniformly to $$f$$ on $$[-1,1]$$.




---
- Other advantages of Fourier transform include interpretability of the coefficients in terms of frequency, etc. It can be used to perform filtering of different frequency components. Also it has a nice property where convolution in the spatial domain is multiplication in the Fourier domain. There field of signal processing is dedicated to such in-depth analysis of Fourier transform and its applications.



----

## Fast Fourier Transform

-- this one is math identity. should be fast to write really -- I have already worked it out somewhere.


## Discrete Fourier Transform and Nyquist Theorem





## Non Periodic 

-- Look into the Fourier transform as an integral which recovers the signal perfectly versus the truncated version (discrete) which essentially assumes periodicity


## 

anything we can say about |a_n| ??

- go over Teschl and see if there are any bits that we can add for additional insights.
- 