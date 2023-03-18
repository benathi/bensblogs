---
layout: distill
title: ChatGPT-4 on Physics Olympiad Problems
date:   2023-03-18
description: 
tags: #gpt4 #aiforscience
categories: gpt4
published: true
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


## How Well Does GPT-4 Understand Physics?

Exploring the limits of human-like language models has always been an exciting endeavor, and with the recent release of GPT-4, the possibilities seem endless. As a physics enthusiast, I decided to put this powerful tool to the test by feeding it a series of challenging Physics Olympiad questions. In this blog, I'll share my findings, analyzing GPT-4's performance, strengths, and limitations when it comes to solving complex physics problems. Let's look at the questions and ChatGPT-4's answer below.


<br>

**Disclaimer**: Please note that the grading process for ChatGPT's answers is subjective and based on my own standard. In line with the grading methodology used in the actual competition, partial credits will be given for correct steps towards the solution, even if the final answer is incorrect.





<embed src="{{site.baseurl}}/assets/pdf/IPhO-2011-P1-gpt4.pdf" type="application/pdf" width="100%" height="600px" toolbar="0" scrollbar="0" />


<br>

* See the full solution [here](https://s3.eu-central-1.amazonaws.com/physprob.com/files/ipho/2011_Thailand_p1Sol.pdf)


### Impression
- GPT-4 certainly understands physics concepts to some degree.
- The weakest part, relative to my expectation, is actually the equation solving abilities.
- Equation solving is quite deterministic so I am a bit surprised when GPT-4 output something that seems plausible but incorrect. Had it spent more time double checking and deriving the solution, I have no doubt the model would get it correct.
- That being said, there are certain logic that are not quite correct. Is this human-level abilities however, I'd say totally!
- Another common error is a syntactic LaTex error where a newline token `//` is often produced as `/` which does not get rendered as newline. I had to do manual fixes.
- Also, for long generation, the model often get stopped before it finishes generating everything. I need to ask the model to continue with something like `please continue starting from XXX`.


### Are we close to ASI? (Artificial Superintelligence)
The development of Artificial Superintelligence (ASI) remains a topic of great interest and speculation in the field of AI. While significant progress has been made in the advancement of Artificial General Intelligence (AGI), further breakthroughs in science knowledge and problem-solving are needed to move closer to ASI. The evaluation of challenging science problems using AI language models, such as ChatGPT, may help guide us in that direction.

For those interested in exploring this topic further, a collection of challenging science problems in TeX format, including problems from the International Physics Olympiad (IPhO), will soon be available on Github. 



