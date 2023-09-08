---
layout: post
title: Lossless Tokenizer for Language Models
date:   2023-02-19
description: Analyzing GPT Tokenizer Design
tags: llms tokenizer
categories: 
published: true
---


## Examples 
- motivate with the ability of LLMs + RLHF to generalize beyond the languages of alignment data the models are trained on.
- also topics like language models are multi-lingual chain of thought reasoners
- How does language model handle such low-resource languages without any loss in information / unknown characters? leads to the next question


OpenAI's gpt2 tokenizer is among the first that handles tokenization in a completely lossless way, meaning that there is no unknown token whatsoever. OpenAI's vision for generality of gpt really shines through from the tokenizer aspect.

In this blog we will be analyzing the gpt2 which is a basis of the newly released `tiktoken` tokenizer. Towards the end of the blog, we will compare different publicly available tokenizers in terms of losslessness, compression rate, etc.

## Byte-Level BPE for Lossless Tokenizer
## Lossless Tokenizer



### Base Vocabulary

BPE builds a vocabulary in a bottom-up approach where it merges tokens starting the `base vocabulary`. The traditional BPE starts with a set of characters. For English, if we 



- Start from a base vocab, which could be, for example, a set of characters.

- describe how many unicode. how many? it is also growing given emojis etc. I think.
- Why character-level unicode cannot be lossless since the base vocab is very high.
- This motivates Byte Level since there are only 256 bytes -- we can use these as the base vocab.



## Practical Implementation

### Space Prefix

### Pretokenization


### 