---
layout: post
title: "Lossless Tokenizer via Byte-level BPE with Tiktoken"
date:   2023-09-30
description: The design of Tiktoken, a byte-level BPE tokenizer behing GPT.
tags: llms tokenizer
categories: 
published: true
social: true
---



OpenAI's gpt2 tokenizer is among the first that handles tokenization in a completely lossless way, meaning that there is no unknown token. In my opinion, OpenAI's vision for generality of GPT really shines through from the tokenizer aspect. In this blog we will be analyzing `tiktoken` which is the tokenizer behind GPT models. 
<!-- Towards the end of the blog, we will compare different publicly available tokenizers in terms of losslessness, compression rate, etc. -->


<!-- ### Encoding Text To Tokens -->

We will describe how Tiktoken encodes a text to tokens. There are three main stages. (1) extracting out special tokens that we never want to be broken up into smaller pieces (2) pre-tokenization based on a pre-defined regular expression patterns, resembling breaking up texts into `words` (3) If such a pre-token is not an actual token, this is the stage where we use the byte-level BPE to break up the pre-token into smaller pieces.

<br>
#### Pre-Tokenization

Let's look at the code that performs step (2). Note that step (1) is omitted in the educational Python tiktoken code, but it is in the Rust code [here](https://github.com/openai/tiktoken/blob/main/src/lib.rs#L235). Below is the Python encode function taken from [tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py#L21). 


```python
def encode(self, text: str, visualise: Optional[str] = "colour") -> list[int]:
    """Encodes a string into tokens.
    >>> enc.encode("hello world")
    [388, 372]
    """
    # Use the regex to split the text into (approximately) words
    words = self._pat.findall(text) # pre-tokens based on word boundary rules
    tokens = []
    for word in words:
        # Turn each word into tokens, using the byte pair encoding algorithm
        word_bytes = word.encode("utf-8")
        word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, visualise=visualise)
        tokens.extend(word_tokens)
    return tokens, words
```

For this encode function, a text (still in string, not bytes) is broken into `words` which we will call pre-tokens. For GPT-4 models, the tokenizer's name is `cl100k_base`. The regular expression pattern `self._pat` is defined below. 

```python
>>> from tiktoken._educational import *
>>> enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
>>> end._pat
regex.Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", flags=regex.V0)
```

GPT-4's short description of such regular expression is given below (long description [here](https://chat.openai.com/share/68144071-d1c9-4deb-8e0f-28aba95103cc)). 

 

<blockquote style="font-size: 0.9em;">
This regex captures common contractions (like 's, 't, etc.) in a case-insensitive manner, sequences of letters possibly preceded by a non-letter, non-number character, sequences of 1 to 3 numbers, sequences of non-letter, non-number characters possibly followed by newlines, sequences of whitespace ending with newlines, whitespace not followed by non-whitespace, or any sequence of whitespace characters.
</blockquote>


Let's see some examples of how the regex breaks up text. Below, we can see that such pattern defines the rule for word boundaries such as how to separate non-whitespace and whitespace, and also imposes certain structure such as space-prefix (the use of space right before non-whitespace character such as " x", " +", " y").

```python
>>> import regex
>>> pat = regex.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
>>> pat.findall("hello worlddddd")
['hello', ' worlddddd']
>>> pat.findall("def add(x, y):\n\treturn x + y")
['def', ' add', '(x', ',', ' y', '):\n', '\treturn', ' x', ' +', ' y']
```

<br>
#### Byte-level BPE

<div style="margin-bottom: 1em;"></div>

#####  Base Vocabulary

BPE builds a vocabulary in a bottom-up approach where it merges tokens starting with the base vocabulary. The traditional BPE starts with a set of characters. The set of all possible characters is very large, and it is growing as we speak. For example, the unicode standard has over 100,000 characters. This makes it very difficult to build a lossless tokenizer. 

However, these Unicode characters are composed of smaller elements, that is, the bytes. Since there are only 256 base bytes which can represent *any* text, we can build a lossless tokenizer where there is absolutely no unknown token. This is a neat tokenizer design that GPT was among the first to adopt (if not the first). (show huggingface implementation). Before GPT, many other models use all sorts of tricks to manage the unknown tokens such as normalization.


Let's see some example of the byte representation of a few Unicode characters. We can see that a normal English character such as 'a' is represented by a single byte. However, a Japanese character such as 'カ' is represented by 3 bytes. An emoji 🐱 is represented by 4 bytes.

```python
>>> "a".encode('utf-8')
b'a'

>>> "🐱".encode('utf-8')
b'\xf0\x9f\x90\xb1'

>>> "カ".encode('utf-8')
b'\xe3\x82\xab'
```

In Tiktoken, the 256 bytes are used as the base vocabulary. Even if the tokenizer has not observed any character or phrases before during the training stage, such phrases can be encoded by the bytes.


##### Encoding
<!-- If the pre-token is not an actual token, this is the stage where we use the byte-level BPE to break up the pre-token into smaller pieces.  -->
Now, let's investigate the ```bpe_encode```  function. Each pre-token is broken up into smaller pieces using the byte-level BPE. First, the pre-token is convert into a list of bytes (`parts` in the code below). Then, for each adjacent pair of parts, we check if the pair is in the vocabulary (`mergeable_ranks`). If it is, we obtain the rank. We go through all the pairs and the adjacent pair with the smallest rank is selected to be merged. In the code below, it is enumerating through the zip of `parts[:-1]` and `parts[1:]` which essentially going through all adjacent pairs. Then, we merge the selected pair and leave other parts intact, and repeat such process again. Observe that each part can become longer than 1 byte due to the process of iteratively merging. In the end, we stop when merging is no longer possible (two adjacent parts are not in the vocabulary anymore).


```python
def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: Optional[str] = "colour"
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if visualise:
            if visualise in ["colour", "color"]:
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    if visualise:
        print()

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens
```

Below is and example where `hello worlddddd` is the input for encoding. From above, we see that it is splitted into two pre-tokens, `hello` and `worlddddd`, which is what we observed below where BPE works on `hello` first. Throughout the merging process, observe BPE merges the pair with lower rank first and keep building up parts. Note that the process is deterministic. Given the pre-built vocabulary and a text, the same sequence of merging will always be performed. 


Interesting, we can also see that the emoji is encoded with 3 tokens for 4 bytes (which is rather inefficient, which implies that other tokens are more important/have lower rank). The Japanese character カ however can be represented with only 1 token despite being 3 bytes. It is possible that カ appears frequently enough that it is part of the vocab itself.

<script src="https://gist.github.com/benathi/90fe8be8c939d0c2baf9412204bbd7a8.js"></script>

<br>
### Training a BPE Tokenizer 

To train a BPE tokenizer (that is, to obtain a vocabulary), we iterate through a text corpus, pre-tokenize, the use the bag of words (each word or pre-token is a sequence of bytes) as our data which will be iteratively merged.

First, we add the base bytes (all 256 bytes) to the vocabulary. Then, we iterate by counting the occurrences of each byte pair. Then, the highest frequency byte pair (`a`, `b`) is added to the vocabulary in the form of token `ab`. The data is then processed by merging any occurrences of adjacent `a`, `b` to be `ab`. That wat, at each stage, all the parts are in the vocabulary. We repeat until the size of the vocabulary reaches the desired size. 



<br>

-----

<br>

# Appendix
We show the modified educational tiktoken code here.
<br>
<script src="https://gist.github.com/benathi/5e41cf34617196a65fd1837d1aa07c96.js#L21-L29"></script>


<!-- 
## Implications
- Recently we have observed the ability of LLMs + RLHF to generalize beyond the English data they are trained on. LLMs also have abilties such as multi-lingual chain of thought reasoning where a chain of thought in English generalizes to other languages.
- How does language model handle such low-resource languages without any loss in information despite the tokenizer having only 50K - 100K vabulary size? One crucial aspect is the losslessness of tokenizer.
-->