---
layout: distill
title: Measuring Code Generation Abilities of GPT-4 in 10+ Languages
date:   2023-03-19
description: 
tags: codegeneration
categories: transformers gpt4
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


## Recap: Coding with ChatGPT-4

Over the past week we have seen tons of examples regarding GPT-4's code generation abilities. Here's a quick recap with three of my favorite examples.

<!-- {% twitter https://twitter.com/MengTo/status/1637110344555417600?s=20 %} -->
{% twitter
https://twitter.com/MengTo/status/1636507977795481601?s=20
%}
{% twitter
https://twitter.com/_rockt/status/1636470054417047554?s=20
%}
{% twitter
https://twitter.com/KeisukeS_/status/1636328610255769600?s=20
%}
<!--
{% twitter
https://twitter.com/AndreTI/status/1635801920223989760?s=20
%}
-->



<!--
<br> 
Overall, the examples demonstrate the remarkable potential of GPT-4 for code generation. 
-->


## Evaluating Code Generation in 10+ Programming Languages

After gaining access to GPT-4, I was thrilled to put it to the test with the code generation benchmarks [multi-lingual humaneval](https://huggingface.co/datasets/mxeval/multi-humaneval) and [mbxp](https://huggingface.co/datasets/mxeval/mbxp). The evaluation covered a wide range of programming languages and yielded impressive results, helping to quantify the model's performance in each language.

Overall, the performance improvement from the previous models is quite expected. However, we observed much high scores than the reported number in the GPT-4 paper. (more details below)

|            | code-davinci-02 | text-davinci-003 | ChatGPT-3.5 (1 shot) | ChatGPT-4 (1 shot) |
|------------|:---------------:|:----------------:|:--------------------:|:------------------:|
| **Multi-lingual HumanEval** |
| Python     |      46.3%      |      56.7%       |        73.2%         |       83.5%        |
| Java       |      49.1%      |      52.2%       |        60.9%         |       78.3%        |
| JavaScript |      51.6%      |      58.4%       |        66.5%         |       71.4%        |
| TypeScript |      50.9%      |      55.9%       |        64.6%         |       78.9%        |
| C#         |      45.3%      |      50.9%       |        32.3%         |        6.8%        |
| Go         |      21.9%      |      35.0%       |        34.4%         |       50.0%        |
| Kotlin     |      39.8%      |      50.3%       |        59.0%         |       68.9%        |
| PHP        |      52.8%      |      58.4%       |        63.4%         |       74.5%        |
| Perl       |      36.0%      |      34.2%       |        55.3%         |       68.3%        |
| Ruby       |      39.8%      |      62.1%       |        13.0%         |       80.7%        |
| Scala      |      45.3%      |      46.0%       |        57.1%         |       28.0%        |
| Swift      |      24.8%      |      39.1%       |        48.4%         |       61.5%        |
|            |    **42.0%**    |    **49.9%**     |      **52.34%**      |     **62.58%**     |




## Finding Highlights

Here are some of the key observations.

#### Few-shot prompting can matter a lot for code generation
- Note that we use 1-shot prompting in the main table for ChatGPT-3.5 and ChatGPT-4.
- 1-shot prompting makes much more sense for ChatGPT and outperform the zero-shot case significantly (including what is reported in the GPT-4 paper).


|           | ChatGPT-3.5 (0 shot) | ChatGPT-4 (0 shot) | GPT-4 (0 shot, reported) |
|-----------|:--------------------:|:------------------:|:-----------------------:|
| Python    |         62.2%        |       65.2%        |           67.0%          |



- This is because ChatGPT are more conversation-like, in which case it can be unclear about what we actually want the model to generate without an example (1-shot prompt).
  - In particular, the format of    Multi-HumanEval is such that the prompt consists of the function signature and the expected completion is the function body.
  - In normal settings of Davinci-0X, this format is quite natural and few-shot prompting does not matter much especially since the GPT models are likely familiar with all languages (see studies [paper](https://arxiv.org/abs/2210.14868)).
  - In the case of ChatGPT, the model can get confused whether to (1) continue generating function body (2) regenerate everything including the function signature. Such confusion can cause the execution evaluation to *unnecessarily* fail even though the code might be correct.
  - By providing a clear example via 1-shot prompting, the performance increased significantly.




#### GPT-4 consistently outperformed its predecessor, GPT-3 Davinci as well as GPT-3.5 (ChatGPT), in most programming languages.
  - Note that these benchmarks are generated post-2021 and was not included in the training data for GPT-4. Therefore, we believe the evaluation results to be valid without any data contamination.
  - The exception where the scores decrease are also related to formatting. For example, for C#, the model does not close the braces for the entire object despite the 1-shot prompting as an example.
  - Note that for Davinci-0X, since the model generally lacks the ability to precisely stop on its own; therefore, we add some logic to close all the functions, classes, and objects. However, the evaluation on ChatGPT is slightly more strict where we expect the model to generate code that can be readily executed. Even with the stricter test, it is clear that the performance is getting much better compared to the previous generation models.




### Examples

#### Prompt
{% highlight python %}
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
{% endhighlight %}

#### Completion
{% highlight python %}
```
def separate_paren_groups(paren_string: str) -> List[str]:
    """
    Separates groups of nested parentheses into separate strings and returns the list of those strings.
    
    Args:
    paren_string: A string containing multiple groups of nested parentheses.
    
    Returns:
    A list of strings, where each string is a separate group of nested parentheses.
    
    Example:
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    stack = []
    result = []
    current = ''
    for char in paren_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
        current += char
        if not stack:
            result.append(current.replace(' ', ''))
            current = ''
    return result
```
{% endhighlight %}

<br> The syntax and code does seem correct in this case. However, repeating the function signature again makes it more susceptible to error.


#### completion given 1 shot prompt
In this case, we provided a 1-shot prompt (not shown) where the completion contains only the function body.
During evaluation, we see below that the generated code contains only the desired function body and is a continuation of the function signature prompt.

{% highlight python %}

    stack = []
    groups = []
    current_group = ''
    for char in paren_string.replace(' ', ''):
        if char == '(':
            stack.append(char)
            current_group += char
        elif char == ')':
            stack.pop()
            current_group += char
            if not stack:
                groups.append(current_group)
                current_group = ''
    return groups

{% endhighlight %}



### Getting Started with Code Generation Evaluation


If you are interested in experimenting with code generation and want to perform evaluations similar to the ones presented in this blog post, you can access the code at the following GitHub repository: `https://github.com/amazon-science/mxeval`. This repository provides a comprehensive set of tools to perform code generation evaluations across a variety of programming languages.


Our datasets are also available on Huggingface Datasets.
```
https://huggingface.co/datasets/mxeval/multi-humaneval
https://huggingface.co/datasets/mxeval/mbxp
```


Part of the evaluation code used for this blog is also available in a fork of OpenAI evals `https://github.com/benathi/evals/tree/multi-humaneval-pr`. One can run the evaluation by running `oaievals multi-humaneval-js` for javascript, for example.

Note that we built upon the Python-only HumanEval benchmark developed by OpenAI, as well as the MBPP benchmark created by Google, to expand the scope of evaluation to over 10 programming languages. We gratefully acknowledge the pioneering work of OpenAI and Google in this area.


<!--

### Citation Information

```
@inproceedings{
athiwaratkun2023multilingual,
title={Multi-lingual Evaluation of Code Generation Models},
author={Ben Athiwaratkun and Sanjay Krishna Gouda and Zijian Wang and Xiaopeng Li and Yuchen Tian and Ming Tan and Wasi Uddin Ahmad and Shiqi Wang and Qing Sun and Mingyue Shang and Sujan Kumar Gonugondla and Hantian Ding and Varun Kumar and Nathan Fulton and Arash Farahani and Siddhartha Jain and Robert Giaquinto and Haifeng Qian and Murali Krishna Ramanathan and Ramesh Nallapati and Baishakhi Ray and Parminder Bhatia and Sudipta Sengupta and Dan Roth and Bing Xiang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Bo7eeXm6An8}
}
```
--> 