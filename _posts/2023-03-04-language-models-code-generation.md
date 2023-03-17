---
layout: distill
title: Unreasonable Effectiveness of LLMs for Code Generation 
date:   2023-03-07
description: 
tags: codegeneration
categories: transformers
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
toc:
  - name: Out of Domain Generalization
  - name: Natural Co-Occurrences of Multi-lingual Knowledge
  - name: Multi-ligual versus Mono-lingual
  - Large Large Multi-Lingual Models Really Shine
  - name: Zero-Shot Translation
  - name: Few-Shot Prompts Helps LLMs on Out-of-Domain Languages


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

At this point, we are no longer surprised about what language models can do.
However, it is still unclear how language models derive such amazing abilities especially in the area of code generation. This blog discusses the highlights from the paper [Multilingual Evaluation of Code Generation Models](https://arxiv.org/pdf/2210.14868.pdf) which give some clue as to how LLMs are so great at coding.



## Out of Domain Generalization
If we train a model on one programming language, it turns out that such a model can also **write code in different programming languages**, especially when the model is large enough!  Let's look at the results and sample generations.


Here, we train a decoder model on three languages: Python, Java, JavaScript. We use the model to sample and generate many versions of code and evaluate with the pass@k metric (one can think of it as accuracies given k chances). The result in Figure 1 shows that not only does it perform well on all languages that are trained on, the model also performs well on unseen languages (PHP, Ruby, Kotlin). How is this possible?


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/sampling-mbxp-4.svg"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure 1: pass@k scores (accuracy) versus sampling budget k" 
%}
</div>





## Natural Co-Occurrences of Multi-lingual Knowledge
It turns out that the natural occurrences of code data are quite common. Take the following code for example, which is a Python code that has JavaScript wrapped as a string.
This piece of data counts as Python data since it parses the Python interpreter, as well as being from a `.py` file. We refer to such multi-lingual occurrences of programming languages as **knowledge spillver**. Such spillover explains why training language models on Python yields a model that can write JavaScript.

The previous result shows the generalization of multi-lingual model trained on three languages. Mono-lingual models can also generalize.



<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/example-python-js-snippet.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure 2: JavaScript as a Python string representing cross-programming-language knowledge spillover."
%}
</div>


## Multi-ligual versus Mono-lingual


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/trend_vs_size_datasetall_mode-large_scale_temp0.6_passat10_grid0.svg"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure 3: pass@k scores (accuracy) versus model size" 
%}
</div>

Figure 2 represents the results including results comparing multi- and mono-lingual models. There are a lot going on, but let's break it down.

* We observe that the Python model (pink) has high accuracy in Java and JavaScript evaluation, which makes sense according to the hypothesis that models can pick up knowledge of other languages embedded in the primary language's code.
* The Java model (blue) and JavaScript model (green) seem to perform quite poorly on Python. We believe it is likely due to the lack of Python knowledge in Java/JavaScript data.
* In the multi-lingual model where we train on Python, Java, JS, we observe the Python performance being very similar to the mono-lingual Python performance. This seems to confirm the above point that there's little Java/JS knowledge in Python data, which means that in the multi-ligual case, the Python performance will be close to that of the mono-lingual Python model.

<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/data-spillover.svg"
  class="img-fluid rounded z-depth-1"
  padding="20px"
  caption="Figure 4: Different programming language's knowledge composition in each primary's language data due to the natural occurrence of data spillover." 
%}
</div>


* In Figure 3, we also observe that multi-lingual models perform especially better than mono-lingual models in out-of-domain languages.
* All these observations are consistent with the explanations in Figure 4 where the knowledge in other programming languages is aggregated across all knowledge in each language's training data.


## Large Multi-Lingual Models Really Shine
* As observed in Figure 3, one can see that if the model size is large enough, the advantages of multi-lingual training is more drastic.
* On out-of-domain evaluation, large multi-lingual models seem to break out of the log-linear trend, akin to being at a cusp of the sigmoid trend going upward, aka **emergent abilities**.


## Zero-Shot Translation
* We find that language models can also translate code, without being specifically trained to do so.
* This ability extends to a mono-lingual model. For instance, a Java model can translate from Python to Java reasonably well. 
* Java to Python is harder for translation with a Java model, since it doesn't know how to write Python well. However, it understands Python as some level and is able to use it to write a more accurate function.
* In fact, problems that are difficult can become much easier.

<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/translation-prompt-example.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure 4: Example of function completion with and without translation." 
%}
</div>


<div class="row mt-3">
<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/translation-from-python.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="(a) Evaluation results on translation, illustrating that with access to reference solutions, the model can generate more correct functions compared to baseline without translations (indicated by dots)" 
%}
</div>
<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/translation-error-analysis.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="(b) Tasks that are previously difficult (low solve rate for the baseline) can become easily solvable with translation.
For each task within MBXP (MBKP in this case), we show a fraction of generations that pass the tests over the total number of samples (solve rate), where the task indices are ranked to show increasing difficulty. 
The translation solve rate can be perfect (solve rate 1) for some tasks that originally have 0 solve rate." 
%}
</div>
</div>



## Few-Shot Prompts Helps LLMs on Out-of-Domain Languages

* On out-of-domain languages, the performance can be improved significantly if we give the model few-shot prompts.


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp/fewshot.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="(a) Few-shot prompting: Improvement on out-of-domain evaluation due to few-shot prompting, where the examples help guide the model to generate more correct code in the given language. 
(b) Few-shot prompts results in lower non-assertion (compile, parsing, syntax) errors on out-of-domain (ood) evaluation but has little effect on in-domain (id), consistent with the results in (a). " 
%}
</div>




<!--
### More Code Generation Abilities
Feel free to check out the paper on evaluation such as code-insertion, robustness, or code summarization.
-->


## Evaluation Datasets


All of the above analyses require evaluation datasets in different programming languages. In our work [Multilingual Evaluation of Code Generation Models](https://arxiv.org/pdf/2210.14868.pdf), we outlined how we obtain such datasets via transpiling the original HumanEval and MBPP into `HumanEvalX` and `MBXP`. We also compose such datasets for different types of evaluation such as Code Insertion evaluation or Code Robustness evaluation.


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp_methodology.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure : Evaluation Data Synthesis in 10+ Programming Languages." 
%}
</div>


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp_conversion_bold.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure : Example of Dataset Language Conversion from Python to Java." 
%}
</div>

## Appendix

### Codex Performance
It is unclear what data and how much the Codex models are trained on. However, a viable guess would be that they're trained on as much code data as possible with sufficient amount of steps until the performance plateaus.

Below, we show the result of `code-cushman-001` and `code-davinci-002` for reference. We can observe that the model performs quite well in all languages.

For the evaluation code, see (link to repo). 


<br>

**Table 1**: Codex Performance on MBXP and HumanEvalX with pass@1 and greedy decoding.

|                    |  **code-cushman-001**  |  **code-davinci-002**  |
|--------------------|:----------------------:|:----------------------:|
| **MBXP**           |
| Python             | <center>43.7%</center> |         58.7%          |
| Java               |         45.1%          | <center>61.0%</center> |
| JavaScript         |         46.4%          | <center>62.3%</center> |
| TypeScript         |         46.0%          |         58.9%          |
| C#                 |         46.2%          |         57.6%          |
| C++                |         49.3%          |         65.7%          |
| Go                 |         32.7%          |         49.2%          |
| Kotlin             |         44.6%          |         60.5%          |
| PHP                |         44.4%          |         60.7%          |
| Perl               |         34.1%          |         44.0%          |
| Ruby               |         43.7%          |         56.3%          |
| Scala              |         41.9%          |         59.8%          |
| Swift              |         31.3%          |         43.5%          |
| **HumanEvalX**     |
| Python             | <center>32.3%</center> |         46.3%          |
| Java               |         32.9%          | <center>49.1%</center> |
| JavaScript         |         28.0%          | <center>51.6%</center> |
| Typescript         |         34.8%          |         50.9%          |
| C#                 |         34.8%          |         45.3%          |
| C++                |                        |                        |
| Go                 |         16.3%          |         21.9%          |
| Kotlin             |         23.0%          |         39.8%          |
| PHP                |         31.1%          |         52.8%          |
| Perl               |         14.9%          |         36.0%          |
| Ruby               |         29.8%          |         39.8%          |
| Scala              |         24.2%          |         45.3%          |
| Swift              |         14.9%          |         24.8%          |









<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->
<!-- .......................................................................... -->


<!--
<d-code  language="python">
</d-code>

Why does highlight work for post but not for distill?
-->

<!--
{% highlight python %}
from org.jython.book.interfaces import BuildingType

class Building(BuildingType):
   def __init__(self, name, address, id):
      self.name = name
      self.address  =  address
      self.id = id

   def getBuildingName(self):
      return self.name

   def getBuildingAddress(self):
      return self.address

   def getBuldingId(self):
      return self.id

package org.jython.book.interfaces;

public interface BuildingType {

    public String getBuildingName();
    public String getBuildingAddress();
    public String getBuildingId();

}

package org.jython.book.util;

import org.jython.book.interfaces.BuildingType;
import org.python.core.PyObject;
import org.python.core.PyString;
import org.python.util.PythonInterpreter;

public class BuildingFactory {

    private PyObject buildingClass;

    public BuildingFactory() {
        PythonInterpreter interpreter = new PythonInterpreter();
        interpreter.exec("from Building import Building");
        buildingClass = interpreter.get("Building");
    }

    public BuildingType create(String name, String location, String id) {
        PyObject buildingObject = buildingClass.__call__(new PyString(name),
new PyString(location),
new PyString(id));
        return (BuildingType)buildingObject.__tojava__(BuildingType.class);
    }

}
{% endhighlight %}
-->

### Unabridged Example of Knowledge Spillover
Below we show a full code snippet of a Python file where JS code is wrapped in a string.

{% highlight python %}

"""Create a Javascript script to encode / decode for a specific encoding
described in a file available at
http://unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WINDOWS/<ENCODING>.TXT
"""

import os
import re
import json
import urllib.request

line_re = re.compile("^(0x[A-Z0-9]+)\s+(0x[A-Z0-9]+)*", re.M)

tmpl = "http://unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WINDOWS/{}.TXT"
encoding = input("Encoding name: ")
req = urllib.request.urlopen(tmpl.format(encoding.upper()))
data = req.read().decode("ascii")

root_dir = os.path.dirname(os.path.dirname(__file__))
libs_dir = os.path.join(root_dir, "www", "src", "libs")
filename = os.path.join(libs_dir, f"encoding_{encoding.lower()}.js")
with open(filename, "w", encoding="utf-8") as out:
    out.write("var _table = [")
    for line in data.split("\n"):
        mo = line_re.match(line)
        if mo:
            key, value = mo.groups()
            out.write(f"{key}, {value or -1},")
    out.write("]\n")
    out.write("var decoding_table = [],\n    encoding_table = []\n")
    out.write("""for(var i = 0, len = _table.length; i < len; i += 2){
var value = _table[i + 1]
if(value !== null){
    encoding_table[value] = _table[i]
}
decoding_table[_table[i]] = _table[i + 1]
}
$module = {encoding_table, decoding_table}
""")
{% endhighlight %}




