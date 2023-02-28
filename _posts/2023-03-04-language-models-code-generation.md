---
layout: distill
title: Unreasonable Effectiveness of LLMs for Code Generation 
date:   2023-02-04
description: 
tags: codegeneration
categories: transformers
published: true
social: true
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
#  - name: Overview
#  - subsections:
#    - name All-Reduce
#  - name: High-Level Illustration
#  - name: Attention Parallel
#  - subsections:
#    - name Output Parallel
#    - name Input Parallel
    
#  - name: MLP Parallel
#  #- subsections:
#    #  - name: Context Computation
#    #  - name: Incremental Decoding


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
However, it is still unclear how language models derive such amazing abilities especially in the area of code generation. This blog post outlines the experiments from [Multilingual Evaluation of Code Generation Models](https://arxiv.org/pdf/2210.14868.pdf) which give some clue as to how LLMs are so good at different code generation tasks.



## Out of Domain Generalization
TL;DR -- If we train a model on one programming language, it turns out that such a model can also write code in a different programming language, especially when the model is large enough!  Let's look at the results and sample generations. 






### How is this possible?
It turns out that the natural occurrences of code data are 



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









## Few-Shot Prompts Teach LLMs New Languages




## Zero-Shot Translation








## Evaluation Datasets


All of the above analyses require evaluation datasets in many different programming languages. In our work [Multilingual Evaluation of Code Generation Models](https://arxiv.org/pdf/2210.14868.pdf), we outlined how we obtain such datasets via transpiling the original HumanEval and MBPP into `HumanEvalX` and `MBXP`. We also compose such datasets for different types of evaluation such as Code Insertion evaluation or Code Robustness evaluation.


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp_methodology.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure 1: Evaluation Data Synthesis in 10+ Programming Languages." 
%}
</div>


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/img/blogs/mbxp_conversion_bold.png"
  class="img-fluid rounded z-depth-1"
  padding="0px"
  caption="Figure 2: Example of Dataset Language Conversion from Python to Java." 
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





