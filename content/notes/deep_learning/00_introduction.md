---
title: Introduction
linktitle: Introduction
toc: true
type: book
date: "2023-06-23"
draft: false
toc: false
menu:
  deep_learning:
    weight: 1
summary: "An introduction to this resource and the motivation behind putting it together."
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 1
---

Lately it seems that everyone is interested in deep learning. 
There's a good reason why this is the case, this technology promises to solve a lot of problems -- from self-driving cars and diagnosing cancer to ... solving Atari games:
{{< youtube TmPfTpjtdgg >}}

There are a lot of cool things you can do these days with machine learning, but I think a lot of the machinery used to do it feel like black boxes. So to demystify the whole thing I decide to start building my own machine learning playground from the bottom-up and document it here. This is by no means meant to compete with the performance or applicability of professionally managed machine learning frameworks like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/). 

This whole thing is meant to be a learning experience so I'll do my best to write and explain everything as well as possible so that hopefully other people can find this to be a useful learning resource as well. Some of the key sources which I used are listed below:
- [Numerical Optimization](https://link.springer.com/book/10.1007/978-0-387-40065-5) by Nocedal and Wright. This textbook played a big role for me in my PhD too so holds a dear place in my heart. A pdf copy can be found [here](https://www.csie.ntu.edu.tw/~r97002/temp/num_optimization.pdf).
- [Numerical Recipes: The Art of Scientific Computing](http://numerical.recipes/)
- [Neural Networks for Pattern Recognition](https://global.oup.com/academic/product/neural-networks-for-pattern-recognition-9780198538646?cc=gb&lang=en&) by Christopher Bishop.
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) by Christopher Bishop. A pdf copy can be found [here](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Information Theory, Inference and Learning Algorithms](https://www.cambridge.org/0521642981) by David J. C. MacKay. A pdf copy can be found [here](https://www.inference.org.uk/itprnn/book.pdf)

{{< list_children >}}