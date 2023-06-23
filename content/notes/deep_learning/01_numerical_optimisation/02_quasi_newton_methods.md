---
title: Quasi-Newton Methods
linktitle: Quasi-Newton Methods
toc: true
type: book
date: "2019-05-05T00:00:00+01:00"
draft: false
url_code: "https://github.com/andneo/andneo_code/tree/master/deep_learning/optimisation_algorithms"
menu:
  deep_learning:
    parent: Optimising Functions
    weight: 4
summary: An overview of the BFGS and L-BFGS minimisation algorithms along with inexact line search algorithms, and their implementation.
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 4
---

Optimisation is a ubiquitous process, people do it everyday. A classic example would be: _How do I get somewhere as (**fast, cheap, environmentally friendly**) as possible?_  Nature optimises too, a big part of what I did in my PhD was finding out how systems of particles optimise their potential energy. 
Needless to say, optimisation is an important process, and quite a bit of effort has been dedicated to developing algorithms that do it well.
Optimisation algorithms form a key component of neural networks, as we will see. 