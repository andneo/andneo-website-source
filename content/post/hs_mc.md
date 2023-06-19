---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Hard-Disk Monte Carlo"
subtitle: "2D Monte Carlo Simulation in Javascript"
summary: "Performing particle simulations in the browser sing javascript."
authors: []
tags: []
categories: []
date: 2019-06-02T12:53:27+01:00
lastmod: 2019-06-02T12:53:27+01:00
featured: false
draft: false

url_code: "https://editor.p5js.org/AndrewNeo/sketches/oo-NewVZl"

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
## Background Theory
\begin{equation} 
    \langle X\rangle = \dfrac{\int\text{d}q^{N}\exp\big[-\beta\mathcal{V}(q^{N})\big]X(q^{N})}{\int\text{d}q^{N}~\exp\big[-\beta\mathcal{V}(q^{N})\big]}
\end{equation}
### Metropolis Criterion
This can be alternatively written down as:
\begin{equation}
    \text{acc}(o\rightarrow n)=\min\bigg(1,\exp\bigg[-\beta\big(\mathcal{V}(n)-\mathcal{V}(o)\big)\bigg]\bigg)
\end{equation}
## The Code
We want to build a Monte Carlo package in javascript using object-oriented programming
### Setting up the System
```javascript
class System {
  // set initial system parameters
  constructor(){
    this.npart     = 100; // number of particles in the system
    this.density   = 0.4; // density of the system
    this.area      = (this.npart*20**2)/this.density; // area of the simulation box
    this.box_x     = Math.sqrt(this.area); // length of the box along the x-axis
    this.box_y     = this.box_x; // length of the box along the y-axis
    this.particles = []; // list of particles in the system
  }
}
```


## Final Result
{{< p5js oo-NewVZl 320 320 >}}