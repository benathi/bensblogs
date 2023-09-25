---
layout: distill
title: Magnetism from Relativistic Electricity
date:   2023-04-20
description: A sketch of how magnetism arises from special relativity.
tags: physics
# categories: 
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

## Introduction

In classical physics, electric and magnetic fields are seen as separate entities. That is, electric fields are present whenever there are net charges. However, once these charges start to move, we can measure another kind of force that the electric field does not account for. As illustrated in Figure 1, on the left, the charge $$q$$ results in a force $$\vec{F}_E$$ on a unit charge, therefore in this frame we say that there is an electric field $$\vec{E} = \vec{F}_E$$. Once this charge starts to move, the force on the unit charge becomes different when the unit charge starts to move! Classically, we call this force the magnetic force (and the force per unit the magnetic field).


In this blog, we will show that this magnetic field is a relativistic effect of the electric field.


<div class="col-sm mt-3 mt-md-0">
{% include figure.html
  path="assets/pdf/magnetism_wire.svg"
  class="img-fluid rounded z-depth-1"
  padding="10px"
    caption="Figure 1: Magnetic field can be seen as a relativistic effect of the electric field."
%}
</div>



### Magnetism from Current in a Long Wire

First, we show a special case where we have an infinitely long wire with constant current $$I$$. For simplicity, we model the current as moving positive charges with some drift velocity.

In general, the electric field produced from a long wire with charge density $$\rho$$ at a distance $$d$$ is given by

$$
\vec{E} = \frac{\rho}{2\pi\epsilon_0 d} \hat{d}
$$

where $$\hat{d}$$ is the unit vector pointing from the wire to the point of interest. 

In the frame of reference where the test charge is moving at speed $$v$$, the wire is neutral, so there is no electric force acting on the test charge.

In the second frame of reference where the test charge is at rest, we observe the positive charges at a drift velocity $$\vec{v}_+ = \vec{v}_d - \vec{v}$$ and the negative charges at a drift velocity $$\vec{v}_- =  \vec{v}$$. The net electric force at the test charge in this frame of reference is no longer zero due to special relativity.

For an object moving at speed $$\vec{w}$$, we observe length contraction along the direction of motion. Specifically, if the length at rest is $$L_0$$, the length in the moving frame of reference becomes

$$
L = \frac{L_0}{\gamma} < L_0
$$

where $$\gamma = \frac{1}{\sqrt{1 - \frac{w^2}{c^2}}}$$.
With length contraction, the electric field becomes different due to the different charge density, or charge per unit length. Specifically, the positive charge density becomes

$$
\rho_+ = \rho \gamma_+ \text{ and } \rho_- = \rho \gamma_-
$$

Therefore, the net electric force on the test charge due to the contracted positive and negative charges become

$$
\vec{F}_+ + \vec{F}_- = \frac{\rho}{2\pi\epsilon_0 d} \hat{d} \cdot \left( \gamma_+ - \gamma_- \right)
$$

where $$\gamma_+ = \frac{1}{\sqrt{1 - \frac{v_+^2}{c^2}}}$$ and $$\gamma_- = \frac{1}{\sqrt{1 - \frac{v_-^2}{c^2}}}$$. 

Now, we can expand $$\gamma_+$$ and $$\gamma_-$$ via Taylor series to obtain

$$
\gamma_+ - \gamma_-
= \left( 1 - \frac{(v-v_d)}{c^2} \right)^{-1/2} -  \left( 1 - \frac{v^2}{c^2} \right)^{-1/2} 
$$
which becomes 
$$
\gamma_+ - \gamma_- \approx 1 + \frac{1}{2} \frac{(v_d - v)^2}{c^2} - 1 - \frac{1}{2} \frac{v^2}{c^2} \approx - \frac{v_d v}{c^2}
$$
where we assume that the drift velocity $$ v_d << v $$ so that $$v_d^2 << v_d v$$ and can be ignored. In fact, we made this assumption in the first frame of reference where we do not take the length contraction of the positive charges into account (since it is up to the squared term $$v_d^2/c^2$$).


Now, the net force on the test charge can be written as

$$
\vec{F}_+ + \vec{F}_- = - \frac{\rho v_d v}{2\pi\epsilon_0 c^2 d} \hat{d} = -\frac{\mu_0 I}{2 \pi d} \hat{d} \cdot v
$$

where (1) $$v_d \rho = I$$ and (2) $$c^2 = \frac{1}{\mu_0 \epsilon_0}$$ and $$\mu_0$$ is the permeability of free space. 


In the original frame of reference, this "mysterious" is considered as coming from magnetic force, where the magnetic field is given by $B = \frac{\mu_0 I}{2 \pi d} \hat{z}$ where $z$ is the direction going into the screen (derived by the right hand rule with respect to the current direction going to the right). Then the magnetic force on a moving test charge with velocity $$\vec{v}$$ is given y $$ \vec{F}_B = \vec{v} \times \vec{B}$$, which is precisely $$\vec{F}_+ + \vec{F}_-$$ is the test charge's frame of reference! 


#### Notes
- The electric field in a long wire can be derived by integrating the electric field from the collection of infinitesimal charges along the wire and Coulomb's law. We omit the derivation here.
- Length contraction can be derived via considering a moving laser pointer and a mirror ceiling and the fact that the speed of light in any frame of reference is $$c$$.
- We can see that up to this point, we pretty much derive the magnetic field from first principles of electric fields and special relativity. Had special relativity be discovered first, this would perhaps be the standard way to describe the magnetic field!
- It is amazing that even though the length contraction is imperceptibly small, its collective effect can produce something noticeable like the magnetic field.


<!--
### Deriving Biot-Savart Law

Now, we will derive the Biot-Savart law, which is the generalization of the magnetic field from a long wire to any current distribution. 

-->