---
layout: distill
title: Helion Fusion in a Nutshell
date:   2023-04-07
description: A personal note on Helion's approach to fusion energy. 
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

### What is Fusion Energy

Fusion energy is the energy that powers the sun and the stars. The most common fusion reaction is the fusion of two hydrogen nuclei into a helium nucleus. The energy generated from the fusion procedure arises from the disparity in rest mass between the reactants involved in the fusion and the resulting product, as governed by the equation E=m.


### Fusion Energy on Earth by Helion

Helion Energy is pioneering advancements in nuclear fusion technology. Their current sixth-generation nuclear fusion generator uses magnetic fields to merge two plasma rings, transforming kinetic energy into thermal energy, heating the plasma to tens of millions of degrees, thus facilitating nuclear fusion. Unlike traditional methods, Helion employs a unique process that keeps the hot fuel off the walls and utilizes pulsating high-intensity magnetic fields. This technique results in a self-confined, self-organized plasma that moves like a piston when fusion begins, efficiently generating electricity. 


Moreover, Helion's approach harnesses a more abundant and safer fuel mixture of deuterium and helium-3. They've develoepd a method to produce the otherwise ultra-rare helium-3. The company's progress continues with the development of their Polaris, a seventh-generation system that's larger and designed to begin electricity capture. Helion's fusion technology is not only promising in terms of its efficiency but also offers hope for a cleaner and more sustainable energy future. The company's advancements have been documented in a video series titled "The Encore Tour." Despite financial challenges, the documentary was completed thanks to community support, and additional behind-the-scenes content is anticipated.

## Technical Details

Even if a fusion process yields high energy overall, the feasibility and efficiency of harnessing that energy largely hinge on the properties of the product particles. These particles are pivotal in determining the challenges and practicality associated with each fusion process.

### Helion: D + He3 Fusion

Helion uses a Deuterium and Helium 3 as fuel for the fusion process. This is different from other fusion process which uses Deuterium and Tritium such as in Tokemak. Let's compare the two.

(I) D + He3 -> p + He4 + 18.3 MeV

(II) D + T -> n + He3 + 17.6 MeV


- While process II produces similar amount of energy compared to process I, there are multiple challenges. (1) The neutron captures 80% of the released fusion energy. This is a problem for the reactor where neutrons are hard to contain (since it has no charge) and can damage the reactor at high energy. (2) Tritium (T) is quite rare. Producing it is challenging. (3) Capturing the energy from neutron and converting it to electricity requires a lot more steps's compared to Helion approach.

- Process I produces a proton (which is charged) whose energy can be captured to produce electricity directly. This is a big advantage for Helion's reactor. To make this possible, Helion developed a process to obtain He3 (which is much more rare compared to Dueterium).

- Caveat: D + He3 does require higher initial temperature which is a challenge. This is solved via a great deal of engineering, using capacitors to capture energy and releasing them in 100 micro seconds, producing 100,000 to 1M Amperes of current. 

### Producing He3

Helion has developed a process to produce He3, which relies on the Deuterium-Deuterium fusion. One of the possible outcomes of such fusion contains He3, that is,

(3) D + D -> He3 + n.

Doesn't this neutron be damaging to the reactor? The answer is that it contains much less energy (5 times less energy compared the neutron trom D+T process in II). The current rationale is that we can develope a reactor simply to generate He3 since if the damage happens, we only need to replace this reactor which generates fuel (He3), not the main energy generator (D+He3).
*** The generator can also capture of energy of He3 which produces a small amount of electricity (~2.45 MeV) in the process.



<!-- 
## How to make this fusion of D + He3 happen?
Helion uses a magnetic field to confine the plasma. The plasma is then compressed to a very high density. This is done by using a piston-like mechanism. The plasma is then heated to a very high temperature (100 million degrees) to initiate the fusion process. The plasma is then compressed further to produce more fusion. The energy released by the fusion process is then captured via changing magnetic fields which produce electricity.
-->


### Electricity Generation in Tokemak vs Helion

Tokemak captures the energy of neutron by slowing them down and generate heat. The heat is then used to create steam which rotates a turbine which then moves magnetic coils to generates electricity. Helion generates the electricity directly by capturing the energy of the proton via changing magnetic field!!

## The Future of Clean Energy
Overall I am super excited for fusion as the next generation of clean energy. Looking forward to see the progress on the 7th generation reactor (Polaris) in 2024!

References:

- https://www.youtube.com/watch?v=_bDXXWQxK38
- https://www.helionenergy.com/technology/




<!--
- Another possible process for D + D is D + D = p + T where T is kept for the decay process to produce He3. -- is this right?



- The neutron produced in the process can be used to produce He3 from Lithium.?????
- I don't think this is the case. Lithium is used to produce tritium in the tokemak layer instead.
The challenge is 

    - solved by Berillium, but it is expensive.




## Producing Tritium
Tritium is rare but there are some processes which can produce them such as n + Li -> He4 + T. However, it does not seem to be an energy efficient process. Berillium is used as a neutron multiplier but it is very expensive and is of very limited supply.
--> 