---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Clark Fischer
    equal-contrib: true
    affiliation: 1

  - name: Jason Chen
    equal-contrib: true 
    affiliation: 1

  - name: Jun Wang
    equal-contrib: true 
    corresponding: true 
    affiliation: 1
affiliations:
 - name: Multiplex Biotechnology Laboratory, Department of Biomedical Engineering, Stony Brook University, Stony Brook, New York 11794, United States
   index: 1

date: 13 Mar 2025
bibliography: paper.bib

---

# Summary

Using CycMIST decoding, a spatial proteomic method for  has enabled the simultaneous measurement of over 400 proteins at the single-cell level, providing an unprecedented resolution of immune cell heterogeneity and protein expression dynamics. However, the sheer scale and complexity of this data present significant challenges in storage, processing, and visualization. MISTBrowser addresses these challenges and speeds up 

# Statement of need

The nature of  

# Registration, Microbead Decoding, and Cell Segmentation

Our browser combines several common features 

`MISTBrowser` is an Python-based program for MIST-like data, designed to support the entire pipeline of processes needed to decode fom MIST microarrays. Our program includes registration, cell segmenantation (based on StarDist) protein decoding, mircobead segmenatation

designed to enable visualizations of protein distrobution over multiple cells, 


designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`MISTBrowser` combines the important features required for generating the spatial data needed for analysis. 

We include multiple features in accordance with the analysis requirements of MIST data.  registration pha

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References