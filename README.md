<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/tycho_transfer_named_horizontal_dark.svg">
    <img alt="Tycho" src="docs/source/_static/tycho_transfer_named_horizontal_light.svg" width="380">
  </picture>
</p>

<p align="center">
  <a href="https://granthecht.github.io/tycho/"><strong>📖 Documentation</strong></a>
  &nbsp;·&nbsp;
  <a href="https://granthecht.github.io/tycho/getting_started/">Getting Started</a>
  &nbsp;·&nbsp;
  <a href="https://granthecht.github.io/tycho/getting_started/install.html">Install &amp; Build</a>
</p>

Tycho is a modular, extensible library for trajectory design and optimal control.
It uses a custom implementation of vector math formalisms to enable rapid 
implementation of dynamical systems and automatic differentiation. The phase object 
is the core of the optimal control functionality, and by linking multiple phases 
together, the user can construct scenarios of arbitrary complexity. A high performance 
interior-point optimizer (PSIOPT) is included with the library, which enables quick 
turnaround from concept to solution.

## Origin

Tycho is an independently maintained fork of 
[ASSET (Astrodynamics Software and Science Enabling Toolkit)](https://github.com/AlabamaASRL/asset_asrl), 
originally developed by the Astrodynamics and Space Research Laboratory at the 
University of Alabama. Full credit and thanks are owed to the original authors 
for building the foundation this project is built upon.

Original ASSET development was funded by NASA under Grant No. 80NSSC19K1643.

## Installation & Building

Tycho is a C++20 / Python library built with CMake and nanobind. The build
produces a native extension module (`_tychopy`) and installs it alongside the
pure-Python `tychopy` package into your active Python environment.

Full, per-platform install and build instructions (macOS, Linux/WSL2, Windows),
prerequisites, CMake options, and build verification live in the documentation:

- **[Installing Tycho](https://granthecht.github.io/tycho/getting_started/install.html)** — step-by-step setup and build for each platform
- **[Getting Started](https://granthecht.github.io/tycho/getting_started/)** — first steps once it's built

## Documentation

The full documentation — guides, tutorials, and the API reference — is hosted at
**[granthecht.github.io/tycho](https://granthecht.github.io/tycho/)**.

## Citation

If you use Tycho in a published work, please cite the original ASSET paper that 
this project is based upon:
```bibtex
@inproceedings{Pezent2022,
        author = {James B. Pezent and Jared Sikes and William Ledbetter and Rohan Sood and Kathleen C. Howell and Jeffrey R. Stuart},
        title = {ASSET: Astrodynamics Software and Science Enabling Toolkit},
        booktitle = {AIAA SCITECH 2022 Forum},
        pages = {AIAA 2022-1131},
        year={2022},
        doi = {10.2514/6.2022-1131}
}
```