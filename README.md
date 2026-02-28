# Tycho

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

## Download

PyPI packaging is coming soon. In the meantime, please build from source.

## Documentation

Documentation is coming soon.

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