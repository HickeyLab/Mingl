# MINGL

Multicellular Neighborhood Analysis using Gaussian Mixture Models to create a probabilistic distribution for classification.

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/jwrightd/MINGLE/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/MINGLE

## Getting started

MINGL is a set of tools and plotting functions for identifying and quantifying borders between hierarchical units and gradients of changing cellular organization across these interfaces. MINGL is also a tool for investigating heterogeneity in hierarchical tissue organization across disease states, between patients, or across tissue samples from the same patient, and can identify changes in cellular organization even when anchor cell types remain unchanged.
MINGL also includes a tool for suggesting a biologically-informed cluster number range as a starting point for hierarchical spatial organization analysis.

MINGL's main tool can be run on MacOS or WindowsOS using CPU, and is also equipped with a GPU accelerated version compatible with cupy and WindowsOS as of version 0.0.1.

Please see instructions on installation and our recommended use below. Happy exploration of "life on the edge" in borders between spatial organization of our tissues!

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

Python 3.11 or newer is required for installation on your system.

We recommend that you install MINGL into a new, fresh environment to avoid any dependency conflicts. First, create your new environment using Python version 3.11 and follow either of the installation methods below.

There are two options to install MINGL:

<!--
1) Install the latest release of `MINGL` from [PyPI][]:

```bash
pip install MINGLE
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/HickeyLab/Mingl.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/jwrightd/MINGLE/issues
[tests]: https://github.com/jwrightd/MINGLE/actions/workflows/test.yaml
[documentation]: https://MINGLE.readthedocs.io
[changelog]: https://MINGLE.readthedocs.io/en/latest/changelog.html
[api documentation]: https://MINGLE.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/MINGLE
