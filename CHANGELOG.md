# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions

### Fixed

- Reworked `mg.tl.cpu_gmm_probability` to score cells in batched NumPy blocks instead of per-cell multiprocessing tasks, which reduces Windows stalls caused by repeated process spawning and object serialization.
