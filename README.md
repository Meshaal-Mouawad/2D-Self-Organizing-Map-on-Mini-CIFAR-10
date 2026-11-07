# SOM CIFAR-10: Neighborhood Cooperation and Prototype Initialization

This repository contains the 2026 experimental study of self-organizing map (SOM) neighborhood cooperation as a prototype initialization strategy for CIFAR-10 clustering.

The full 320-run confirmatory study, frozen protocols, model artifacts, statistical analyses, figures, and reproducibility records are available in [SOM_CIFAR10_Study_2026](SOM_CIFAR10_Study_2026/).

The primary comparison is SOM versus winner-only initialization under paired data draws and shared refinement. The development-set analysis finds significant quantization-error reductions in three of four configurations after Holm correction; no ARI or NMI differences survive correction.

See [RESULTS.md](SOM_CIFAR10_Study_2026/RESULTS.md) for complete results and the [study README](SOM_CIFAR10_Study_2026/README.md) for the evidence package.
