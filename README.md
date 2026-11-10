# SOM CIFAR-10: Neighborhood Cooperation and Prototype Initialization

**Study:** Quantization gains and reduced class agreement after self-organizing map initialization on CIFAR-10  
**Author:** Meshaal Mouawad  
**Journal:** Pattern Recognition Letters — Regular Article  
**Manuscript:** PRLETTERS-D-26-02894  
**Editorial milestone (10 November 2026):** Author-approved technical resubmission received by the Journal Office; awaiting forwarding to the Journal Editor for scientific assessment. **Peer review has not been confirmed.**

## Scientific findings

A controlled comparison of SOM neighborhood cooperation against winner-only initialization, random sampling, and greedy k-means++ used 320 trained prototype banks (10 paired draws × 2 starts × 2 training budgets × 2 feature representations × 4 initializers) followed by common Lloyd refinement.

All archived models were evaluated on the official 10,000-image CIFAR-10 test partition. Relative to winner-only learning, SOM reduced held-out quantization error by **1.3511%** (1,000-image RGB), **0.7954%** (1,000-image PCA-64), and **0.4174%** (5,000-image RGB); each comparison passed Holm correction across the twelve test contrasts. At the 1,000-image budget, adjusted Rand index was significantly lower under both representations; no normalized mutual information difference survived correction.

## Research files and publication record

- [Complete reproducible study, archived models, analyses, and figures](SOM_CIFAR10_Study_2026/)
- [Research methods and results](SOM_CIFAR10_Study_2026/RESULTS.md)
- [Publication and submission status](SOM_CIFAR10_Study_2026/PUBLICATION_STATUS.md)
- [Official-test evaluation analyses](SOM_CIFAR10_Study_2026/analysis/reserved_test_v1/)

The study's original archives and their SHA-256 manifest are retained unchanged. Signed forms, journal-system-generated submission PDFs, and editorial correspondence are maintained in the author's private submission archive, rather than in this public repository.

**Status terminology:** Journal Office processing is not acceptance or confirmation of external peer review.
