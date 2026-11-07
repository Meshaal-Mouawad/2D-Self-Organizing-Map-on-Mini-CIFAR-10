# SOM neighborhood cooperation on Mini CIFAR-10 — confirmatory evidence (2026)

**Author:** Meshaal Mouawad  
**Status:** 320/320 confirmatory runs complete; development-set statistical analysis complete.  
**Primary question:** Does neighborhood cooperation in a 4 × 4 SOM warm start affect the quantization and semantic alignment of 16 prototypes after identical Lloyd refinement?

- **Study report with complete methods and statistics:** [`RESULTS.md`](RESULTS.md)
- **Experiment settings frozen before the 320 confirmatory runs:** [`design/protocol_v1/confirmatory_protocol.json`](design/protocol_v1/confirmatory_protocol.json)
- **Split indices / audit:** [`design/splits_v1/indices.npz`](design/splits_v1/indices.npz)
- **Controller:** [`som_confirmatory_controller_v1.py`](som_confirmatory_controller_v1.py)
- **Original code dependencies:** [`som_four_arm_development_v1.py`](som_four_arm_development_v1.py)
- **320 arm checkpoints and sidecars:** [`runs/confirmatory_v1/arms/`](runs/confirmatory_v1/arms/)
- **Results:** [`runs/confirmatory_v1/results.csv`](runs/confirmatory_v1/results.csv)
- **Paired inferential results:** [`analysis/confirmatory_v1/paired_comparisons.csv`](analysis/confirmatory_v1/paired_comparisons.csv)
- **Holm-adjusted primary comparisons:** [`analysis/confirmatory_v1/primary_holm_corrected.csv`](analysis/confirmatory_v1/primary_holm_corrected.csv)
- **Vector figures:** [`figures/`](figures/)
- **File-level integrity manifest:** [`MANIFEST_SHA256.txt`](MANIFEST_SHA256.txt)

## Reproduction

Use Python 3.12 with `requirements-frozen.txt`. Obtain the official CIFAR-10 Python version from https://www.cs.toronto.edu/~kriz/cifar.html (archive MD5: `c58f30108f718f92721af3b95e74349a`). The archive is intentionally omitted from GitHub. The controller resolves `data/`, `design/` and `runs/` relative to its own `.py` file. To reproduce from scratch, create a **separate clean working directory**, copy the archived `som_*.py` and the complete `design/` directory into it, extract CIFAR-10 to `<clean-workspace>/data/raw/cifar-10-batches-py/`, then execute `python som_confirmatory_controller_v1.py --run --all` there. The archived `runs/confirmatory_v1/` contains verified completed checkpoints for auditing and should remain unchanged.

The public repository's existing `Code/` MATLAB PCA material is historical work and remains preserved outside this new study directory.
