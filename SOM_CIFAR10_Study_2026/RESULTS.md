# Confirmatory results: SOM neighborhood-cooperation study

## Research question
Does the 4 × 4 SOM neighborhood update during prototype initialization alter later 16-cluster quantization and correspondence with CIFAR-10 class labels after identical Lloyd refinement?

## Dataset and experimental design
- CIFAR-10: 50,000 official training images, 10,000 reserved test images, 32 × 32 RGB, 10 labeled classes.
- Fixed stratified development partition: 5,000 images (500/class). Training pool: 45,000 (4,500/class).
- Ten stratified, paired training draws: nested 1,000 images (100/class) and 5,000 images (500/class).
- Two starts per draw; deterministic seed = 20261107 + draw × 100 + start.
- Representations: mean-centered RGB pixels in float32 / 255, and PCA-64 fitted to the training subset using randomized SVD.
- Methods: 4 × 4 SOM neighborhood update; winner-only update; 16 sampled prototypes; k-means++.
- SOM and winner-only use the same prototype bank and sample order within each paired start.
- SOM and winner-only: five training epochs, exponentially decaying learning rate 0.5 → 0.05; SOM neighborhood width 2 → 0.5.
- Common downstream Lloyd refinement: max 300 iterations, relative prototype-shift tolerance 1e-6; deterministic farthest-residual empty-cluster reseeding.
- 2 budgets × 2 representations × 4 initializers × 10 draws × 2 starts = 320 model runs.
- Evaluation on fixed 5,000-image development partition: quantization mean squared Euclidean distance, ARI, NMI, occupied prototype count, and timing/iterations.
- Inferential unit: the ten draws, after averaging the two starts within each draw; paired two-sided t-based 95% intervals.
- 12 designated primary inferential contrasts (SOM vs winner-only; three endpoints × four settings), Holm familywise correction at alpha=0.05. This inferential family was formalized during post-execution analysis.

## Descriptive results
All entries below are means over 20 runs per budget × representation × method (ten draws, two starts). Standard deviations are shown in the supplementary CSV.

| Training N | Representation | Initializer | ARI | NMI | Quantization MSE |
|---:|---|---|---:|---:|---:|
| 1000 | centered_rgb | kmeans++ | 0.03862 | 0.08950 | 118.62317 |
| 1000 | centered_rgb | random | 0.03899 | 0.08834 | 119.19399 |
| 1000 | centered_rgb | som | 0.03854 | 0.08960 | 117.92796 |
| 1000 | centered_rgb | winner | 0.04079 | 0.08840 | 119.70681 |
| 1000 | pca64 | kmeans++ | 0.03933 | 0.09190 | 89.63709 |
| 1000 | pca64 | random | 0.03969 | 0.09066 | 90.05572 |
| 1000 | pca64 | som | 0.03881 | 0.09010 | 89.30444 |
| 1000 | pca64 | winner | 0.04066 | 0.09114 | 90.14517 |
| 5000 | centered_rgb | kmeans++ | 0.04052 | 0.09455 | 115.68973 |
| 5000 | centered_rgb | random | 0.04036 | 0.09304 | 115.77821 |
| 5000 | centered_rgb | som | 0.04081 | 0.09453 | 115.57967 |
| 5000 | centered_rgb | winner | 0.04079 | 0.09346 | 116.02967 |
| 5000 | pca64 | kmeans++ | 0.04058 | 0.09425 | 89.35978 |
| 5000 | pca64 | random | 0.04068 | 0.09427 | 89.26758 |
| 5000 | pca64 | som | 0.04033 | 0.09310 | 89.22780 |
| 5000 | pca64 | winner | 0.04137 | 0.09563 | 89.13294 |

## Primary SOM-minus-winner-only paired effects
Negative MSE favors SOM; positive ARI/NMI favors SOM. The shown 95% confidence intervals are unadjusted; Holm-corrected p-values determine confirmatory significance.

| Training N | Representation | Endpoint | Mean difference | Unadjusted 95% CI | Holm p | Significant? |
|---:|---|---|---:|---|---:|---|
| 1000 | centered_rgb | ARI | -0.002246 | [-0.003696, -0.000796] | 0.059295 | False |
| 1000 | centered_rgb | Quantization MSE | -1.778845 | [-2.081197, -1.476494] | 0.000004 | True |
| 1000 | centered_rgb | NMI | +0.001195 | [-0.000340, +0.002731] | 0.448604 | False |
| 1000 | pca64 | ARI | -0.001849 | [-0.003040, -0.000659] | 0.059295 | False |
| 1000 | pca64 | Quantization MSE | -0.840733 | [-1.132463, -0.549003] | 0.001090 | True |
| 1000 | pca64 | NMI | -0.001043 | [-0.003224, +0.001138] | 0.712862 | False |
| 5000 | centered_rgb | ARI | +0.000024 | [-0.001051, +0.001100] | 0.960377 | False |
| 5000 | centered_rgb | Quantization MSE | -0.449998 | [-0.573897, -0.326100] | 0.000197 | True |
| 5000 | centered_rgb | NMI | +0.001074 | [-0.000846, +0.002993] | 0.712862 | False |
| 5000 | pca64 | ARI | -0.001034 | [-0.001968, -0.000100] | 0.201903 | False |
| 5000 | pca64 | Quantization MSE | +0.094862 | [-0.016730, +0.206454] | 0.433193 | False |
| 5000 | pca64 | NMI | -0.002528 | [-0.004569, -0.000487] | 0.144641 | False |

## Relative quantization differences
Each percentage uses the mean winner-only MSE from the same representation and budget as its denominator.

- 1000 images, centered_rgb: SOM-minus-winner-only MSE -1.4860%.
- 1000 images, pca64: SOM-minus-winner-only MSE -0.9326%.
- 5000 images, centered_rgb: SOM-minus-winner-only MSE -0.3878%.
- 5000 images, pca64: SOM-minus-winner-only MSE +0.1064%.

## Interpretation
Under the primary Holm family, SOM decreases quantization error significantly in the 1,000-image RGB, 1,000-image PCA-64, and 5,000-image RGB settings. The 5,000-image PCA-64 quantization contrast is not significant. No ARI or NMI contrast survives Holm correction. Accordingly, the evidence supports an objective-specific quantization benefit, not improved class-label alignment.

## Supplementary records
- `analysis/confirmatory_v1/descriptive_summary.csv`: all descriptive mean/SD estimates.
- `analysis/confirmatory_v1/paired_comparisons.csv`: 36 exploratory/primary contrasts, with paired draw-level confidence intervals and unadjusted p-values.
- `analysis/confirmatory_v1/primary_holm_corrected.csv`: 12 Holm-adjusted primary contrasts.
- `runs/confirmatory_v1/results.csv`: all 320 outcomes, times, occupancy, and Lloyd iterations.
- `runs/confirmatory_v1/arms/`: 320 serialized checkpoints plus 320 JSON metadata records with SHA256.
- `runs/confirmatory_v1/events.jsonl`: append-only completion record.

## Remaining analyses and scope
- The ten-thousand-image CIFAR-10 test partition remains unused and reserved for final independent evaluation.
- Direct SOM topology-preservation measurements and full initialization-time accounting are not yet reported.
- RGB and PCA-64 quantization MSE values have different feature spaces; interpret MSE differences within the same representation.
- PCA time was recorded per run in the controller, with cached PCA reused within groups; do not sum those fields as independent fit times.
- Two starts within each draw are technical repeats, not 20 independent training-subset draws.
- The 12-test primary Holm family was designated during analysis, rather than separately preregistered before model execution.
