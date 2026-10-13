# Generated results

This directory is intentionally excluded from Git except for this manifest.
Trained Julia models, likelihood matrices, generated FASTAs, and PDF figures
are large but reproducible from the tracked analysis scripts and final MSA.

Important local output directories include:

- `featuredca/jdomain_topt_ave_generation/`: paired temperature-conditioned generation.
- `featuredca/jdomain_pca2_generation/`: PCA-conditioned positive control.
- `featuredca/jdomain_temperature_ladder/`: high-temperature generation ladder.
- `featuredca/jdomain_natural_temperature_recovery/`: held-out natural recovery.
- `featuredca/jdomain_temperature_pca_recovery/`: temperature plus PC1/PC2 recovery.
- `featuredca/jdomain_temperature_pca_dimension_sweep/`: 2-64 PC sweep.
- `featuredca/jdomain_temperature_fixedbin_cv5/`: fixed-bin five-fold cross-validation.

The latest presentation is tracked at
`deliverables/jdomain_temperature_featuredca_results_v5.pptx`.
