# TemperatureDCA

TemperatureDCA studies how J-domain and Hsp70 protein-family sequence
distributions relate to the optimal growth temperature of their source
organisms. The current analyses focus on a 62-position J-domain multiple
sequence alignment and use the local Julia package
[FeatureDCA.jl](../FeatureDCA.jl) for conditional autoregressive modeling.

## Current dataset

The corrected UniProt exports are `new_full_jdom.fasta` and
`new_full_hsp70.fasta`. Candidate proteins are matched to TEMPURA by exact
UniProt `OX` taxonomy ID, filtered with family profile HMMs, aligned with
`hmmalign --trim`, and filtered at a maximum gap fraction of 0.5.

The final J-domain cohort is tracked in:

- `data/processed_new/jdomain/jdomain.aligned.maxgap50.fasta`
- `data/processed_new/jdomain/jdomain.metadata.maxgap50.csv`

It contains 14,944 proteins, 4,855 organisms, 62 alignment positions, and all
available TEMPURA temperature fields. Multiple homologs from the same organism
are retained.

## Repository layout

```text
analysis/                  Julia and Python analysis workflows
data/                      Final tracked cohort and data manifest
deliverables/              Latest curated presentation
results/                   Local generated artifacts (ignored by Git)
200617_TEMPURA.csv         Organism-level temperature metadata
new_full_*.fasta           Corrected UniProt candidate exports
*.hmm                      Protein-family profile HMMs
prepare_temperature_datasets.py
Project.toml / Manifest.toml
```

Large obsolete FASTAs, HMMER intermediates, trained models, generated
sequences, likelihood matrices, caches, and presentation build intermediates
remain available locally but are excluded from version control.

## Requirements

- Python 3 with packages in `requirements.txt`
- HMMER (`hmmsearch` and `hmmalign` on `PATH`)
- Julia 1.10
- A sibling checkout of `FeatureDCA.jl` at `../FeatureDCA.jl`

Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Julia setup with the local FeatureDCA checkout:

```bash
julia --project=. -e 'using Pkg; Pkg.develop(path="../FeatureDCA.jl"); Pkg.instantiate()'
```

## Dataset preparation

Prepare both corrected families with exact taxonomy matching:

```bash
python3 prepare_temperature_datasets.py \
  --hsp70-fasta new_full_hsp70.fasta \
  --jdomain-fasta new_full_jdom.fasta
```

The J-domain HMM lacks a curated GA cutoff, so the pipeline uses sequence and
domain E-value thresholds of `1e-5`. The Hsp70 PF00012 model uses its curated
GA threshold. See `data/README.md` for tracked and regenerated files.

## Core FeatureDCA analyses

Train the temperature-conditioned J-domain model:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/train_jdomain_featuredca.jl
```

Run paired generation and sequence-statistics validation:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/evaluate_jdomain_generation.jl
```

Run the PCA-conditioned positive control:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/train_jdomain_pca_featuredca.jl
JULIA_NUM_THREADS=auto julia --project=. analysis/evaluate_jdomain_pca_generation.jl
```

Generate and analyze the high-temperature ladder:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/generate_temperature_ladder.jl
julia --project=. analysis/analyze_temperature_ladder_diversity.jl
```

## Held-out temperature recovery

The original organism-separated train/test experiment is:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/evaluate_natural_temperature_recovery.jl --quiet
```

The preferred validation is fixed-temperature-bin, organism-separated,
five-fold cross-validation. Every organism appears in a test fold exactly once,
and every fold contains all eight temperature bins:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/crossvalidate_fixed_temperature_bins.jl
```

This workflow reports both sequence-level recovery and organism-level recovery
after averaging homolog likelihood profiles. It evaluates both 0.5 degree and
10 degree temperature grids and writes pooled and macro-averaged metrics.

## Temperature plus PCA conditioning

Train and evaluate a combined `(Topt_ave, PC1, PC2)` feature model:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/run_temperature_pca_recovery.jl
```

Sweep over 2, 4, 8, 16, 32, and 64 PCA components:

```bash
JULIA_NUM_THREADS=auto julia --project=. analysis/sweep_temperature_pca_dimensions.jl
```

PCA for held-out experiments is fitted only on the training alignment. The
full-data generation experiment evaluates temperature recovery both with the
original PC coordinates used for generation and with PCs projected from each
generated sequence.

## Reproducibility notes

- Temperature is standardized using training data only in held-out analyses.
- Taxonomy IDs, rather than individual proteins, define train/test boundaries.
- Higher conditional log-likelihood is better, even though values are negative.
- Per-site likelihood plots divide sequence log-likelihood by the 62 alignment positions.
- Generated artifacts are checkpointed under `results/featuredca/` and can be
  regenerated without changing tracked source files.
