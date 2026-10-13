# Data layout

The repository tracks the compact inputs needed by the J-domain analyses:

- `../200617_TEMPURA.csv`: organism-level growth temperatures from TEMPURA.
- `../new_full_jdom.fasta`: corrected UniProt J-domain candidate export.
- `../new_full_hsp70.fasta`: corrected UniProt Hsp70 candidate export.
- `../dnaj_hscb_djlc_aligned_filtered.hmm`: J-domain profile HMM.
- `../PF00012.hmm`: Hsp70 profile HMM.
- `processed_new/jdomain/jdomain.aligned.maxgap50.fasta`: final 62-position J-domain MSA.
- `processed_new/jdomain/jdomain.metadata.maxgap50.csv`: metadata aligned row-for-row with the final MSA.

HMMER search logs, hit FASTAs, unfiltered alignments, and the older
`data/processed/` tree are intentionally ignored because they are large and
reproducible. Recreate them with `prepare_temperature_datasets.py`.

The final J-domain cohort contains 14,944 proteins from 4,855 organisms. All
homologs are retained. Every sequence has `Tmin`, `Topt_ave`, `Tmax`, and
`Tmax_Tmin`; `Topt_low` and `Topt_high` are present only when TEMPURA reports an
optimal-temperature interval.
