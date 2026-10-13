#!/usr/bin/env python3
"""Filter an aligned FASTA and metadata CSV by per-sequence gap fraction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from Bio import SeqIO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-fasta", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--max-gap-fraction", type=float, default=0.5)
    args = parser.parse_args()

    with args.metadata_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys() if rows else []

    keep_ids = {
        row["sequence_id"]
        for row in rows
        if float(row["gap_fraction"]) <= args.max_gap_fraction
    }

    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    with args.output_fasta.open("w") as handle:
        for record in SeqIO.parse(args.aligned_fasta, "fasta"):
            if record.id in keep_ids:
                SeqIO.write(record, handle, "fasta")

    with args.output_metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if row["sequence_id"] in keep_ids)

    print(f"Kept {len(keep_ids)} of {len(rows)} records.")


if __name__ == "__main__":
    main()
