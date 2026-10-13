#!/usr/bin/env python3
"""Prepare temperature-annotated Hsp70 and J-domain MSAs.

The pipeline is:
  1. Read TEMPURA temperature metadata.
  2. Keep UniProt FASTA records whose organism can be matched to TEMPURA.
  3. Use hmmsearch to select family/domain homologs.
  4. Use hmmalign to align the selected hits to the supplied profile HMM.
  5. Write aligned FASTA and per-sequence metadata CSV.

HMMER must be installed and available on PATH for the search/alignment steps.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


TEMPERATURE_COLUMNS = (
    "Tmin",
    "Topt_ave",
    "Topt_low",
    "Topt_high",
    "Tmax",
    "Tmax_Tmin",
)

UNIPROT_ACCESSION_RE = re.compile(r"^(?:sp|tr)\|([^|]+)\|")
OX_RE = re.compile(r"\bOX=(\d+)")
OS_RE = re.compile(r"\bOS=(.*?)(?:\s+OX=)")
GN_RE = re.compile(r"\bGN=([^\s]+)")
PAREN_RE = re.compile(r"\s*\([^)]*\)")


@dataclass(frozen=True)
class TempuraRecord:
    taxonomy_id: str
    genus_and_species: str
    strain: str
    superkingdom: str
    temperatures: dict[str, str]


@dataclass
class ProteinMetadata:
    original_id: str
    pipeline_id: str
    accession: str
    organism: str
    normalized_species: str
    taxonomy_id: str
    gene: str
    description: str
    tempura_match_mode: str
    tempura: TempuraRecord
    hmm_full_evalue: str = ""
    hmm_full_score: str = ""
    hmm_domain_evalue: str = ""
    hmm_domain_score: str = ""


def normalize_species_name(name: str) -> str:
    name = name.strip().strip('"')
    name = PAREN_RE.sub("", name).strip()
    parts = name.split()
    if len(parts) >= 3 and parts[0].lower() == "candidatus":
        return " ".join(parts[:3]).lower()
    if len(parts) >= 2 and parts[0].lower() not in {"uncultured", "environmental"}:
        return " ".join(parts[:2]).lower()
    return name.lower()


def read_tempura(path: Path) -> tuple[dict[str, TempuraRecord], dict[str, TempuraRecord]]:
    by_taxonomy: dict[str, TempuraRecord] = {}
    by_species: dict[str, TempuraRecord] = {}
    ambiguous_species: set[str] = set()

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            taxonomy_id = row["taxonomy_id"].strip()
            species = row["genus_and_species"].strip().strip('"')
            record = TempuraRecord(
                taxonomy_id=taxonomy_id,
                genus_and_species=species,
                strain=row.get("strain", "").strip(),
                superkingdom=row.get("superkingdom", "").strip(),
                temperatures={col: row.get(col, "").strip() for col in TEMPERATURE_COLUMNS},
            )
            if taxonomy_id:
                by_taxonomy.setdefault(taxonomy_id, record)

            key = normalize_species_name(species)
            if key in by_species and by_species[key].temperatures != record.temperatures:
                ambiguous_species.add(key)
            else:
                by_species[key] = record

    for key in ambiguous_species:
        by_species.pop(key, None)

    return by_taxonomy, by_species


def uniprot_accession(record_id: str) -> str:
    match = UNIPROT_ACCESSION_RE.search(record_id)
    return match.group(1) if match else record_id.split()[0]


def parse_field(pattern: re.Pattern[str], description: str) -> str:
    match = pattern.search(description)
    return match.group(1).strip() if match else ""


def organism_from_description(description: str) -> str:
    return parse_field(OS_RE, description)


def taxonomy_from_description(description: str) -> str:
    return parse_field(OX_RE, description)


def gene_from_description(description: str) -> str:
    return parse_field(GN_RE, description)


def metadata_for_record(
    record: SeqRecord,
    by_taxonomy: dict[str, TempuraRecord],
    by_species: dict[str, TempuraRecord],
    allow_species_fallback: bool,
) -> ProteinMetadata | None:
    description = record.description
    taxonomy_id = taxonomy_from_description(description)
    organism = organism_from_description(description)
    normalized_species = normalize_species_name(organism) if organism else ""

    if taxonomy_id in by_taxonomy:
        tempura = by_taxonomy[taxonomy_id]
        match_mode = "taxonomy_id"
    elif allow_species_fallback and normalized_species in by_species:
        tempura = by_species[normalized_species]
        match_mode = "species_name"
    else:
        return None

    return ProteinMetadata(
        original_id=record.id,
        pipeline_id="",
        accession=uniprot_accession(record.id),
        organism=organism,
        normalized_species=normalized_species,
        taxonomy_id=taxonomy_id,
        gene=gene_from_description(description),
        description=description,
        tempura_match_mode=match_mode,
        tempura=tempura,
    )


def matched_records(
    fasta: Path,
    by_taxonomy: dict[str, TempuraRecord],
    by_species: dict[str, TempuraRecord],
    allow_species_fallback: bool,
) -> Iterable[tuple[SeqRecord, ProteinMetadata]]:
    for index, record in enumerate(SeqIO.parse(fasta, "fasta"), start=1):
        meta = metadata_for_record(record, by_taxonomy, by_species, allow_species_fallback)
        if meta is None:
            continue
        meta.pipeline_id = f"seq{index:09d}|{meta.accession}|tax{meta.taxonomy_id or 'NA'}"
        yield record, meta


def write_matched_fasta(
    fasta: Path,
    output: Path,
    by_taxonomy: dict[str, TempuraRecord],
    by_species: dict[str, TempuraRecord],
    allow_species_fallback: bool,
) -> dict[str, ProteinMetadata]:
    metadata_by_id: dict[str, ProteinMetadata] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record, meta in matched_records(
            fasta, by_taxonomy, by_species, allow_species_fallback
        ):
            metadata_by_id[meta.pipeline_id] = meta
            out_record = SeqRecord(
                Seq(str(record.seq)),
                id=meta.pipeline_id,
                description=record.description,
            )
            SeqIO.write(out_record, handle, "fasta")
    return metadata_by_id


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required executable '{name}' was not found on PATH. "
            "Install HMMER or load the environment/module that provides it."
        )


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


def hmm_has_ga_threshold(hmm: Path) -> bool:
    with hmm.open() as handle:
        for line in handle:
            if line.startswith("GA "):
                return True
    return False


def run_hmmsearch(
    hmm: Path,
    fasta: Path,
    domtblout: Path,
    hits_fasta: Path,
    evalue: float,
    domain_evalue: float,
) -> set[str]:
    require_executable("hmmsearch")
    domtblout.parent.mkdir(parents=True, exist_ok=True)

    thresholds = (
        ["--cut_ga"]
        if hmm_has_ga_threshold(hmm)
        else ["-E", str(evalue), "--domE", str(domain_evalue)]
    )
    run_command(
        [
            "hmmsearch",
            *thresholds,
            "--domtblout",
            str(domtblout),
            "-o",
            str(domtblout.with_suffix(".log")),
            str(hmm),
            str(fasta),
        ]
    )

    hit_ids = parse_domtblout(domtblout)
    with hits_fasta.open("w") as handle:
        for record in SeqIO.parse(fasta, "fasta"):
            if record.id in hit_ids:
                SeqIO.write(record, handle, "fasta")
    return hit_ids


def parse_domtblout(path: Path) -> set[str]:
    hit_ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            hit_ids.add(parts[0])
    return hit_ids


def add_hmm_scores(metadata: dict[str, ProteinMetadata], domtblout: Path) -> None:
    best: dict[str, tuple[float, list[str]]] = {}
    with domtblout.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            target = parts[0]
            domain_score = float(parts[13])
            if target not in best or domain_score > best[target][0]:
                best[target] = (domain_score, parts)

    for target, (_, parts) in best.items():
        if target not in metadata:
            continue
        meta = metadata[target]
        meta.hmm_full_evalue = parts[6]
        meta.hmm_full_score = parts[7]
        meta.hmm_domain_evalue = parts[12]
        meta.hmm_domain_score = parts[13]


def run_hmmalign(hmm: Path, hits_fasta: Path, aligned_fasta: Path) -> None:
    require_executable("hmmalign")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_alignment = Path(tmpdir) / "aligned.a2m"
        run_command(
            [
                "hmmalign",
                "--outformat",
                "A2M",
                "--trim",
                "-o",
                str(tmp_alignment),
                str(hmm),
                str(hits_fasta),
            ]
        )

        aligned_fasta.parent.mkdir(parents=True, exist_ok=True)
        with aligned_fasta.open("w") as out_handle:
            for record in SeqIO.parse(tmp_alignment, "fasta"):
                aligned = "".join(base for base in str(record.seq) if not base.islower())
                aligned = aligned.replace(".", "")
                SeqIO.write(
                    SeqRecord(Seq(aligned), id=record.id, description=record.description),
                    out_handle,
                    "fasta",
                )


def gap_fraction(sequence: str) -> float:
    return sequence.count("-") / len(sequence) if sequence else 0.0


def write_metadata_csv(
    metadata: dict[str, ProteinMetadata],
    aligned_fasta: Path,
    output: Path,
) -> None:
    aligned_ids = []
    alignment_lengths: dict[str, int] = {}
    gap_fractions: dict[str, float] = {}
    for record in SeqIO.parse(aligned_fasta, "fasta"):
        aligned_ids.append(record.id)
        seq = str(record.seq)
        alignment_lengths[record.id] = len(seq)
        gap_fractions[record.id] = gap_fraction(seq)

    columns = [
        "sequence_id",
        "original_id",
        "accession",
        "taxonomy_id",
        "organism",
        "normalized_species",
        "gene",
        "tempura_match_mode",
        "tempura_taxonomy_id",
        "tempura_genus_and_species",
        "tempura_strain",
        "tempura_superkingdom",
        *TEMPERATURE_COLUMNS,
        "hmm_full_evalue",
        "hmm_full_score",
        "hmm_domain_evalue",
        "hmm_domain_score",
        "alignment_length",
        "gap_fraction",
        "description",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sequence_id in aligned_ids:
            meta = metadata[sequence_id]
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "original_id": meta.original_id,
                    "accession": meta.accession,
                    "taxonomy_id": meta.taxonomy_id,
                    "organism": meta.organism,
                    "normalized_species": meta.normalized_species,
                    "gene": meta.gene,
                    "tempura_match_mode": meta.tempura_match_mode,
                    "tempura_taxonomy_id": meta.tempura.taxonomy_id,
                    "tempura_genus_and_species": meta.tempura.genus_and_species,
                    "tempura_strain": meta.tempura.strain,
                    "tempura_superkingdom": meta.tempura.superkingdom,
                    **meta.tempura.temperatures,
                    "hmm_full_evalue": meta.hmm_full_evalue,
                    "hmm_full_score": meta.hmm_full_score,
                    "hmm_domain_evalue": meta.hmm_domain_evalue,
                    "hmm_domain_score": meta.hmm_domain_score,
                    "alignment_length": alignment_lengths[sequence_id],
                    "gap_fraction": f"{gap_fractions[sequence_id]:.6f}",
                    "description": meta.description,
                }
            )


def prepare_family(
    name: str,
    fasta: Path,
    hmm: Path,
    tempura: Path,
    output_dir: Path,
    allow_species_fallback: bool,
    skip_hmmer: bool,
    evalue: float,
    domain_evalue: float,
) -> None:
    by_taxonomy, by_species = read_tempura(tempura)
    family_dir = output_dir / name
    family_dir.mkdir(parents=True, exist_ok=True)

    matched_fasta = family_dir / f"{name}.tempura_matched.fasta"
    domtblout = family_dir / f"{name}.hmmsearch.domtblout"
    hits_fasta = family_dir / f"{name}.hmm_hits.fasta"
    aligned_fasta = family_dir / f"{name}.aligned.fasta"
    metadata_csv = family_dir / f"{name}.metadata.csv"

    if not skip_hmmer:
        require_executable("hmmsearch")
        require_executable("hmmalign")

    print(f"[{name}] Writing TEMPURA-matched FASTA...", file=sys.stderr)
    metadata = write_matched_fasta(
        fasta, matched_fasta, by_taxonomy, by_species, allow_species_fallback
    )
    print(f"[{name}] TEMPURA-matched records: {len(metadata)}", file=sys.stderr)

    if skip_hmmer:
        print(f"[{name}] Skipping HMMER steps by request.", file=sys.stderr)
        return

    print(f"[{name}] Searching profile HMM...", file=sys.stderr)
    hit_ids = run_hmmsearch(hmm, matched_fasta, domtblout, hits_fasta, evalue, domain_evalue)
    print(f"[{name}] HMM hits: {len(hit_ids)}", file=sys.stderr)
    add_hmm_scores(metadata, domtblout)

    print(f"[{name}] Aligning HMM hits...", file=sys.stderr)
    run_hmmalign(hmm, hits_fasta, aligned_fasta)

    print(f"[{name}] Writing metadata...", file=sys.stderr)
    write_metadata_csv(metadata, aligned_fasta, metadata_csv)
    print(f"[{name}] Done: {aligned_fasta} and {metadata_csv}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tempura", type=Path, default=Path("200617_TEMPURA.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--species-fallback",
        action="store_true",
        help="Also match UniProt OS species names to TEMPURA genus_and_species.",
    )
    parser.add_argument(
        "--skip-hmmer",
        action="store_true",
        help="Only write TEMPURA-matched FASTAs; do not run hmmsearch/hmmalign.",
    )
    parser.add_argument(
        "--family",
        choices=("hsp70", "jdomain", "all"),
        default="all",
        help="Which family to prepare.",
    )
    parser.add_argument(
        "--hsp70-fasta",
        type=Path,
        default=Path("full_hsp70.fasta"),
        help="Input UniProt FASTA for Hsp70 candidates.",
    )
    parser.add_argument(
        "--jdomain-fasta",
        type=Path,
        default=Path("full_jdomain.fasta"),
        help="Input UniProt FASTA for J-domain candidates.",
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=1e-5,
        help="Sequence E-value threshold for HMMs without GA thresholds.",
    )
    parser.add_argument(
        "--domain-evalue",
        type=float,
        default=1e-5,
        help="Domain E-value threshold for HMMs without GA thresholds.",
    )
    args = parser.parse_args()

    families = {
        "hsp70": {
            "fasta": args.hsp70_fasta,
            "hmm": Path("PF00012.hmm"),
        },
        "jdomain": {
            "fasta": args.jdomain_fasta,
            "hmm": Path("dnaj_hscb_djlc_aligned_filtered.hmm"),
        },
    }

    selected = families if args.family == "all" else {args.family: families[args.family]}
    for name, config in selected.items():
        prepare_family(
            name=name,
            fasta=config["fasta"],
            hmm=config["hmm"],
            tempura=args.tempura,
            output_dir=args.output_dir,
            allow_species_fallback=args.species_fallback,
            skip_hmmer=args.skip_hmmer,
            evalue=args.evalue,
            domain_evalue=args.domain_evalue,
        )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
