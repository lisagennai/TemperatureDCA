#!/usr/bin/env python3
"""Project temperature-annotated MSAs into one-hot PCA space.

The script analyzes each protein family independently. It reads the max-gap
filtered alignment and metadata, one-hot encodes every aligned position,
computes the first two PCs with a small randomized SVD, and writes both the
coordinates and PDF scatter plots colored by available temperature fields.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


FAMILIES = ("hsp70", "jdomain")
TEMPERATURE_COLUMNS = ("Tmin", "Topt_ave", "Topt_low", "Topt_high", "Tmax", "Tmax_Tmin")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())

    if current_id is not None:
        records.append((current_id, "".join(chunks)))

    return records


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["sequence_id"]: row for row in csv.DictReader(handle)}


def one_hot_encode(sequences: list[str]) -> tuple[np.ndarray, str]:
    if not sequences:
        raise ValueError("No sequences to encode.")

    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError(f"Expected aligned sequences of equal length, got {sorted(lengths)}.")

    alphabet = "".join(sorted({char for seq in sequences for char in seq}))
    alphabet_index = {char: idx for idx, char in enumerate(alphabet)}
    n_sequences = len(sequences)
    alignment_length = len(sequences[0])
    n_states = len(alphabet)

    encoded = np.zeros((n_sequences, alignment_length * n_states), dtype=np.float32)
    for row_index, seq in enumerate(sequences):
        for position, char in enumerate(seq):
            encoded[row_index, position * n_states + alphabet_index[char]] = 1.0

    return encoded, alphabet


def randomized_pca(
    matrix: np.ndarray,
    n_components: int = 2,
    oversamples: int = 8,
    n_iter: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA scores and explained variance ratio for the first components."""

    if matrix.shape[0] < 3:
        raise ValueError("PCA requires at least three observations.")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank = min(n_components + oversamples, centered.shape[0] - 1, centered.shape[1])
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((centered.shape[1], rank), dtype=np.float32)

    sketch = centered @ omega
    for _ in range(n_iter):
        sketch, _ = np.linalg.qr(sketch, mode="reduced")
        sketch = centered @ (centered.T @ sketch)
    q_matrix, _ = np.linalg.qr(sketch, mode="reduced")
    small = q_matrix.T @ centered
    u_hat, singular_values, _ = np.linalg.svd(small, full_matrices=False)

    scores = q_matrix @ (u_hat[:, :n_components] * singular_values[:n_components])
    variances = singular_values**2 / (centered.shape[0] - 1)
    total_variance = np.var(centered, axis=0, ddof=1).sum()
    explained = variances[:n_components] / total_variance
    return scores, explained


def pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x_centered = x_values - x_values.mean()
    y_centered = y_values - y_values.mean()
    denominator = math.sqrt(float((x_centered @ x_centered) * (y_centered @ y_centered)))
    if denominator == 0:
        return float("nan")
    return float((x_centered @ y_centered) / denominator)


def nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / 10**exponent
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * 10**exponent


def tick_values(values: np.ndarray, n_ticks: int = 7) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.array([vmin])
    step = nice_step((vmax - vmin) / max(n_ticks - 1, 1))
    tick_min = math.floor(vmin / step) * step
    tick_max = math.ceil(vmax / step) * step
    if tick_min > 0:
        tick_min = 0.0
    if tick_max < 0:
        tick_max = 0.0
    ticks = np.arange(tick_min, tick_max + step * 0.5, step)
    ticks[np.abs(ticks) < step * 1e-8] = 0.0
    return ticks


def write_pdf(
    path: Path,
    family: str,
    scores: np.ndarray,
    temperatures: np.ndarray,
    explained: np.ndarray,
    temperature_column: str,
) -> None:
    available = np.isfinite(temperatures)
    plot_scores = scores[available]
    plot_temperatures = temperatures[available]
    order = np.argsort(plot_temperatures)
    x_ticks = tick_values(scores[:, 0])
    y_ticks = tick_values(scores[:, 1])

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    scatter = ax.scatter(
        plot_scores[order, 0],
        plot_scores[order, 1],
        c=plot_temperatures[order],
        cmap="viridis",
        s=4,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(f"{family} PCA colored by {temperature_column}", loc="center", fontsize=13, pad=12)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.2f}% variance)")
    ax.set_xlim(float(x_ticks[0]), float(x_ticks[-1]))
    ax.set_ylim(float(y_ticks[0]), float(y_ticks[-1]))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, color="#d7dce2", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#5b6573")

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.025, shrink=0.92)
    colorbar.set_label(temperature_column)
    colorbar.ax.tick_params(labelsize=8, colors="#1f2933")
    colorbar.outline.set_linewidth(0.6)

    fig.savefig(path, format="pdf", dpi=300)
    plt.close(fig)


def write_coordinates(
    path: Path,
    records: list[tuple[str, str]],
    rows: list[dict[str, str]],
    scores: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = [
            "sequence_id",
            "accession",
            "organism",
            "taxonomy_id",
            *TEMPERATURE_COLUMNS,
            "PC1",
            "PC2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (sequence_id, _), row, score in zip(records, rows, scores):
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "accession": row.get("accession", ""),
                    "organism": row.get("organism", ""),
                    "taxonomy_id": row.get("taxonomy_id", ""),
                    **{column: row.get(column, "") for column in TEMPERATURE_COLUMNS},
                    "PC1": f"{score[0]:.8g}",
                    "PC2": f"{score[1]:.8g}",
                }
            )


def temperature_array(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(column, "")
        values.append(float(value) if value else np.nan)
    return np.array(values, dtype=float)


def summarize_temperature(
    family: str,
    column: str,
    temperatures: np.ndarray,
    scores: np.ndarray,
    explained: np.ndarray,
    base_summary: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    available = np.isfinite(temperatures)
    available_temperatures = temperatures[available]
    available_scores = scores[available]
    return {
        **base_summary,
        "temperature_column": column,
        "temperature_n": int(available.sum()),
        "temperature_min": float(available_temperatures.min()),
        "temperature_median": float(np.median(available_temperatures)),
        "temperature_max": float(available_temperatures.max()),
        "pc1_temperature_pearson": pearson(available_scores[:, 0], available_temperatures),
        "pc2_temperature_pearson": pearson(available_scores[:, 1], available_temperatures),
    }


def analyze_family(input_dir: Path, output_dir: Path, family: str, seed: int) -> list[dict[str, float | int | str]]:
    fasta_path = input_dir / family / f"{family}.aligned.maxgap50.fasta"
    metadata_path = input_dir / family / f"{family}.metadata.maxgap50.csv"

    metadata = read_metadata(metadata_path)
    fasta_records = read_fasta(fasta_path)
    records: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []

    for sequence_id, sequence in fasta_records:
        row = metadata.get(sequence_id)
        if row is None:
            continue
        records.append((sequence_id, sequence))
        rows.append(row)

    sequences = [sequence for _, sequence in records]
    encoded, alphabet = one_hot_encode(sequences)
    scores, explained = randomized_pca(encoded, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_coordinates(output_dir / f"{family}.pca_temperature.csv", records, rows, scores)

    base_summary = {
        "family": family,
        "n_sequences": len(records),
        "alignment_length": len(sequences[0]) if sequences else 0,
        "alphabet": alphabet,
        "pc1_explained": float(explained[0]),
        "pc2_explained": float(explained[1]),
    }
    summaries = []
    for column in TEMPERATURE_COLUMNS:
        temperatures = temperature_array(rows, column)
        if not np.isfinite(temperatures).any():
            continue
        write_pdf(
            output_dir / f"{family}.pca_{column}.pdf",
            family,
            scores,
            temperatures,
            explained,
            column,
        )
        summaries.append(summarize_temperature(family, column, temperatures, scores, explained, base_summary))

    return summaries


def write_summary(path: Path, summaries: list[dict[str, float | int | str]]) -> None:
    lines = [
        "One-hot MSA PCA temperature-space summary",
        "",
        f"Temperature columns: {', '.join(TEMPERATURE_COLUMNS)}",
        "Input: data/processed_new/<family>/<family>.aligned.maxgap50.fasta",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"{summary['family']}",
                f"  sequences: {summary['n_sequences']}",
                f"  alignment length: {summary['alignment_length']}",
                f"  alphabet: {summary['alphabet']}",
                f"  explained variance: PC1 {summary['pc1_explained'] * 100:.2f}%, PC2 {summary['pc2_explained'] * 100:.2f}%",
                f"  temperature column: {summary['temperature_column']}; n={summary['temperature_n']}",
                f"  {summary['temperature_column']} range: {summary['temperature_min']:.1f} - {summary['temperature_max']:.1f} C; median {summary['temperature_median']:.1f} C",
                f"  Pearson corr({summary['temperature_column']}, PC1): {summary['pc1_temperature_pearson']:.3f}",
                f"  Pearson corr({summary['temperature_column']}, PC2): {summary['pc2_temperature_pearson']:.3f}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed_new"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pca_temperature"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summaries = [
        summary
        for family in FAMILIES
        for summary in analyze_family(args.input_dir, args.output_dir, family, seed=args.seed)
    ]
    write_summary(args.output_dir / "summary.txt", summaries)

    for summary in summaries:
        print(
            f"{summary['family']} {summary['temperature_column']}: n={summary['temperature_n']}, "
            f"PC1={summary['pc1_explained'] * 100:.2f}%, "
            f"PC2={summary['pc2_explained'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
