#!/usr/bin/env julia

using Printf
using PyPlot
using Statistics

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_FASTA = joinpath(ROOT, "results", "featuredca", "jdomain_temperature_ladder", "jdomain.temperature_ladder.fasta")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_temperature_ladder")

function parse_args(args)
    options = Dict{String,Any}("fasta" => DEFAULT_FASTA, "output_dir" => DEFAULT_OUTPUT)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/analyze_temperature_ladder_diversity.jl [--fasta PATH] [--output-dir PATH]")
            exit(0)
        elseif arg in ("--fasta", "--output-dir")
            i == length(args) && error("Missing value after $arg")
            options[replace(arg[3:end], "-" => "_")] = abspath(args[i + 1])
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return options
end

function read_ladder_fasta(path)
    groups = Dict{Float64,Vector{String}}()
    current_temperature = nothing
    chunks = String[]

    function finish_record!()
        current_temperature === nothing && return
        sequence = uppercase(join(chunks))
        push!(get!(groups, current_temperature, String[]), sequence)
    end

    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        if startswith(line, ">")
            finish_record!()
            matched = match(r"conditioned_Topt_ave=([-+0-9.eE]+)", line)
            matched === nothing && error("Missing conditioning temperature in FASTA header: $line")
            current_temperature = parse(Float64, matched.captures[1])
            empty!(chunks)
        else
            push!(chunks, line)
        end
    end
    finish_record!()
    return groups
end

function mean_pairwise_hamming(sequences)
    n = length(sequences)
    n >= 2 || error("At least two sequences are required")
    lengths = unique(length.(sequences))
    length(lengths) == 1 || error("Sequences are not aligned")
    alignment_length = only(lengths)
    pairs = n * (n - 1) ÷ 2
    total_differences = 0

    for position in 1:alignment_length
        counts = Dict{Char,Int}()
        for sequence in sequences
            residue = sequence[position]
            counts[residue] = get(counts, residue, 0) + 1
        end
        identical_pairs = sum(count * (count - 1) ÷ 2 for count in values(counts))
        total_differences += pairs - identical_pairs
    end

    mean_sites = total_differences / pairs
    return mean_sites, mean_sites / alignment_length, alignment_length
end

function calculate(groups)
    rows = NamedTuple[]
    for temperature in sort(collect(keys(groups)))
        sequences = groups[temperature]
        unique_count = length(unique(sequences))
        mean_sites, mean_fraction, alignment_length = mean_pairwise_hamming(sequences)
        push!(rows, (
            temperature=temperature,
            samples=length(sequences),
            unique_sequences=unique_count,
            unique_fraction=unique_count / length(sequences),
            duplicate_draws=length(sequences) - unique_count,
            mean_hamming_sites=mean_sites,
            mean_hamming_fraction=mean_fraction,
            alignment_length=alignment_length,
        ))
    end
    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "temperature,samples,unique_sequences,unique_fraction,duplicate_draws,mean_hamming_sites,mean_hamming_fraction,alignment_length")
        for row in rows
            @printf(io, "%.6g,%d,%d,%.10f,%d,%.10f,%.10f,%d\n",
                    row.temperature, row.samples, row.unique_sequences, row.unique_fraction,
                    row.duplicate_draws, row.mean_hamming_sites, row.mean_hamming_fraction,
                    row.alignment_length)
        end
    end
end

function plot_diversity(path, rows)
    temperatures = [row.temperature for row in rows]
    unique_fractions = 100 .* [row.unique_fraction for row in rows]
    hamming_sites = [row.mean_hamming_sites for row in rows]

    fig, axes = subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=true)
    axes[1].bar(temperatures, unique_fractions, width=3.2, color="#176b87")
    for (temperature, fraction, row) in zip(temperatures, unique_fractions, rows)
        axes[1].text(temperature, fraction + 1.2, "$(row.unique_sequences)", ha="center", va="bottom", fontsize=8)
    end
    axes[1].set_ylim(0, max(105, maximum(unique_fractions) + 8))
    axes[1].set_xlabel("Generation temperature (degrees C)")
    axes[1].set_ylabel("Unique sequences (%)")
    axes[1].set_title("Exact sequence uniqueness")

    axes[2].plot(temperatures, hamming_sites, marker="o", linewidth=1.8, markersize=5, color="#d05a34")
    for (temperature, distance) in zip(temperatures, hamming_sites)
        axes[2].text(temperature, distance + 0.25, @sprintf("%.2f", distance), ha="center", va="bottom", fontsize=8)
    end
    axes[2].set_xlabel("Generation temperature (degrees C)")
    axes[2].set_ylabel("Mean pairwise Hamming distance (positions / 62)")
    axes[2].set_title("Within-temperature sequence diversity")
    axes[2].set_ylim(minimum(hamming_sites) - 0.2, maximum(hamming_sites) + 0.35)

    for ax in axes
        ax.axvline(80, color="#6b7280", linestyle="--", linewidth=1)
        ax.grid(true, axis="y", color="#d8dde3", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(true)
    end
    fig.savefig(path, dpi=300)
    close(fig)
end

function main(args=ARGS)
    options = parse_args(args)
    isfile(options["fasta"]) || error("FASTA not found: $(options["fasta"])")
    groups = read_ladder_fasta(options["fasta"])
    rows = calculate(groups)
    mkpath(options["output_dir"])
    write_csv(joinpath(options["output_dir"], "diversity.csv"), rows)
    plot_diversity(joinpath(options["output_dir"], "sequence_diversity.pdf"), rows)
    open(joinpath(options["output_dir"], "diversity_summary.txt"), "w") do io
        println(io, "Hamming distance includes all 62 aligned positions, including gaps.")
        println(io, "The mean uses all unordered pairs; duplicate draws contribute distance zero.")
        for row in rows
            @printf(io, "T=%.0f C: unique=%d/%d (%.2f%%), duplicates=%d, mean_hamming=%.4f/62 (%.4f)\n",
                    row.temperature, row.unique_sequences, row.samples, 100 * row.unique_fraction,
                    row.duplicate_draws, row.mean_hamming_sites, row.mean_hamming_fraction)
        end
    end
    println("Diversity analysis written to $(options["output_dir"])")
end

main()
