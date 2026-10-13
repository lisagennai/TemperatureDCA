#!/usr/bin/env julia

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

using Printf
using PyPlot
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(ROOT, "results", "featuredca", "jdomain_natural_temperature_recovery")
const ALIGNMENT_LENGTH = 62

function parse_args(args)
    options = Dict{String,Any}("input_dir" => DEFAULT_INPUT, "bucket_width" => 10.0)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/plot_natural_test_likelihood_profiles.jl [--input-dir PATH] [--bucket-width 10]")
            exit(0)
        elseif arg in ("--input-dir", "--bucket-width")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            options[key] = key == "input_dir" ? abspath(args[i + 1]) : parse(Float64, args[i + 1])
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    options["bucket_width"] > 0 || error("--bucket-width must be positive")
    return options
end

function parse_grid(header)
    columns = findall(startswith("loglik_T"), header)
    grid = [parse(Float64, replace(header[c][length("loglik_T") + 1:end], "p" => ".")) for c in columns]
    return columns, grid
end

function read_likelihoods(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    columns, grid = parse_grid(header)
    rows = split.(lines[2:end], ',')
    ids = getindex.(rows, 1)
    taxonomy = getindex.(rows, 2)
    natural = parse.(Float64, getindex.(rows, 3))
    likelihoods = reduce(vcat, [reshape(parse.(Float64, row[columns]), 1, :) for row in rows])
    return ids, taxonomy, natural, likelihoods, grid
end

function aggregate_organisms(taxonomy, natural, likelihoods)
    organism_ids = unique(taxonomy)
    organism_t = Vector{Float64}(undef, length(organism_ids))
    organism_profiles = Matrix{Float64}(undef, length(organism_ids), size(likelihoods, 2))
    for (i, id) in enumerate(organism_ids)
        rows = findall(==(id), taxonomy)
        temperatures = unique(natural[rows])
        length(temperatures) == 1 || error("Organism $id has multiple temperatures")
        organism_t[i] = only(temperatures)
        organism_profiles[i, :] .= vec(mean(likelihoods[rows, :]; dims=1))
    end
    return organism_ids, organism_t, organism_profiles
end

function assign_buckets(temperatures, width)
    centers = collect(width:width:ceil(maximum(temperatures) / width) * width)
    assignments = [centers[argmin(abs.(centers .- temperature))] for temperature in temperatures]
    return assignments, centers
end

function profile_summaries(temperatures, likelihoods, grid, width)
    assignments, centers = assign_buckets(temperatures, width)
    summaries = NamedTuple[]
    for center in centers
        rows = findall(==(center), assignments)
        isempty(rows) && continue
        profiles = likelihoods[rows, :] ./ ALIGNMENT_LENGTH
        mean_profile = vec(mean(profiles; dims=1))
        standard_error = length(rows) > 1 ? vec(std(profiles; dims=1)) ./ sqrt(length(rows)) : zeros(length(grid))
        push!(summaries, (
            center=center, rows=rows, n=length(rows), mean_temperature=mean(temperatures[rows]),
            minimum_temperature=minimum(temperatures[rows]), maximum_temperature=maximum(temperatures[rows]),
            mean_profile=mean_profile, standard_error=standard_error,
            profile_peak=grid[argmax(mean_profile)],
        ))
    end
    return summaries
end

function plot_profiles(path, sequence_summaries, organism_summaries, grid)
    fig, axes = subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=true, sharey=true)
    colors = PyPlot.cm.viridis(range(0.05, 0.95, length=length(sequence_summaries)))
    for (panel, summaries, title) in zip(axes, (sequence_summaries, organism_summaries),
                                         ("Individual test sequences", "Homologs combined by organism"))
        for (i, summary) in enumerate(summaries)
            label = @sprintf("%.0f °C bucket (n=%d)", summary.center, summary.n)
            panel.plot(grid, summary.mean_profile, color=colors[i, :], linewidth=1.7, label=label)
            panel.fill_between(grid, summary.mean_profile .- 1.96 .* summary.standard_error,
                               summary.mean_profile .+ 1.96 .* summary.standard_error,
                               color=colors[i, :], alpha=0.08, linewidth=0)
            peak_index = argmax(summary.mean_profile)
            panel.scatter([grid[peak_index]], [summary.mean_profile[peak_index]], color=colors[i:i, :],
                          s=26, edgecolors="white", linewidths=0.5, zorder=3)
        end
        panel.set_xlabel("Temperature used for likelihood evaluation (degrees C)")
        panel.set_title(title)
        panel.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    end
    axes[1].set_ylabel("Mean log-likelihood per site")
    axes[2].legend(frameon=false, fontsize=7.5, loc="best")
    fig.suptitle("Conditional likelihood profiles of held-out natural J-domain sequences", fontsize=14)
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_relative_profiles(path, organism_summaries, grid)
    fig, ax = subplots(figsize=(7.8, 5.4), constrained_layout=true)
    colors = PyPlot.cm.viridis(range(0.05, 0.95, length=length(organism_summaries)))
    for (i, summary) in enumerate(organism_summaries)
        relative = summary.mean_profile .- maximum(summary.mean_profile)
        ax.plot(grid, relative, color=colors[i, :], linewidth=1.8,
                label=@sprintf("%.0f °C bucket: mean %.1f °C, peak %.1f °C",
                               summary.center, summary.mean_temperature, summary.profile_peak))
    end
    ax.axhline(0, color="#6b7280", linewidth=0.8)
    ax.set_xlabel("Temperature used for likelihood evaluation (degrees C)")
    ax.set_ylabel("Mean log-likelihood per site relative to profile maximum")
    ax.set_title("Organism-level natural likelihood profiles, aligned at zero")
    ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=false, fontsize=8)
    fig.savefig(path, dpi=300)
    close(fig)
end

function write_summary(path, sequence_summaries, organism_summaries)
    open(path, "w") do io
        println(io, "unit,bucket_center,n,natural_temperature_min,natural_temperature_max,natural_temperature_mean,mean_profile_argmax")
        for (unit, summaries) in (("sequence", sequence_summaries), ("organism", organism_summaries))
            for s in summaries
                @printf(io, "%s,%.6f,%d,%.6f,%.6f,%.6f,%.6f\n", unit, s.center, s.n,
                        s.minimum_temperature, s.maximum_temperature, s.mean_temperature, s.profile_peak)
            end
        end
    end
end

function main(args=ARGS)
    options = parse_args(args)
    input = joinpath(options["input_dir"], "test_likelihoods.csv")
    isfile(input) || error("Likelihood file not found: $input")
    _, taxonomy, natural, likelihoods, grid = read_likelihoods(input)
    _, organism_t, organism_likelihoods = aggregate_organisms(taxonomy, natural, likelihoods)
    sequence_summaries = profile_summaries(natural, likelihoods, grid, options["bucket_width"])
    organism_summaries = profile_summaries(organism_t, organism_likelihoods, grid, options["bucket_width"])

    plot_profiles(joinpath(options["input_dir"], "natural_likelihood_profiles.pdf"),
                  sequence_summaries, organism_summaries, grid)
    plot_relative_profiles(joinpath(options["input_dir"], "natural_likelihood_profiles_relative.pdf"),
                           organism_summaries, grid)
    write_summary(joinpath(options["input_dir"], "natural_likelihood_profile_summary.csv"),
                  sequence_summaries, organism_summaries)
    println("Likelihood-profile plots written to $(options["input_dir"])")
end

main()
