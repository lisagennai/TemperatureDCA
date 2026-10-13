#!/usr/bin/env julia

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

using Printf
using PyPlot
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(ROOT, "results", "featuredca", "jdomain_natural_temperature_recovery_grid10_train90")
const ALIGNMENT_LENGTH = 62

function parse_args(args)
    options = Dict{String,Any}("input_dir" => DEFAULT_INPUT, "bucket_width" => 10.0,
                               "first_center" => 10.0, "last_center" => 80.0)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/plot_fixed_temperature_likelihood_heatmap.jl [--input-dir PATH]")
            exit(0)
        elseif arg in ("--input-dir", "--bucket-width", "--first-center", "--last-center")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            options[key] = key == "input_dir" ? abspath(args[i + 1]) : parse(Float64, args[i + 1])
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return options
end

function read_likelihoods(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    columns = findall(startswith("loglik_T"), header)
    grid = [parse(Float64, replace(header[c][length("loglik_T") + 1:end], "p" => ".")) for c in columns]
    rows = split.(lines[2:end], ',')
    temperatures = parse.(Float64, getindex.(rows, 3))
    likelihoods = reduce(vcat, [reshape(parse.(Float64, row[columns]), 1, :) for row in rows])
    return temperatures, likelihoods, grid
end

function assign_fixed_bucket(temperature, centers, width)
    index = floor(Int, (temperature - first(centers) + width / 2) / width) + 1
    return clamp(index, 1, length(centers))
end

function summarize_buckets(temperatures, likelihoods, grid, centers, width)
    assignments = [assign_fixed_bucket(t, centers, width) for t in temperatures]
    summaries = NamedTuple[]
    for (index, center) in enumerate(centers)
        rows = findall(==(index), assignments)
        isempty(rows) && continue
        profiles = likelihoods[rows, :] ./ ALIGNMENT_LENGTH
        mean_profile = vec(mean(profiles; dims=1))
        standard_error = length(rows) > 1 ? vec(std(profiles; dims=1)) ./ sqrt(length(rows)) : fill(NaN, length(grid))
        push!(summaries, (
            center=center, lower=center - width / 2, upper=center + width / 2,
            n=length(rows), natural_mean=mean(temperatures[rows]),
            natural_min=minimum(temperatures[rows]), natural_max=maximum(temperatures[rows]),
            profile=mean_profile, standard_error=standard_error, peak=grid[argmax(mean_profile)]))
    end
    return summaries
end

function draw_heatmap(path, summaries, grid; relative=false)
    values = reduce(vcat, [reshape(s.profile, 1, :) for s in summaries])
    colorbar_label = "Mean log-likelihood per site"
    title = "Held-out likelihood profiles in fixed 10 °C natural-temperature bins"
    if relative
        values .-= maximum(values; dims=2)
        colorbar_label = "Mean log-likelihood per site relative to row maximum"
        title = "Fixed-bin likelihood profiles, normalized to each row maximum"
    end

    fig, ax = subplots(figsize=(10.2, 5.8))
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.14, top=0.90)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=relative ? "viridis" : "magma",
                      extent=[first(grid) - 5, last(grid) + 5, 0.5, length(summaries) + 0.5])
    labels = [@sprintf("%.0f °C  [%.0f, %.0f)   n=%d", s.center, s.lower, s.upper, s.n) for s in summaries]
    ax.set_yticks(1:length(summaries))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(grid)
    ax.set_xlabel("Temperature used for likelihood evaluation (degrees C)")
    ax.set_ylabel("Natural Topt_ave bin")
    ax.set_title(title)
    for (row, summary) in enumerate(summaries)
        column = argmin(abs.(grid .- summary.peak))
        ax.scatter([grid[column]], [row], marker="s", s=42, facecolors="none", edgecolors="cyan",
                   linewidths=1.2)
    end
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=300)
    close(fig)
end

function write_summary(path, summaries)
    open(path, "w") do io
        println(io, "bucket_center,bucket_lower_inclusive,bucket_upper_exclusive,n,natural_temperature_min,natural_temperature_max,natural_temperature_mean,mean_profile_argmax")
        for s in summaries
            @printf(io, "%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6f\n", s.center, s.lower, s.upper,
                    s.n, s.natural_min, s.natural_max, s.natural_mean, s.peak)
        end
    end
end

function main(args=ARGS)
    options = parse_args(args)
    source = joinpath(options["input_dir"], "test_likelihoods.csv")
    isfile(source) || error("Likelihood CSV not found: $source")
    temperatures, likelihoods, grid = read_likelihoods(source)
    centers = collect(options["first_center"]:options["bucket_width"]:options["last_center"])
    summaries = summarize_buckets(temperatures, likelihoods, grid, centers, options["bucket_width"])
    draw_heatmap(joinpath(options["input_dir"], "likelihood_heatmap_fixed10.pdf"), summaries, grid)
    draw_heatmap(joinpath(options["input_dir"], "likelihood_heatmap_fixed10_relative.pdf"), summaries, grid; relative=true)
    write_summary(joinpath(options["input_dir"], "likelihood_heatmap_fixed10_summary.csv"), summaries)
    println("Fixed-bin heatmaps written to $(options["input_dir"])")
end

main()
