#!/usr/bin/env julia

using FeatureDCA
using LinearAlgebra
using MultivariateStats
using Printf
using PyPlot
using Random
using Serialization
using Statistics

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_MODEL = joinpath(ROOT, "results", "featuredca", "jdomain_topt_ave.jls")
const DEFAULT_PCA_MODEL = joinpath(ROOT, "results", "featuredca", "jdomain_pca2.jls")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_temperature_ladder")
const DEFAULT_TEMPERATURES = [60.0, 70.0, 75.0, 80.0, 85.0, 90.0]
const AA = collect("ACDEFGHIKLMNPQRSTVWY-")

function parse_args(args)
    options = Dict{String,Any}(
        "model" => DEFAULT_MODEL,
        "pca_model" => DEFAULT_PCA_MODEL,
        "output_dir" => DEFAULT_OUTPUT,
        "temperatures" => copy(DEFAULT_TEMPERATURES),
        "samples" => 2000,
        "seed" => 1,
        "plot_points" => 2500,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/generate_temperature_ladder.jl [--temperatures 60,70,75,80,85,90] [--samples 2000] [--seed 1]")
            exit(0)
        elseif arg in ("--model", "--pca-model", "--output-dir", "--temperatures", "--samples", "--seed", "--plot-points")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            if key == "temperatures"
                options[key] = parse.(Float64, split(value, ','))
            elseif key in ("samples", "seed", "plot_points")
                options[key] = parse(Int, value)
            else
                options[key] = abspath(value)
            end
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    issorted(options["temperatures"]) || error("Temperatures must be sorted")
    length(unique(options["temperatures"])) == length(options["temperatures"]) || error("Temperatures must be unique")
    options["samples"] > 0 || error("--samples must be positive")
    return options
end

function logsumexp(values)
    maximum_value = maximum(values)
    return maximum_value + log(sum(exp, values .- maximum_value))
end

function sequence_loglikelihood(network, sequence, y)
    H, G, J = network.H, network.G, network.J
    logits = H[1] .+ G[1] * y
    total = logits[sequence[1]] - logsumexp(logits)
    for site in 2:length(H)
        logits .= H[site] .+ G[site] * y
        couplings = J[site - 1]
        @inbounds for previous in 1:site-1, amino_acid in eachindex(logits)
            logits[amino_acid] += couplings[previous, amino_acid, sequence[previous]]
        end
        total += logits[sequence[site]] - logsumexp(logits)
    end
    return total
end

function likelihood_grid(network, sequences, standardized_temperatures)
    n_sequences = size(sequences, 2)
    n_temperatures = length(standardized_temperatures)
    likelihoods = Matrix{Float64}(undef, n_sequences, n_temperatures)
    Threads.@threads for index in 1:n_sequences
        sequence = view(sequences, :, index)
        for temperature_index in 1:n_temperatures
            y = [standardized_temperatures[temperature_index]]
            likelihoods[index, temperature_index] = sequence_loglikelihood(network, sequence, y)
        end
    end
    return likelihoods
end

function write_fasta(path, sequences, labels, temperatures)
    open(path, "w") do io
        for m in axes(sequences, 2)
            @printf(io, ">ladder_%06d conditioned_Topt_ave=%.6g\n", m, temperatures[labels[m]])
            sequence = join(AA[sequences[:, m]])
            for start in 1:80:length(sequence)
                println(io, sequence[start:min(start + 79, end)])
            end
        end
    end
end

function write_likelihood_csv(path, likelihoods, labels, temperatures)
    open(path, "w") do io
        header = ["sequence_id", "conditioning_temperature", "argmax_temperature", ["loglik_T$(replace(string(t), "." => "p"))" for t in temperatures]...]
        println(io, join(header, ','))
        for m in axes(likelihoods, 1)
            winner = argmax(view(likelihoods, m, :))
            values = Any[@sprintf("ladder_%06d", m), temperatures[labels[m]], temperatures[winner], likelihoods[m, :]...]
            println(io, join(values, ','))
        end
    end
end

function plot_profiles(path, likelihoods, labels, temperatures, samples)
    fig, ax = subplots(figsize=(7.4, 5.4), constrained_layout=true)
    colors = PyPlot.cm.viridis(range(0.08, 0.92, length=length(temperatures)))
    for (index, conditioning_temperature) in enumerate(temperatures)
        rows = findall(==(index), labels)
        values = likelihoods[rows, :] ./ 62
        means = vec(mean(values; dims=1))
        errors = vec(std(values; dims=1)) ./ sqrt(samples)
        ax.errorbar(temperatures, means, yerr=1.96 .* errors, marker="o", markersize=4, linewidth=1.3,
                    color=colors[index, :], label=@sprintf("generated at %.0f C", conditioning_temperature))
    end
    ax.axvline(80, color="#6b7280", linestyle="--", linewidth=1, label="training maximum")
    ax.set_xlabel("Temperature used for likelihood evaluation (degrees C)")
    ax.set_ylabel("Mean log-likelihood per site")
    ax.set_title("Conditional likelihood profiles")
    ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=false, fontsize=8, ncol=2)
    fig.savefig(path, dpi=300)
    close(fig)
end

function confusion_matrix(likelihoods, labels, n_temperatures)
    counts = zeros(Int, n_temperatures, n_temperatures)
    for m in axes(likelihoods, 1)
        counts[labels[m], argmax(view(likelihoods, m, :))] += 1
    end
    return counts
end

function plot_confusion(path, counts, temperatures)
    fractions = counts ./ sum(counts; dims=2)
    fig, ax = subplots(figsize=(6.5, 5.5), constrained_layout=true)
    image = ax.imshow(fractions, origin="lower", vmin=0, vmax=1, cmap="Blues")
    for row in axes(counts, 1), column in axes(counts, 2)
        color = fractions[row, column] > 0.55 ? "white" : "#17202a"
        ax.text(column - 1, row - 1, @sprintf("%.1f%%\n%d", 100 * fractions[row, column], counts[row, column]),
                ha="center", va="center", fontsize=7, color=color)
    end
    labels = [@sprintf("%.0f", temperature) for temperature in temperatures]
    ax.set_xticks(0:length(temperatures)-1)
    ax.set_yticks(0:length(temperatures)-1)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Temperature with maximum log-likelihood (degrees C)")
    ax.set_ylabel("Generation temperature (degrees C)")
    ax.set_title("Likelihood-based temperature recovery")
    fig.colorbar(image, ax=ax, label="Row fraction")
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_pca(path, natural_coordinates, generated_coordinates, labels, temperatures, plot_points, seed)
    rng = MersenneTwister(seed)
    natural_selected = sort(randperm(rng, size(natural_coordinates, 2))[1:min(plot_points, size(natural_coordinates, 2))])
    fig, ax = subplots(figsize=(8.0, 6.1), constrained_layout=true)
    ax.scatter(natural_coordinates[1, natural_selected], natural_coordinates[2, natural_selected], s=5, alpha=0.14,
               color="#667085", linewidths=0, label="Natural", rasterized=true)
    colors = PyPlot.cm.plasma(range(0.06, 0.94, length=length(temperatures)))
    for (index, temperature) in enumerate(temperatures)
        members = findall(==(index), labels)
        selected = length(members) > plot_points ? members[randperm(rng, length(members))[1:plot_points]] : members
        ax.scatter(generated_coordinates[1, selected], generated_coordinates[2, selected], s=7, alpha=0.28,
                   color=colors[index:index, :], linewidths=0, label=@sprintf("%.0f C", temperature), rasterized=true)
    end
    ax.set_xlabel("Natural PCA PC1")
    ax.set_ylabel("Natural PCA PC2")
    ax.set_title("Temperature-ladder sequences in natural PCA space")
    ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=false, fontsize=8, ncol=2)
    fig.savefig(path, dpi=300)
    close(fig)
end

function main(args=ARGS)
    options = parse_args(args)
    model = deserialize(options["model"])
    pca_model = deserialize(options["pca_model"])
    model.feature_names == ["Topt_ave"] || error("Expected a Topt_ave-conditioned model")
    temperatures = options["temperatures"]
    standardized = (temperatures .- only(model.feature_means)) ./ only(model.feature_scales)
    samples = options["samples"]
    n_temperatures = length(temperatures)
    total = samples * n_temperatures
    generated = Matrix{Int}(undef, model.alignment_length, total)
    labels = repeat(1:n_temperatures; inner=samples)

    Random.seed!(options["seed"])
    for index in eachindex(temperatures)
        @printf("Generating %d sequences at %.1f C (standardized y=%.4f)...\n", samples, temperatures[index], standardized[index])
        columns = (index - 1) * samples + 1:index * samples
        generated[:, columns] .= FeatureDCA.sample(model.network, [standardized[index]], samples)
    end

    println("Evaluating every generated sequence across the full temperature ladder...")
    likelihoods = likelihood_grid(model.network, generated, standardized)
    counts = confusion_matrix(likelihoods, labels, n_temperatures)
    pca_generated = Matrix{Float64}(predict(pca_model.pca_map, FeatureDCA.one_hot(generated; q=model.alphabet_size)))

    output_dir = options["output_dir"]
    mkpath(output_dir)
    write_fasta(joinpath(output_dir, "jdomain.temperature_ladder.fasta"), generated, labels, temperatures)
    write_likelihood_csv(joinpath(output_dir, "likelihoods.csv"), likelihoods, labels, temperatures)
    plot_profiles(joinpath(output_dir, "likelihood_profiles.pdf"), likelihoods, labels, temperatures, samples)
    plot_confusion(joinpath(output_dir, "likelihood_argmax_confusion.pdf"), counts, temperatures)
    plot_pca(joinpath(output_dir, "pca_temperature_ladder.pdf"), pca_model.Y, pca_generated, labels, temperatures, options["plot_points"], options["seed"])

    exact_accuracy = sum(diag(counts)) / total
    adjacent_accuracy = sum(counts[row, column] for row in 1:n_temperatures, column in 1:n_temperatures if abs(row - column) <= 1) / total
    open(joinpath(output_dir, "summary.txt"), "w") do io
        @printf(io, "temperatures=%s\nsamples_per_temperature=%d\ntotal_sequences=%d\nseed=%d\n", join(temperatures, ","), samples, total, options["seed"])
        @printf(io, "exact_argmax_accuracy=%.10f\nadjacent_argmax_accuracy=%.10f\n", exact_accuracy, adjacent_accuracy)
        for row in 1:n_temperatures
            row_accuracy = counts[row, row] / samples
            average_ll = vec(mean(likelihoods[labels .== row, :]; dims=1)) ./ model.alignment_length
            winner = argmax(average_ll)
            @printf(io, "T%.0f_exact_accuracy=%.10f\nT%.0f_mean_profile_argmax=%.1f\n", temperatures[row], row_accuracy, temperatures[row], temperatures[winner])
        end
    end
    println("Analysis written to $output_dir")
end

main()
