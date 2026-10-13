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
const DEFAULT_MODEL = joinpath(ROOT, "results", "featuredca", "jdomain_pca2.jls")
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_pca2_generation")
const AA = collect("ACDEFGHIKLMNPQRSTVWY-")

function parse_args(args)
    options = Dict{String,Any}("model" => DEFAULT_MODEL, "fasta" => DEFAULT_FASTA, "output_dir" => DEFAULT_OUTPUT, "seed" => 1, "plot_points" => 7000)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/evaluate_jdomain_pca_generation.jl [--model PATH] [--fasta PATH] [--output-dir PATH] [--seed N]")
            exit(0)
        elseif arg in ("--model", "--fasta", "--output-dir", "--seed", "--plot-points")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key in ("seed", "plot_points") ? parse(Int, value) : abspath(value)
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return options
end

function pearson(x, y)
    xc = x .- mean(x)
    yc = y .- mean(y)
    return dot(xc, yc) / (norm(xc) * norm(yc))
end

function write_fasta(path, Z, ids, conditioned)
    open(path, "w") do io
        for m in axes(Z, 2)
            @printf(io, ">%s|generated conditioned_PC1=%.8g conditioned_PC2=%.8g\n", ids[m], conditioned[1, m], conditioned[2, m])
            sequence = join(AA[Z[:, m]])
            for start in 1:80:length(sequence)
                println(io, sequence[start:min(start + 79, end)])
            end
        end
    end
end

function axis_limits(natural, generated)
    values = vcat(vec(natural), vec(generated))
    low, high = quantile(values, [0.001, 0.999])
    padding = 0.04 * (high - low)
    return low - padding, high + padding
end

function plot_distributions(path, natural, generated, explained, npoints, seed)
    M = size(natural, 2)
    selected = sort(randperm(MersenneTwister(seed), M)[1:min(M, npoints)])
    xlim = axis_limits(natural[1, :], generated[1, :])
    ylim = axis_limits(natural[2, :], generated[2, :])
    fig, axes = subplots(1, 3, figsize=(14.8, 4.7), constrained_layout=true, sharex=true, sharey=true)
    axes[1].scatter(natural[1, selected], natural[2, selected], s=5, alpha=0.3, color="#176b87", linewidths=0, rasterized=true)
    axes[1].set_title("Natural")
    axes[2].scatter(generated[1, selected], generated[2, selected], s=5, alpha=0.3, color="#d05a34", linewidths=0, rasterized=true)
    axes[2].set_title("Generated")
    axes[3].scatter(natural[1, selected], natural[2, selected], s=5, alpha=0.20, color="#176b87", label="Natural", linewidths=0, rasterized=true)
    axes[3].scatter(generated[1, selected], generated[2, selected], s=5, alpha=0.20, color="#d05a34", label="Generated", linewidths=0, rasterized=true)
    axes[3].set_title("Overlay")
    axes[3].legend(frameon=false, markerscale=2)
    for ax in axes
        ax.set_xlim(xlim...)
        ax.set_ylim(ylim...)
        ax.set_xlabel(@sprintf("PC1 (%.2f%% natural variance)", 100 * explained[1]))
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    end
    axes[1].set_ylabel(@sprintf("PC2 (%.2f%% natural variance)", 100 * explained[2]))
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_condition_recovery(path, conditioned, generated)
    fig, axes = subplots(1, 2, figsize=(10.2, 4.7), constrained_layout=true)
    for k in 1:2
        ax = axes[k]
        ax.scatter(conditioned[k, :], generated[k, :], s=3, alpha=0.15, color="#176b87", linewidths=0, rasterized=true)
        low = min(minimum(conditioned[k, :]), minimum(generated[k, :]))
        high = max(maximum(conditioned[k, :]), maximum(generated[k, :]))
        ax.plot([low, high], [low, high], "--", color="#b42318", linewidth=1)
        ax.set_xlabel("Conditioning PC$k")
        ax.set_ylabel("Generated sequence projected PC$k")
        ax.set_title(@sprintf("PC%d recovery, r = %.4f", k, pearson(conditioned[k, :], generated[k, :])))
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    end
    fig.savefig(path, dpi=300)
    close(fig)
end

function main(args=ARGS)
    options = parse_args(args)
    artifact = deserialize(options["model"])
    weights, Znatural, L, M, q, annotations = FeatureDCA.read_annotated_fasta(options["fasta"]; remove_dups=false, max_gap_fraction=1.0)
    ids = [first(split(strip(only(annotation)))) for annotation in annotations]
    ids == artifact.sequence_ids || error("Model sequence IDs do not match the natural FASTA")
    Ynatural = artifact.Y
    size(Ynatural) == (2, M) || error("Expected a 2 x M PCA conditioning matrix")

    Random.seed!(options["seed"])
    println("Generating one sequence for each of $M natural PCA coordinates...")
    Zgenerated = FeatureDCA.sample(artifact.network, Ynatural)
    Ygenerated = Matrix{Float64}(predict(artifact.pca_map, FeatureDCA.one_hot(Zgenerated; q=q)))
    mkpath(options["output_dir"])
    write_fasta(joinpath(options["output_dir"], "jdomain.generated.fasta"), Zgenerated, ids, Ynatural)

    plot_distributions(joinpath(options["output_dir"], "pca_distribution.pdf"), Ynatural, Ygenerated, artifact.explained_variance, options["plot_points"], options["seed"])
    plot_condition_recovery(joinpath(options["output_dir"], "pca_condition_recovery.pdf"), Ynatural, Ygenerated)
    open(joinpath(options["output_dir"], "pca_scores.csv"), "w") do io
        println(io, "sequence_id,natural_PC1,natural_PC2,generated_PC1,generated_PC2")
        for m in 1:M
            @printf(io, "%s,%.10g,%.10g,%.10g,%.10g\n", ids[m], Ynatural[1, m], Ynatural[2, m], Ygenerated[1, m], Ygenerated[2, m])
        end
    end
    cov_natural = cov(permutedims(Ynatural))
    cov_generated = cov(permutedims(Ygenerated))
    open(joinpath(options["output_dir"], "summary.txt"), "w") do io
        @printf(io, "sequences=%d\nalignment_length=%d\nseed=%d\n", M, L, options["seed"])
        for k in 1:2
            @printf(io, "PC%d_explained_variance=%.10f\n", k, artifact.explained_variance[k])
            @printf(io, "PC%d_natural_mean=%.10f\nPC%d_generated_mean=%.10f\n", k, mean(Ynatural[k, :]), k, mean(Ygenerated[k, :]))
            @printf(io, "PC%d_natural_std=%.10f\nPC%d_generated_std=%.10f\n", k, std(Ynatural[k, :]), k, std(Ygenerated[k, :]))
            @printf(io, "PC%d_condition_recovery_pearson=%.10f\n", k, pearson(Ynatural[k, :], Ygenerated[k, :]))
        end
        @printf(io, "natural_PC1_PC2_covariance=%.10f\ngenerated_PC1_PC2_covariance=%.10f\n", cov_natural[1, 2], cov_generated[1, 2])
    end
    println("Analysis written to $(options["output_dir"])")
end

main()
