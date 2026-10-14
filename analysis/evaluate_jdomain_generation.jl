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
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_topt_ave_generation")
const AA = collect("ACDEFGHIKLMNPQRSTVWY-")

function parse_args(args)
    options = Dict{String,Any}(
        "model" => DEFAULT_MODEL,
        "fasta" => DEFAULT_FASTA,
        "output-dir" => DEFAULT_OUTPUT,
        "seed" => 1,
        "pca_points" => 6000,
        "pair_points" => 250000,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/evaluate_jdomain_generation.jl [--model PATH] [--fasta PATH] [--output-dir PATH] [--seed N]")
            exit(0)
        elseif arg in ("--model", "--fasta", "--output-dir", "--seed", "--pca-points", "--pair-points")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key in ("seed", "pca_points", "pair_points") ? parse(Int, value) : abspath(value)
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
    denom = norm(xc) * norm(yc)
    return denom == 0 ? NaN : dot(xc, yc) / denom
end

rmse(x, y) = sqrt(mean(abs2, x .- y))

function one_hot(Z; q=21)
    L, M = size(Z)
    encoded = zeros(Float32, L * q, M)
    @inbounds for m in 1:M, i in 1:L
        encoded[(i - 1) * q + Z[i, m], m] = 1
    end
    return encoded
end

function randomized_pca(encoded; n_components=2, oversamples=10, n_iter=3, seed=1)
    n_features, n_observations = size(encoded)
    rank = min(n_components + oversamples, n_features, n_observations - 1)
    mean_vector = vec(mean(encoded; dims=2))
    centered = encoded .- mean_vector
    rng = MersenneTwister(seed)
    sketch = centered * randn(rng, Float32, n_observations, rank)
    for _ in 1:n_iter
        sketch = Matrix(qr(sketch).Q)
        sketch = centered * (centered' * sketch)
    end
    basis = Matrix(qr(sketch).Q)
    decomposition = svd(basis' * centered; full=false)
    projection = basis * decomposition.U[:, 1:n_components]
    scores = projection' * centered
    variances = decomposition.S[1:n_components] .^ 2 ./ (n_observations - 1)
    total_variance = sum(abs2, centered) / (n_observations - 1)
    return scores, variances ./ total_variance
end

function generated_fasta(path, Z, ids, temperatures)
    open(path, "w") do io
        for m in axes(Z, 2)
            @printf(io, ">%s|generated Topt_ave=%.6g\n", ids[m], temperatures[m])
            sequence = join(AA[Z[:, m]])
            for start in 1:80:length(sequence)
                println(io, sequence[start:min(start + 79, end)])
            end
        end
    end
end

function frequency_vectors(Znatural, Zgenerated, weights)
    single_natural, pair_natural = FeatureDCA.compute_freq(Matrix{Int}(Znatural), Float64.(weights))
    single_generated, pair_generated = FeatureDCA.compute_freq(Matrix{Int}(Zgenerated), Float64.(weights))
    L = size(Znatural, 1)
    q_no_gap = 20
    pair_mask = falses(L * q_no_gap, L * q_no_gap)
    for i in 1:L, j in (i + 1):L
        pair_mask[(i - 1) * q_no_gap + 1:i * q_no_gap,
                  (j - 1) * q_no_gap + 1:j * q_no_gap] .= true
    end
    return single_natural, single_generated, pair_natural[pair_mask], pair_generated[pair_mask], pair_natural, pair_generated
end

function plot_frequency_scatter(path, fn1, fg1, fn2, fg2, pair_points, seed)
    rng = MersenneTwister(seed)
    selected = length(fn2) > pair_points ? sort(randperm(rng, length(fn2))[1:pair_points]) : eachindex(fn2)
    fig, axes = subplots(1, 2, figsize=(10.4, 4.7), constrained_layout=true)
    for (ax, x, y, title) in zip(axes, (fn1, fn2[selected]), (fg1, fg2[selected]), ("Single-site frequencies", "Pair frequencies"))
        ax.scatter(x, y, s=4, alpha=0.18, color="#176b87", linewidths=0, rasterized=true)
        limit = max(maximum(x), maximum(y)) * 1.02
        ax.plot([0, limit], [0, limit], color="#b42318", linewidth=1.2, linestyle="--")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Natural weighted frequency")
        ax.set_ylabel("Generated weighted frequency")
        ax.set_title(title)
        ax.text(0.04, 0.95, @sprintf("r = %.4f\nRMSE = %.5f", pearson(x, y), rmse(x, y)), transform=ax.transAxes, va="top")
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    end
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_single_heatmaps(path, fn, fg, L)
    natural = reshape(fn, 20, L)
    generated = reshape(fg, 20, L)
    difference = generated - natural
    vmax = maximum(abs, difference)
    fig, axes = subplots(3, 1, figsize=(12, 7.5), constrained_layout=true, sharex=true)
    images = Any[
        axes[1].imshow(natural, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=max(maximum(natural), maximum(generated))),
        axes[2].imshow(generated, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=max(maximum(natural), maximum(generated))),
        axes[3].imshow(difference, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax),
    ]
    for (ax, title, image) in zip(axes, ("Natural", "Generated", "Generated - natural"), images)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Amino acid")
        ax.set_yticks(0:19)
        ax.set_yticklabels(string.(AA[1:20]), fontsize=7)
        fig.colorbar(image, ax=ax, pad=0.01, aspect=25)
    end
    axes[3].set_xlabel("Alignment position")
    position_ticks = collect(0:4:L-1)
    axes[3].set_xticks(position_ticks)
    axes[3].set_xticklabels(string.(position_ticks .+ 1))
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_pair_rmse(path, pn, pg, L)
    residual = zeros(L, L)
    for i in 1:L, j in i+1:L
        rows = (i - 1) * 20 + 1:i * 20
        cols = (j - 1) * 20 + 1:j * 20
        residual[i, j] = rmse(pn[rows, cols], pg[rows, cols])
        residual[j, i] = residual[i, j]
    end
    fig, ax = subplots(figsize=(7.2, 6.2), constrained_layout=true)
    image = ax.imshow(residual, origin="lower", interpolation="nearest", cmap="magma")
    ax.set_xlabel("Alignment position")
    ax.set_ylabel("Alignment position")
    ax.set_title("Pair-frequency RMSE by position pair")
    fig.colorbar(image, ax=ax, label="RMSE across 20 x 20 amino-acid pairs")
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_pca(path, scores, M, temperatures, explained, pca_points, seed)
    rng = MersenneTwister(seed)
    nplot = min(M, pca_points)
    selected = sort(randperm(rng, M)[1:nplot])
    natural = scores[:, 1:M]
    generated = scores[:, M+1:2M]
    fig, axes = subplots(1, 3, figsize=(15, 4.7), constrained_layout=true, sharex=true, sharey=true)
    axes[1].scatter(natural[1, selected], natural[2, selected], s=5, alpha=0.25, color="#176b87", label="Natural", linewidths=0, rasterized=true)
    axes[1].scatter(generated[1, selected], generated[2, selected], s=5, alpha=0.25, color="#d05a34", label="Generated", linewidths=0, rasterized=true)
    axes[1].legend(frameon=false, markerscale=2)
    axes[1].set_title("Natural and generated")
    vmin, vmax = extrema(temperatures)
    scatter_n = axes[2].scatter(natural[1, selected], natural[2, selected], c=temperatures[selected], s=6, alpha=0.45, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0, rasterized=true)
    axes[2].set_title("Natural, colored by Topt")
    axes[3].scatter(generated[1, selected], generated[2, selected], c=temperatures[selected], s=6, alpha=0.45, cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0, rasterized=true)
    axes[3].set_title("Generated, colored by conditioning Topt")
    for ax in axes
        ax.set_xlabel(@sprintf("PC1 (%.2f%%)", 100 * explained[1]))
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    end
    axes[1].set_ylabel(@sprintf("PC2 (%.2f%%)", 100 * explained[2]))
    fig.colorbar(scatter_n, ax=axes[2:3], label="Topt_ave (degrees C)", pad=0.02)
    fig.savefig(path, dpi=300)
    close(fig)
end

function write_pca_csv(path, ids, temperatures, scores, M)
    open(path, "w") do io
        println(io, "sequence_id,type,Topt_ave,PC1,PC2")
        for m in 1:M
            @printf(io, "%s,natural,%.8g,%.8g,%.8g\n", ids[m], temperatures[m], scores[1, m], scores[2, m])
            @printf(io, "%s,generated,%.8g,%.8g,%.8g\n", ids[m], temperatures[m], scores[1, M + m], scores[2, M + m])
        end
    end
end

function main(args=ARGS)
    options = parse_args(args)
    isfile(options["model"]) || error("Model not found: $(options["model"])")
    isfile(options["fasta"]) || error("FASTA not found: $(options["fasta"])")
    mkpath(options["output_dir"])

    artifact = deserialize(options["model"])
    artifact.feature_names == ["Topt_ave"] || error("Expected a one-feature Topt_ave model")
    weights, Znatural, L, M, q, annotations = FeatureDCA.read_annotated_fasta(options["fasta"]; remove_dups=false, max_gap_fraction=1.0)
    ids = [first(split(strip(only(annotation)))) for annotation in annotations]
    ids == artifact.sequence_ids || error("Model sequence IDs do not match FASTA order")
    #weights = artifact.weights
    weights = ones(length(ids))   # unweighted comparison
    cohort_path = replace(options["model"], r"\.[^.]*$" => ".cohort.csv")
    temperatures = Float64[]
    standardized = Float64[]
    for (line_number, line) in enumerate(eachline(cohort_path))
        line_number == 1 && continue
        fields = split(line, ',')
        push!(temperatures, parse(Float64, fields[5]))
        push!(standardized, parse(Float64, fields[6]))
    end
    length(temperatures) == M || error("Cohort and FASTA have different lengths")
    expected = (temperatures .- only(artifact.feature_means)) ./ only(artifact.feature_scales)
    maximum(abs, expected .- standardized) < 1e-10 || error("Stored standardized temperatures are inconsistent")
    Y = reshape(standardized, 1, :)

    Random.seed!(options["seed"])
    println("Generating one sequence for each of $M temperatures...")
    Zgenerated = FeatureDCA.sample(artifact.network, Y)
    generated_fasta(joinpath(options["output_dir"], "$(options["model"])_generated.fasta"), Zgenerated, ids, temperatures)

    println("Computing weighted one- and two-point frequencies...")
    fn1, fg1, fn2, fg2, pn, pg = frequency_vectors(Znatural, Zgenerated, weights)
    gap_natural = [dot(weights, Znatural[i, :] .== 21) / sum(weights) for i in 1:L]
    gap_generated = [dot(weights, Zgenerated[i, :] .== 21) / sum(weights) for i in 1:L]
    plot_frequency_scatter(joinpath(options["output_dir"], "frequency_scatter.pdf"), fn1, fg1, fn2, fg2, options["pair_points"], options["seed"])
    plot_single_heatmaps(joinpath(options["output_dir"], "single_frequency_heatmaps.pdf"), fn1, fg1, L)
    plot_pair_rmse(joinpath(options["output_dir"], "pair_frequency_rmse.pdf"), pn, pg, L)

    println("Fitting shared two-component PCA to natural and generated one-hot encodings...")
    encoded = one_hot(hcat(Znatural, Zgenerated); q=q)
    scores, explained = randomized_pca(encoded; seed=options["seed"])
    plot_pca(joinpath(options["output_dir"], "pca_comparison.pdf"), scores, M, temperatures, explained, options["pca_points"], options["seed"])
    write_pca_csv(joinpath(options["output_dir"], "pca_scores.csv"), ids, temperatures, scores, M)

    summary_path = joinpath(options["output_dir"], "summary.txt")
    open(summary_path, "w") do io
        @printf(io, "sequences=%d\nalignment_length=%d\nseed=%d\n", M, L, options["seed"])
        @printf(io, "single_frequency_pearson=%.10f\nsingle_frequency_rmse=%.10f\n", pearson(fn1, fg1), rmse(fn1, fg1))
        @printf(io, "pair_frequency_pearson=%.10f\npair_frequency_rmse=%.10f\n", pearson(fn2, fg2), rmse(fn2, fg2))
        @printf(io, "gap_frequency_pearson=%.10f\ngap_frequency_rmse=%.10f\n", pearson(gap_natural, gap_generated), rmse(gap_natural, gap_generated))
        @printf(io, "pca_pc1_explained=%.10f\npca_pc2_explained=%.10f\n", explained[1], explained[2])
        @printf(io, "natural_pc1_temperature_pearson=%.10f\nnatural_pc2_temperature_pearson=%.10f\n", pearson(scores[1, 1:M], temperatures), pearson(scores[2, 1:M], temperatures))
        @printf(io, "generated_pc1_temperature_pearson=%.10f\ngenerated_pc2_temperature_pearson=%.10f\n", pearson(scores[1, M+1:2M], temperatures), pearson(scores[2, M+1:2M], temperatures))
    end
    println("Analysis written to $(options["output_dir"])")
end

main()
