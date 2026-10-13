#!/usr/bin/env julia

using Printf
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(ROOT, "results", "featuredca", "jdomain_natural_temperature_recovery")

function parse_args(args)
    options = Dict{String,Any}("input_dir" => DEFAULT_INPUT, "threshold" => 40.0,
                               "permutations" => 10_000, "bootstraps" => 5_000, "seed" => 23)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia analysis/analyze_natural_recovery_significance.jl [--input-dir PATH] [--threshold 40] [--permutations 10000] [--bootstraps 5000] [--seed 23]")
            exit(0)
        elseif arg in ("--input-dir", "--threshold", "--permutations", "--bootstraps", "--seed")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key == "input_dir" ? abspath(value) :
                           key in ("permutations", "bootstraps", "seed") ? parse(Int, value) : parse(Float64, value)
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return options
end

function tied_ranks(values)
    order = sortperm(values)
    ranks = Vector{Float64}(undef, length(values))
    first = 1
    while first <= length(order)
        last = first
        while last < length(order) && values[order[last + 1]] == values[order[first]]
            last += 1
        end
        ranks[order[first:last]] .= (first + last) / 2
        first = last + 1
    end
    return ranks
end

spearman(x, y) = cor(tied_ranks(x), tied_ranks(y))

function parse_grid(header)
    columns = findall(startswith("loglik_T"), header)
    values = [parse(Float64, replace(header[c][length("loglik_T") + 1:end], "p" => ".")) for c in columns]
    return columns, values
end

function read_likelihoods(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    columns, grid = parse_grid(header)
    rows = split.(lines[2:end], ',')
    taxonomy = getindex.(rows, 2)
    natural = parse.(Float64, getindex.(rows, 3))
    likelihoods = reduce(vcat, [reshape(parse.(Float64, row[columns]), 1, :) for row in rows])
    return taxonomy, natural, likelihoods, grid
end

function aggregate_organisms(taxonomy, natural, likelihoods)
    ids = unique(taxonomy)
    temperatures = Vector{Float64}(undef, length(ids))
    profiles = Matrix{Float64}(undef, length(ids), size(likelihoods, 2))
    counts = Vector{Int}(undef, length(ids))
    for (i, id) in enumerate(ids)
        rows = findall(==(id), taxonomy)
        values = unique(natural[rows])
        length(values) == 1 || error("Organism $id has multiple natural temperatures")
        temperatures[i] = only(values)
        profiles[i, :] .= vec(mean(likelihoods[rows, :]; dims=1))
        counts[i] = length(rows)
    end
    return ids, temperatures, profiles, counts
end

infer_temperature(likelihoods, grid) = [grid[argmax(view(likelihoods, i, :))] for i in axes(likelihoods, 1)]

function metrics(actual, inferred)
    errors = inferred .- actual
    return (n=length(actual), pearson=cor(actual, inferred), spearman=spearman(actual, inferred),
            mae=mean(abs.(errors)), rmse=sqrt(mean(errors .^ 2)), bias=mean(errors))
end

function permutation_pvalue(actual, inferred, statistic, n, rng; absolute=true)
    observed = statistic(actual, inferred)
    extreme = 0
    for _ in 1:n
        null = statistic(actual, inferred[randperm(rng, length(inferred))])
        extreme += absolute ? abs(null) >= abs(observed) : null >= observed
    end
    return (extreme + 1) / (n + 1)
end

function bootstrap_interval(actual, inferred, statistic, n, rng)
    values = Vector{Float64}(undef, n)
    for b in 1:n
        sample = rand(rng, eachindex(actual), length(actual))
        values[b] = statistic(actual[sample], inferred[sample])
    end
    return quantile(values, [0.025, 0.975])
end

function auc(labels, scores)
    positives = findall(labels)
    negatives = findall(.!labels)
    isempty(positives) || isempty(negatives) && return NaN
    ranks = tied_ranks(scores)
    return (sum(ranks[positives]) - length(positives) * (length(positives) + 1) / 2) /
           (length(positives) * length(negatives))
end

function classification_metrics(actual, inferred, likelihoods, grid, threshold)
    truth = actual .>= threshold
    predicted = inferred .>= threshold
    tp, tn = sum(truth .& predicted), sum((.!truth) .& (.!predicted))
    fp, fn = sum((.!truth) .& predicted), sum(truth .& (.!predicted))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    high_columns, low_columns = findall(grid .>= threshold), findall(grid .< threshold)
    score = [maximum(view(likelihoods, i, high_columns)) - maximum(view(likelihoods, i, low_columns)) for i in axes(likelihoods, 1)]
    return (positive=sum(truth), negative=sum(.!truth), prevalence=mean(truth), tp=tp, tn=tn, fp=fp, fn=fn,
            sensitivity=sensitivity, specificity=specificity, balanced_accuracy=(sensitivity + specificity) / 2,
            precision=precision, auc=auc(truth, score), score=score, truth=truth)
end

function main(args=ARGS)
    options = parse_args(args)
    taxonomy, natural, likelihoods, grid = read_likelihoods(joinpath(options["input_dir"], "test_likelihoods.csv"))
    sequence_inferred = infer_temperature(likelihoods, grid)
    organism_ids, organism_t, organism_likelihoods, homolog_counts = aggregate_organisms(taxonomy, natural, likelihoods)
    organism_inferred = infer_temperature(organism_likelihoods, grid)
    threshold = options["threshold"]
    rng = MersenneTwister(options["seed"])

    sequence_all = metrics(natural, sequence_inferred)
    organism_all = metrics(organism_t, organism_inferred)
    sequence_high_rows = findall(natural .>= threshold)
    organism_high_rows = findall(organism_t .>= threshold)
    sequence_high = metrics(natural[sequence_high_rows], sequence_inferred[sequence_high_rows])
    organism_high = metrics(organism_t[organism_high_rows], organism_inferred[organism_high_rows])
    classification = classification_metrics(organism_t, organism_inferred, organism_likelihoods, grid, threshold)

    pearson_p = permutation_pvalue(organism_t, organism_inferred, cor, options["permutations"], rng)
    spearman_p = permutation_pvalue(organism_t, organism_inferred, spearman, options["permutations"], rng)
    high_pearson_p = permutation_pvalue(organism_t[organism_high_rows], organism_inferred[organism_high_rows], cor,
                                        options["permutations"], rng)
    high_spearman_p = permutation_pvalue(organism_t[organism_high_rows], organism_inferred[organism_high_rows], spearman,
                                         options["permutations"], rng)
    auc_p = permutation_pvalue(Float64.(classification.truth), classification.score, (x, y) -> auc(Bool.(x), y),
                               options["permutations"], rng; absolute=false)
    pearson_ci = bootstrap_interval(organism_t, organism_inferred, cor, options["bootstraps"], rng)
    spearman_ci = bootstrap_interval(organism_t, organism_inferred, spearman, options["bootstraps"], rng)
    mae_ci = bootstrap_interval(organism_t, organism_inferred, (x, y) -> mean(abs.(y .- x)), options["bootstraps"], rng)
    high_pearson_ci = bootstrap_interval(organism_t[organism_high_rows], organism_inferred[organism_high_rows], cor,
                                         options["bootstraps"], rng)
    high_spearman_ci = bootstrap_interval(organism_t[organism_high_rows], organism_inferred[organism_high_rows], spearman,
                                          options["bootstraps"], rng)
    high_mae_ci = bootstrap_interval(organism_t[organism_high_rows], organism_inferred[organism_high_rows],
                                     (x, y) -> mean(abs.(y .- x)), options["bootstraps"], rng)
    auc_ci = bootstrap_interval(Float64.(classification.truth), classification.score,
                                (x, y) -> auc(Bool.(x), y), options["bootstraps"], rng)

    output = joinpath(options["input_dir"], @sprintf("significance_Tge%.0f.txt", threshold))
    open(output, "w") do io
        @printf(io, "threshold_C=%.6f\npermutations=%d\nbootstraps=%d\nseed=%d\n\n", threshold, options["permutations"], options["bootstraps"], options["seed"])
        for (name, m) in (("sequence_all", sequence_all), ("organism_all", organism_all),
                          ("sequence_high", sequence_high), ("organism_high", organism_high))
            @printf(io, "[%s]\nn=%d\npearson=%.10f\nspearman=%.10f\nmae_C=%.10f\nrmse_C=%.10f\nbias_C=%.10f\n\n",
                    name, m.n, m.pearson, m.spearman, m.mae, m.rmse, m.bias)
        end
        @printf(io, "[organism_all_inference]\npearson_permutation_p=%.10g\nspearman_permutation_p=%.10g\n", pearson_p, spearman_p)
        @printf(io, "pearson_95pct_CI=%.10f,%.10f\nspearman_95pct_CI=%.10f,%.10f\nmae_95pct_CI_C=%.10f,%.10f\n\n",
                pearson_ci..., spearman_ci..., mae_ci...)
        @printf(io, "[organism_high_inference]\npearson_permutation_p=%.10g\nspearman_permutation_p=%.10g\n", high_pearson_p, high_spearman_p)
        @printf(io, "pearson_95pct_CI=%.10f,%.10f\nspearman_95pct_CI=%.10f,%.10f\nmae_95pct_CI_C=%.10f,%.10f\n\n",
                high_pearson_ci..., high_spearman_ci..., high_mae_ci...)
        @printf(io, "[organism_high_classification]\npositive=%d\nnegative=%d\nprevalence=%.10f\nTP=%d\nTN=%d\nFP=%d\nFN=%d\n",
                classification.positive, classification.negative, classification.prevalence,
                classification.tp, classification.tn, classification.fp, classification.fn)
        @printf(io, "sensitivity=%.10f\nspecificity=%.10f\nbalanced_accuracy=%.10f\nprecision=%.10f\nAUROC=%.10f\nAUROC_permutation_p=%.10g\nAUROC_95pct_CI=%.10f,%.10f\n",
                classification.sensitivity, classification.specificity, classification.balanced_accuracy,
                classification.precision, classification.auc, auc_p, auc_ci...)
        @printf(io, "\norganisms=%d\nmean_homologs_per_organism=%.10f\nmax_homologs_per_organism=%d\n",
                length(organism_ids), mean(homolog_counts), maximum(homolog_counts))
    end
    println("Analysis written to $output")
end

main()
