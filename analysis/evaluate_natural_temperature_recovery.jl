#!/usr/bin/env julia

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

using FeatureDCA
using Printf
using PyPlot
using Random
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_METADATA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.metadata.maxgap50.csv")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_natural_temperature_recovery")
const AA = collect("ACDEFGHIKLMNPQRSTVWY-")

function parse_args(args)
    options = Dict{String,Any}(
        "fasta" => DEFAULT_FASTA, "metadata" => DEFAULT_METADATA,
        "output_dir" => DEFAULT_OUTPUT, "test_fraction" => 0.2,
        "strata" => 10, "grid_min" => nothing, "grid_max" => nothing,
        "grid_step" => 0.5, "seed" => 17,
        "lambda_h" => 0.01, "lambda_g" => 0.01, "lambda_j" => 0.01,
        "epsconv" => 1e-5, "maxit" => 1000, "verbose" => true,
    )
    value_args = Set(["--fasta", "--metadata", "--output-dir", "--test-fraction", "--strata",
        "--grid-min", "--grid-max", "--grid-step", "--seed", "--lambda_h", "--lambda_g", "--lambda_j",
                      "--epsconv", "--maxit"])
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/evaluate_natural_temperature_recovery.jl [options]")
            println("Splits organisms into stratified 80/20 train/test sets, trains FeatureDCA, and evaluates held-out temperature recovery.")
            exit(0)
        elseif arg == "--quiet"
            options["verbose"] = false
        elseif arg in value_args
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            if key in ("fasta", "metadata", "output_dir")
                options[key] = abspath(value)
            elseif key in ("strata", "seed", "maxit")
                options[key] = parse(Int, value)
            else
                options[key] = parse(Float64, value)
            end
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    0 < options["test_fraction"] < 1 || error("--test-fraction must be between zero and one")
    options["strata"] >= 2 || error("--strata must be at least 2")
    options["grid_step"] > 0 || error("--grid-step must be positive")
    (options["grid_min"] === nothing || options["grid_max"] === nothing ||
     options["grid_min"] <= options["grid_max"]) || error("--grid-min must not exceed --grid-max")
    return options
end

function parse_csv(text::AbstractString)
    rows, fields = Vector{Vector{String}}(), String[]
    buffer, in_quotes = IOBuffer(), false
    i = firstindex(text)
    while i <= lastindex(text)
        char = text[i]
        if char == '"'
            next_i = nextind(text, i)
            if in_quotes && next_i <= lastindex(text) && text[next_i] == '"'
                print(buffer, '"'); i = next_i
            else
                in_quotes = !in_quotes
            end
        elseif char == ',' && !in_quotes
            push!(fields, String(take!(buffer)))
        elseif (char == '\n' || char == '\r') && !in_quotes
            if char == '\r'
                next_i = nextind(text, i)
                next_i <= lastindex(text) && text[next_i] == '\n' && (i = next_i)
            end
            push!(fields, String(take!(buffer)))
            !(length(fields) == 1 && isempty(fields[1])) && push!(rows, fields)
            fields = String[]
        else
            print(buffer, char)
        end
        i = nextind(text, i)
    end
    if position(buffer) > 0 || !isempty(fields)
        push!(fields, String(take!(buffer))); push!(rows, fields)
    end
    in_quotes && error("Unterminated quoted CSV field")
    return rows
end

function read_metadata(path)
    parsed = parse_csv(read(path, String))
    header = parsed[1]
    return Dict(row[findfirst(==("sequence_id"), header)] =>
                Dict(header[i] => row[i] for i in eachindex(header)) for row in parsed[2:end])
end

annotation_text(x::AbstractString) = x
annotation_text(x::Tuple) = only(x)
sequence_id(x) = first(split(strip(annotation_text(x))))

function organism_stratified_split(ids, temperatures, taxonomy_ids, fraction, n_strata, seed)
    groups = Dict{String,Vector{Int}}()
    for i in eachindex(ids)
        push!(get!(groups, taxonomy_ids[i], Int[]), i)
    end
    group_ids = collect(keys(groups))
    group_temperatures = [only(unique(temperatures[groups[id]])) for id in group_ids]
    order = sortperm(group_temperatures)
    strata = [Int[] for _ in 1:n_strata]
    for (rank, group_index) in enumerate(order)
        stratum = min(n_strata, fld((rank - 1) * n_strata, length(order)) + 1)
        push!(strata[stratum], group_index)
    end

    rng = MersenneTwister(seed)
    test_groups = Set{String}()
    for members in strata
        shuffled = members[randperm(rng, length(members))]
        target = fraction * sum(length(groups[group_ids[g]]) for g in members)
        selected = 0
        for g in shuffled
            size = length(groups[group_ids[g]])
            if selected < target && (selected + size <= target || selected == 0 || rand(rng) < (target - selected) / size)
                push!(test_groups, group_ids[g]); selected += size
            end
        end
    end
    test = findall(i -> taxonomy_ids[i] in test_groups, eachindex(ids))
    train = setdiff(collect(eachindex(ids)), test)
    isempty(intersect(Set(taxonomy_ids[train]), Set(taxonomy_ids[test]))) || error("Taxonomy leakage in split")
    return train, test
end

function logsumexp(values)
    m = maximum(values)
    return m + log(sum(exp, values .- m))
end

function sequence_loglikelihood(network, sequence, y)
    H, G, J = network.H, network.G, network.J
    logits = H[1] .+ G[1] .* y
    total = logits[sequence[1]] - logsumexp(logits)
    for site in 2:length(H)
        logits .= H[site] .+ G[site] .* y
        coupling = J[site - 1]
        @inbounds for previous in 1:site-1, state in eachindex(logits)
            logits[state] += coupling[previous, state, sequence[previous]]
        end
        total += logits[sequence[site]] - logsumexp(logits)
    end
    return total
end

function likelihood_grid(network, Z, standardized_grid)
    values = Matrix{Float64}(undef, size(Z, 2), length(standardized_grid))
    Threads.@threads for m in axes(Z, 2)
        for t in eachindex(standardized_grid)
            values[m, t] = sequence_loglikelihood(network, view(Z, :, m), [standardized_grid[t]])
        end
    end
    return values
end

function csv_escape(value)
    text = string(value)
    return any(c -> occursin(c, text), [',', '"', '\n']) ? "\"" * replace(text, "\"" => "\"\"") * "\"" : text
end

function write_fasta(path, Z, ids, temperatures)
    open(path, "w") do io
        for m in axes(Z, 2)
            @printf(io, ">%s Topt_ave=%.6g\n", ids[m], temperatures[m])
            sequence = join(AA[Z[:, m]])
            for start in 1:80:length(sequence)
                println(io, sequence[start:min(start + 79, end)])
            end
        end
    end
end

function temperature_bins(values, n)
    bins = zeros(Int, length(values))
    order = sortperm(values)
    for (rank, index) in enumerate(order)
        bins[index] = min(n, fld((rank - 1) * n, length(values)) + 1)
    end
    return bins
end

function plot_split(path, train_t, test_t)
    bins = range(minimum(vcat(train_t, test_t)), maximum(vcat(train_t, test_t)), length=21)
    fig, ax = subplots(figsize=(7.2, 4.8), constrained_layout=true)
    ax.hist(train_t, bins=bins, density=true, alpha=0.55, color="#237f8e", label="Train")
    ax.hist(test_t, bins=bins, density=true, histtype="step", linewidth=2, color="#d05a47", label="Test")
    ax.set(xlabel="Topt_ave (degrees C)", ylabel="Density", title="Temperature-stratified organism-level split")
    ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7); ax.legend(frameon=false)
    fig.savefig(path, dpi=300); close(fig)
end

function plot_recovery(path, true_t, predicted_t, step)
    fig, ax = subplots(figsize=(6.2, 5.6), constrained_layout=true)
    hb = ax.hexbin(true_t, predicted_t, gridsize=35, mincnt=1, bins="log", cmap="viridis")
    limits = extrema(vcat(true_t, predicted_t))
    ax.plot(limits, limits, linestyle="--", color="#d05a47", linewidth=1.5, label="Perfect recovery")
    ax.set(xlabel="Natural Topt_ave (degrees C)", ylabel="Maximum-likelihood temperature (degrees C)",
           title=@sprintf("Held-out natural sequence recovery (grid step %.1f C)", step))
    ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.6); ax.legend(frameon=false)
    fig.colorbar(hb, ax=ax, label="log10 sequence count")
    fig.savefig(path, dpi=300); close(fig)
end

function plot_error(path, errors)
    fig, ax = subplots(figsize=(7.0, 4.7), constrained_layout=true)
    ax.hist(errors, bins=40, color="#237f8e", alpha=0.82)
    ax.axvline(0, color="#17202a", linewidth=1); ax.axvline(mean(errors), color="#d05a47", linestyle="--", label=@sprintf("mean = %.2f C", mean(errors)))
    ax.set(xlabel="Inferred minus natural Topt_ave (degrees C)", ylabel="Sequences", title="Temperature-recovery error")
    ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7); ax.legend(frameon=false)
    fig.savefig(path, dpi=300); close(fig)
end

function plot_likelihood_heatmap(path, likelihoods, true_t, grid)
    bins = temperature_bins(true_t, 10)
    profiles = fill(NaN, 10, length(grid))
    centers = zeros(10)
    for b in 1:10
        rows = findall(==(b), bins)
        profiles[b, :] .= vec(mean(likelihoods[rows, :]; dims=1)) ./ 62
        centers[b] = mean(true_t[rows])
    end
    fig, ax = subplots(figsize=(8.0, 5.2), constrained_layout=true)
    image = ax.imshow(profiles, origin="lower", aspect="auto", cmap="magma",
                      extent=[first(grid), last(grid), 0.5, 10.5])
    ax.set_yticks(1:10); ax.set_yticklabels([@sprintf("%.1f", c) for c in centers])
    ax.set(xlabel="Temperature used for likelihood evaluation (degrees C)",
           ylabel="Mean natural temperature in test decile (degrees C)", title="Mean held-out log-likelihood per site")
    fig.colorbar(image, ax=ax, label="Mean log-likelihood per site")
    fig.savefig(path, dpi=300); close(fig)
end

function correlation(x, y)
    return cor(Float64.(x), Float64.(y))
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
        rank = (first + last) / 2
        ranks[order[first:last]] .= rank
        first = last + 1
    end
    return ranks
end

spearman_correlation(x, y) = cor(tied_ranks(x), tied_ranks(y))

function main(args=ARGS)
    options = parse_args(args)
    metadata = read_metadata(options["metadata"])
    _, Z, L, M, q, annotations = FeatureDCA.read_annotated_fasta(options["fasta"]; remove_dups=false, max_gap_fraction=1.0)
    ids = sequence_id.(annotations)
    all(haskey(metadata, id) for id in ids) || error("Some FASTA records lack metadata")
    temperatures = [parse(Float64, metadata[id]["Topt_ave"]) for id in ids]
    taxonomy_ids = [metadata[id]["taxonomy_id"] for id in ids]
    train, test = organism_stratified_split(ids, temperatures, taxonomy_ids, options["test_fraction"], options["strata"], options["seed"])
    Z_train, Z_test = Z[:, train], Z[:, test]
    train_t, test_t = temperatures[train], temperatures[test]
    #weights, _ = FeatureDCA.compute_weights(Z_train, :auto)
    weights = ones(length(train_t))
    normalized_weights = weights ./ sum(weights)
    temperature_mean = sum(train_t .* normalized_weights)
    temperature_scale = sqrt(sum(((train_t .- temperature_mean) .^ 2) .* normalized_weights))
    Y_train = reshape((train_t .- temperature_mean) ./ temperature_scale, 1, :)

    output_dir = options["output_dir"]
    mkpath(output_dir)
    write_fasta(joinpath(output_dir, "train.fasta"), Z_train, ids[train], train_t)
    write_fasta(joinpath(output_dir, "test.fasta"), Z_test, ids[test], test_t)
    plot_split(joinpath(output_dir, "split_temperature_distribution.pdf"), train_t, test_t)
    open(joinpath(output_dir, "split.csv"), "w") do io
        println(io, "sequence_id,taxonomy_id,organism,Topt_ave,split")
        for (split_name, indices) in (("train", train), ("test", test)), i in indices
            println(io, join(csv_escape.([ids[i], taxonomy_ids[i], metadata[ids[i]]["organism"], temperatures[i], split_name]), ','))
        end
    end

    @printf("Training on %d sequences (%d organisms); testing on %d sequences (%d organisms).\n",
            length(train), length(unique(taxonomy_ids[train])), length(test), length(unique(taxonomy_ids[test])))
    @printf("Training-only standardization: mean=%.6f C, scale=%.6f C\n", temperature_mean, temperature_scale)
    started = time()
    network, arvar, objective = trainer(Z_train, Float64.(weights); Y=Y_train,
        lambdaH=options["lambda_h"], lambdaG=options["lambda_g"], lambdaJ=options["lambda_j"],
        epsconv=options["epsconv"], maxit=options["maxit"], verbose=options["verbose"])
    elapsed = time() - started

    grid_min = something(options["grid_min"], floor(minimum(temperatures) / options["grid_step"]) * options["grid_step"])
    grid_max = something(options["grid_max"], ceil(maximum(temperatures) / options["grid_step"]) * options["grid_step"])
    grid = collect(grid_min:options["grid_step"]:grid_max)
    standardized_grid = (grid .- temperature_mean) ./ temperature_scale
    println("Evaluating $(length(test)) held-out sequences at $(length(grid)) temperatures...")
    likelihoods = likelihood_grid(network, Z_test, standardized_grid)
    winners = [argmax(view(likelihoods, m, :)) for m in axes(likelihoods, 1)]
    inferred = grid[winners]
    errors = inferred .- test_t
    true_grid_index = [argmin(abs.(grid .- t)) for t in test_t]
    true_is_argmax = [true_grid_index[m] == winners[m] for m in eachindex(winners)]

    serialize(joinpath(output_dir, "model.jls"), (
        network=network, objective=objective, feature_names=["Topt_ave"],
        feature_means=[temperature_mean], feature_scales=[temperature_scale],
        train_sequence_ids=ids[train], test_sequence_ids=ids[test], weights=arvar.weights,
        alignment_length=L, alphabet_size=q, temperature_grid=grid, seed=options["seed"],
        elapsed_seconds=elapsed,
    ))
    open(joinpath(output_dir, "test_likelihoods.csv"), "w") do io
        header = ["sequence_id", "taxonomy_id", "natural_temperature", "inferred_temperature", "error", "absolute_error", "true_grid_is_argmax",
                  ["loglik_T" * replace(string(t), "." => "p") for t in grid]...]
        println(io, join(header, ','))
        for m in eachindex(test)
            values = Any[ids[test[m]], taxonomy_ids[test[m]], test_t[m], inferred[m], errors[m], abs(errors[m]), true_is_argmax[m], likelihoods[m, :]...]
            println(io, join(values, ','))
        end
    end
    plot_recovery(joinpath(output_dir, "temperature_recovery.pdf"), test_t, inferred, options["grid_step"])
    plot_error(joinpath(output_dir, "temperature_error.pdf"), errors)
    plot_likelihood_heatmap(joinpath(output_dir, "likelihood_heatmap.pdf"), likelihoods, test_t, grid)

    mae, rmse = mean(abs.(errors)), sqrt(mean(errors .^ 2))
    open(joinpath(output_dir, "summary.txt"), "w") do io
        @printf(io, "seed=%d\ntrain_sequences=%d\ntest_sequences=%d\ntrain_organisms=%d\ntest_organisms=%d\n", options["seed"], length(train), length(test), length(unique(taxonomy_ids[train])), length(unique(taxonomy_ids[test])))
        @printf(io, "temperature_grid_min=%.6f\ntemperature_grid_max=%.6f\ntemperature_grid_step=%.6f\n", grid_min, grid_max, options["grid_step"])
        @printf(io, "training_temperature_mean=%.10f\ntraining_temperature_scale=%.10f\ntraining_seconds=%.3f\n", temperature_mean, temperature_scale, elapsed)
        @printf(io, "recovery_correlation=%.10f\nrecovery_pearson=%.10f\nrecovery_spearman=%.10f\nmae_C=%.10f\nrmse_C=%.10f\nbias_C=%.10f\n",
                correlation(test_t, inferred), correlation(test_t, inferred), spearman_correlation(test_t, inferred), mae, rmse, mean(errors))
        @printf(io, "true_grid_exact_argmax_fraction=%.10f\nwithin_1C_fraction=%.10f\nwithin_2C_fraction=%.10f\nwithin_5C_fraction=%.10f\n",
                mean(true_is_argmax), mean(abs.(errors) .<= 1), mean(abs.(errors) .<= 2), mean(abs.(errors) .<= 5))
    end
    @printf("Done. MAE %.3f C, RMSE %.3f C, correlation %.3f. Results: %s\n", mae, rmse, correlation(test_t, inferred), output_dir)
end

main()
