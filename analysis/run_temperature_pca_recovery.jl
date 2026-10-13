#!/usr/bin/env julia

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

using FeatureDCA
using MultivariateStats
using Printf
using PyPlot
using Random
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const METADATA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.metadata.maxgap50.csv")
const PCA_ARTIFACT = joinpath(ROOT, "results", "featuredca", "jdomain_pca2.jls")
const TEMPERATURE_ARTIFACT = joinpath(ROOT, "results", "featuredca", "jdomain_topt_ave.jls")
const SPLIT_ARTIFACT = joinpath(ROOT, "results", "featuredca", "jdomain_natural_temperature_recovery", "model.jls")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_temperature_pca_recovery")
const AA = collect("ACDEFGHIKLMNPQRSTVWY-")

function parse_args(args)
    options = Dict{String,Any}(
        "mode" => "all", "output_dir" => DEFAULT_OUTPUT, "seed" => 31,
        "grid_step" => 0.5, "lambda_h" => 0.01, "lambda_g" => 0.01,
        "lambda_j" => 0.01, "epsconv" => 1e-5, "maxit" => 1000,
        "verbose" => true,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/run_temperature_pca_recovery.jl [--mode all|full|test|baseline] [options]")
            exit(0)
        elseif arg == "--quiet"
            options["verbose"] = false
        elseif arg in ("--mode", "--output-dir", "--seed", "--grid-step", "--lambda-h", "--lambda-g", "--lambda-j", "--epsconv", "--maxit")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key in ("seed", "maxit") ? parse(Int, value) :
                           key in ("grid_step", "lambda_h", "lambda_g", "lambda_j", "epsconv") ? parse(Float64, value) :
                           key == "output_dir" ? abspath(value) : value
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    options["mode"] in ("all", "full", "test", "baseline") || error("--mode must be all, full, test, or baseline")
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

function weighted_standardize(Y, weights)
    normalized = weights ./ sum(weights)
    means = vec(Y * normalized)
    centered = Y .- means
    scales = sqrt.(vec((centered .^ 2) * normalized))
    return centered ./ scales, means, scales
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

function recovery_metrics(actual, inferred)
    errors = inferred .- actual
    return (pearson=cor(actual, inferred), spearman=spearman(actual, inferred),
            mae=mean(abs.(errors)), rmse=sqrt(mean(errors .^ 2)), bias=mean(errors),
            within1=mean(abs.(errors) .<= 1), within2=mean(abs.(errors) .<= 2),
            within5=mean(abs.(errors) .<= 5))
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
        @inbounds for previous in 1:site-1, state in eachindex(logits)
            logits[state] += couplings[previous, state, sequence[previous]]
        end
        total += logits[sequence[site]] - logsumexp(logits)
    end
    return total
end

function likelihood_grid(network, sequences, grid, feature_means, feature_scales, pc_values)
    n_sequences, n_temperatures = size(sequences, 2), length(grid)
    n_pcs = size(pc_values, 1)
    length(feature_means) == n_pcs + 1 || error("Feature and PC dimensions do not match")
    likelihoods = Matrix{Float64}(undef, n_sequences, n_temperatures)
    Threads.@threads for m in 1:n_sequences
        y = Vector{Float64}(undef, n_pcs + 1)
        for p in 1:n_pcs
            y[p + 1] = (pc_values[p, m] - feature_means[p + 1]) / feature_scales[p + 1]
        end
        for t in 1:n_temperatures
            y[1] = (grid[t] - feature_means[1]) / feature_scales[1]
            likelihoods[m, t] = sequence_loglikelihood(network, view(sequences, :, m), y)
        end
    end
    return likelihoods
end

function likelihood_grid_temperature_only(network, sequences, grid, temperature_mean, temperature_scale)
    likelihoods = Matrix{Float64}(undef, size(sequences, 2), length(grid))
    Threads.@threads for m in axes(sequences, 2)
        for t in eachindex(grid)
            y = [(grid[t] - temperature_mean) / temperature_scale]
            likelihoods[m, t] = sequence_loglikelihood(network, view(sequences, :, m), y)
        end
    end
    return likelihoods
end

infer(likelihoods, grid) = [grid[argmax(view(likelihoods, m, :))] for m in axes(likelihoods, 1)]

function fit_network(Z, Y, weights, options)
    started = time()
    network, arvar, objective = trainer(
        Z, Float64.(weights); Y=Matrix{Float64}(Y),
        lambdaH=options["lambda_h"], lambdaG=options["lambda_g"], lambdaJ=options["lambda_j"],
        epsconv=options["epsconv"], maxit=options["maxit"], verbose=options["verbose"])
    return network, arvar, objective, time() - started
end

function temperature_grid(temperatures, step)
    low = floor(minimum(temperatures) / step) * step
    high = ceil(maximum(temperatures) / step) * step
    return collect(low:step:high)
end

function profile_groups(temperatures, likelihoods, grid; width=10.0)
    centers = collect(width:width:ceil(maximum(temperatures) / width) * width)
    assignments = [centers[argmin(abs.(centers .- t))] for t in temperatures]
    result = NamedTuple[]
    for center in centers
        rows = findall(==(center), assignments)
        isempty(rows) && continue
        profile = vec(mean(likelihoods[rows, :]; dims=1)) ./ 62
        push!(result, (center=center, n=length(rows), mean_temperature=mean(temperatures[rows]),
                       profile=profile, peak=grid[argmax(profile)]))
    end
    return result
end

function plot_profile_comparison(path, groups_list, labels, grid, title)
    fig, axes = subplots(1, length(groups_list), figsize=(6.5 * length(groups_list), 5.2),
                         constrained_layout=true, sharey=true)
    length(groups_list) == 1 && (axes = [axes])
    colors = PyPlot.cm.viridis(range(0.05, 0.95, length=length(groups_list[1])))
    for (ax, groups, panel_title) in zip(axes, groups_list, labels)
        for (i, group) in enumerate(groups)
            ax.plot(grid, group.profile, linewidth=1.7, color=colors[i, :],
                    label=@sprintf("%.0f °C (n=%d, peak %.1f)", group.center, group.n, group.peak))
            peak_index = argmax(group.profile)
            ax.scatter([grid[peak_index]], [group.profile[peak_index]], s=24,
                       color=colors[i:i, :], edgecolors="white", linewidths=0.4, zorder=3)
        end
        ax.set_xlabel("Temperature used for likelihood evaluation (degrees C)")
        ax.set_title(panel_title)
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
        ax.legend(frameon=false, fontsize=7.5)
    end
    axes[1].set_ylabel("Mean log-likelihood per site")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=300)
    close(fig)
end

function plot_recovery(path, actual, inferred_list, labels, title)
    fig, axes = subplots(1, length(inferred_list), figsize=(6.0 * length(inferred_list), 5.2),
                         constrained_layout=true, sharex=true, sharey=true)
    length(inferred_list) == 1 && (axes = [axes])
    limits = extrema(vcat(actual, inferred_list...))
    for (ax, inferred, label) in zip(axes, inferred_list, labels)
        metric = recovery_metrics(actual, inferred)
        image = ax.hexbin(actual, inferred, gridsize=34, mincnt=1, bins="log", cmap="viridis")
        ax.plot(limits, limits, "--", color="#d05a47", linewidth=1.2)
        ax.set_xlabel("Reference temperature (degrees C)")
        ax.set_title(@sprintf("%s\nr=%.3f, ρ=%.3f, MAE=%.2f °C", label, metric.pearson, metric.spearman, metric.mae))
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.6)
        fig.colorbar(image, ax=ax, label="log10 count")
    end
    axes[1].set_ylabel("Maximum-likelihood temperature (degrees C)")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=300)
    close(fig)
end

function write_generated_fasta(path, Z, ids, temperatures, original_pc, resulting_pc)
    open(path, "w") do io
        for m in axes(Z, 2)
            @printf(io, ">%s|generated T=%.6g original_PC1=%.8g original_PC2=%.8g resulting_PC1=%.8g resulting_PC2=%.8g\n",
                    ids[m], temperatures[m], original_pc[1, m], original_pc[2, m], resulting_pc[1, m], resulting_pc[2, m])
            sequence = join(AA[Z[:, m]])
            for start in 1:80:length(sequence)
                println(io, sequence[start:min(start + 79, end)])
            end
        end
    end
end

function print_metrics(io, prefix, actual, inferred)
    m = recovery_metrics(actual, inferred)
    @printf(io, "%s_n=%d\n%s_pearson=%.10f\n%s_spearman=%.10f\n", prefix, length(actual), prefix, m.pearson, prefix, m.spearman)
    @printf(io, "%s_mae_C=%.10f\n%s_rmse_C=%.10f\n%s_bias_C=%.10f\n", prefix, m.mae, prefix, m.rmse, prefix, m.bias)
    @printf(io, "%s_within_1C=%.10f\n%s_within_2C=%.10f\n%s_within_5C=%.10f\n", prefix, m.within1, prefix, m.within2, prefix, m.within5)
end

function save_likelihoods(path, ids, temperatures, inferred_list, labels, likelihoods_list, grid)
    open(path, "w") do io
        columns = Any["sequence_id", "reference_temperature"]
        append!(columns, ["inferred_" * label for label in labels])
        for (label, _) in zip(labels, likelihoods_list), t in grid
            push!(columns, "loglik_" * label * "_T" * replace(string(t), "." => "p"))
        end
        println(io, join(columns, ','))
        for m in eachindex(ids)
            values = Any[ids[m], temperatures[m]]
            append!(values, [inferred[m] for inferred in inferred_list])
            for likelihoods in likelihoods_list
                append!(values, likelihoods[m, :])
            end
            println(io, join(values, ','))
        end
    end
end

function run_full(Z, ids, temperatures, weights, options)
    output = joinpath(options["output_dir"], "full_generated")
    mkpath(output)
    pca_artifact = deserialize(PCA_ARTIFACT)
    ids == pca_artifact.sequence_ids || error("Full PCA artifact sequence order differs from FASTA")
    pca_raw = Matrix{Float64}(pca_artifact.Y)
    Y_raw = vcat(reshape(temperatures, 1, :), pca_raw)
    Y, means, scales = weighted_standardize(Y_raw, weights)
    println("Training full-data FeatureDCA with temperature, PC1, and PC2...")
    network, arvar, objective, elapsed = fit_network(Z, Y, weights, options)

    Random.seed!(options["seed"])
    println("Generating one sequence per natural (temperature, PC1, PC2) coordinate...")
    generated = FeatureDCA.sample(network, Y)
    generated_pc = Matrix{Float64}(predict(pca_artifact.pca_map, FeatureDCA.one_hot(generated; q=21)))
    grid = temperature_grid(temperatures, options["grid_step"])
    println("Scanning generated sequences with original PCs...")
    likelihood_original = likelihood_grid(network, generated, grid, means, scales, pca_raw)
    println("Scanning generated sequences with resulting generated PCs...")
    likelihood_resulting = likelihood_grid(network, generated, grid, means, scales, generated_pc)
    inferred_original = infer(likelihood_original, grid)
    inferred_resulting = infer(likelihood_resulting, grid)

    artifact = (network=network, objective=objective, pca_map=pca_artifact.pca_map,
                feature_names=["Topt_ave", "PC1", "PC2"], feature_means=means, feature_scales=scales,
                sequence_ids=ids, weights=arvar.weights, alignment_length=62, alphabet_size=21,
                elapsed_seconds=elapsed, seed=options["seed"])
    serialize(joinpath(output, "model.jls"), artifact)
    write_generated_fasta(joinpath(output, "generated.fasta"), generated, ids, temperatures, pca_raw, generated_pc)
    save_likelihoods(joinpath(output, "likelihoods.csv"), ids, temperatures,
                     [inferred_original, inferred_resulting], ["original_pc", "resulting_pc"],
                     [likelihood_original, likelihood_resulting], grid)
    plot_recovery(joinpath(output, "temperature_recovery.pdf"), temperatures,
                  [inferred_original, inferred_resulting], ["Original conditioning PCs", "Generated sequence PCs"],
                  "Generated-sequence temperature recovery with PC1 and PC2 fixed")
    groups_original = profile_groups(temperatures, likelihood_original, grid)
    groups_resulting = profile_groups(temperatures, likelihood_resulting, grid)
    plot_profile_comparison(joinpath(output, "likelihood_profiles.pdf"),
                            [groups_original, groups_resulting], ["Original conditioning PCs", "Generated sequence PCs"],
                            grid, "Generated-sequence likelihood profiles with temperature plus PCA conditioning")
    open(joinpath(output, "summary.txt"), "w") do io
        @printf(io, "training_sequences=%d\ntraining_seconds=%.3f\ngrid_step_C=%.6f\n", length(ids), elapsed, options["grid_step"])
        print_metrics(io, "original_pc", temperatures, inferred_original)
        print_metrics(io, "resulting_pc", temperatures, inferred_resulting)
        @printf(io, "PC1_original_resulting_correlation=%.10f\nPC2_original_resulting_correlation=%.10f\n",
                cor(pca_raw[1, :], generated_pc[1, :]), cor(pca_raw[2, :], generated_pc[2, :]))
    end
end

function run_test(Z, ids, temperatures, weights_all, options)
    output = joinpath(options["output_dir"], "heldout_natural")
    mkpath(output)
    split = deserialize(SPLIT_ARTIFACT)
    train_lookup, test_lookup = Set(split.train_sequence_ids), Set(split.test_sequence_ids)
    train = findall(id -> id in train_lookup, ids)
    test = findall(id -> id in test_lookup, ids)
    length(train) == length(train_lookup) || error("Could not recover all training IDs")
    length(test) == length(test_lookup) || error("Could not recover all test IDs")
    Z_train, Z_test = Z[:, train], Z[:, test]
    weights, _ = FeatureDCA.compute_weights(Z_train, :auto)
    println("Fitting leakage-free PCA on $(length(train)) training sequences...")
    encoded_train = FeatureDCA.one_hot(Z_train; q=21)
    pca_map = fit(PCA, encoded_train, maxoutdim=2)
    pc_train = Matrix{Float64}(predict(pca_map, encoded_train))
    pc_test = Matrix{Float64}(predict(pca_map, FeatureDCA.one_hot(Z_test; q=21)))
    Y_raw = vcat(reshape(temperatures[train], 1, :), pc_train)
    Y, means, scales = weighted_standardize(Y_raw, weights)
    println("Training held-out FeatureDCA with temperature, PC1, and PC2...")
    network, arvar, objective, elapsed = fit_network(Z_train, Y, weights, options)
    grid = temperature_grid(temperatures, options["grid_step"])
    println("Scanning held-out natural sequences with their training-PCA projections...")
    likelihoods = likelihood_grid(network, Z_test, grid, means, scales, pc_test)
    inferred = infer(likelihoods, grid)

    artifact = (network=network, objective=objective, pca_map=pca_map,
                feature_names=["Topt_ave", "PC1", "PC2"], feature_means=means, feature_scales=scales,
                train_sequence_ids=ids[train], test_sequence_ids=ids[test], weights=arvar.weights,
                alignment_length=62, alphabet_size=21, elapsed_seconds=elapsed, seed=options["seed"])
    serialize(joinpath(output, "model.jls"), artifact)
    save_likelihoods(joinpath(output, "likelihoods.csv"), ids[test], temperatures[test],
                     [inferred], ["test_sequence_pc"], [likelihoods], grid)
    plot_recovery(joinpath(output, "temperature_recovery.pdf"), temperatures[test], [inferred],
                  ["Test sequence PCs"], "Held-out natural temperature recovery with PC1 and PC2 fixed")
    groups = profile_groups(temperatures[test], likelihoods, grid)
    plot_profile_comparison(joinpath(output, "likelihood_profiles.pdf"), [groups], ["Held-out natural sequences"],
                            grid, "Natural test likelihood profiles with temperature plus PCA conditioning")
    open(joinpath(output, "summary.txt"), "w") do io
        @printf(io, "training_sequences=%d\ntest_sequences=%d\ntraining_seconds=%.3f\ngrid_step_C=%.6f\n",
                length(train), length(test), elapsed, options["grid_step"])
        print_metrics(io, "test", temperatures[test], inferred)
    end
end

function run_baseline(Z, ids, temperatures, options)
    output = joinpath(options["output_dir"], "full_generated_temperature_only_baseline")
    mkpath(output)
    artifact = deserialize(TEMPERATURE_ARTIFACT)
    ids == artifact.sequence_ids || error("Temperature-only artifact sequence order differs from FASTA")
    standardized = (temperatures .- only(artifact.feature_means)) ./ only(artifact.feature_scales)
    Random.seed!(options["seed"])
    println("Generating matched temperature-only baseline sequences...")
    generated = FeatureDCA.sample(artifact.network, reshape(standardized, 1, :))
    grid = temperature_grid(temperatures, options["grid_step"])
    println("Scanning matched temperature-only baseline sequences...")
    likelihoods = likelihood_grid_temperature_only(artifact.network, generated, grid,
                                                     only(artifact.feature_means), only(artifact.feature_scales))
    inferred = infer(likelihoods, grid)
    save_likelihoods(joinpath(output, "likelihoods.csv"), ids, temperatures,
                     [inferred], ["temperature_only"], [likelihoods], grid)
    plot_recovery(joinpath(output, "temperature_recovery.pdf"), temperatures, [inferred],
                  ["Temperature only"], "Generated-sequence temperature recovery baseline")
    groups = profile_groups(temperatures, likelihoods, grid)
    plot_profile_comparison(joinpath(output, "likelihood_profiles.pdf"), [groups], ["Temperature only"],
                            grid, "Generated-sequence likelihood profiles under temperature-only conditioning")
    open(joinpath(output, "summary.txt"), "w") do io
        @printf(io, "sequences=%d\ngrid_step_C=%.6f\nseed=%d\n", length(ids), options["grid_step"], options["seed"])
        print_metrics(io, "temperature_only", temperatures, inferred)
    end
end

function main(args=ARGS)
    options = parse_args(args)
    metadata = read_metadata(METADATA)
    weights, Z, _, M, _, annotations = FeatureDCA.read_annotated_fasta(FASTA; remove_dups=false, max_gap_fraction=1.0)
    ids = sequence_id.(annotations)
    temperatures = [parse(Float64, metadata[id]["Topt_ave"]) for id in ids]
    length(ids) == M || error("FASTA parsing mismatch")
    mkpath(options["output_dir"])
    options["mode"] in ("all", "full") && run_full(Z, ids, temperatures, weights, options)
    options["mode"] in ("all", "test") && run_test(Z, ids, temperatures, weights, options)
    options["mode"] in ("all", "baseline") && run_baseline(Z, ids, temperatures, options)
    println("Combined temperature plus PCA analysis written to $(options["output_dir"])")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
