#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_temperature_pca_recovery.jl"))

const CV_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_temperature_fixedbin_cv5")
const BIN_CENTERS = collect(10.0:10.0:80.0)

function cv_args(args)
    options = Dict{String,Any}(
        "output_dir" => CV_OUTPUT, "folds" => 5, "seed" => 47, "grid_step" => 0.5,
        "lambda_h" => 0.01, "lambda_g" => 0.01, "lambda_j" => 0.01,
        "epsconv" => 1e-5, "maxit" => 1000, "verbose" => false, "resume" => true)
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/crossvalidate_fixed_temperature_bins.jl [--output-dir PATH] [--seed 47] [--no-resume]")
            exit(0)
        elseif arg == "--verbose"
            options["verbose"] = true
        elseif arg == "--no-resume"
            options["resume"] = false
        elseif arg in ("--output-dir", "--folds", "--seed", "--grid-step", "--maxit")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key in ("folds", "seed", "maxit") ? parse(Int, value) :
                           key == "grid_step" ? parse(Float64, value) : abspath(value)
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return options
end

fixed_bin_index(t) = clamp(floor(Int, (t - 5) / 10) + 1, 1, length(BIN_CENTERS))

function build_folds(taxonomy, temperatures, nfolds, seed)
    organism_rows = Dict{String,Vector{Int}}()
    for i in eachindex(taxonomy)
        push!(get!(organism_rows, taxonomy[i], Int[]), i)
    end
    by_bin = [String[] for _ in BIN_CENTERS]
    for (organism, rows) in organism_rows
        values = unique(temperatures[rows])
        length(values) == 1 || error("Organism $organism has multiple temperatures")
        push!(by_bin[fixed_bin_index(only(values))], organism)
    end
    all(length(group) >= nfolds for group in by_bin) || error("Every bin must contain at least $nfolds organisms")

    rng = MersenneTwister(seed)
    organism_fold = Dict{String,Int}()
    for organisms in by_bin
        shuffled = organisms[randperm(rng, length(organisms))]
        ordered = sort(shuffled; by=id -> -length(organism_rows[id]), alg=MergeSort)
        loads = zeros(Int, nfolds)
        for organism in ordered
            minimum_load = minimum(loads)
            candidates = findall(==(minimum_load), loads)
            fold = rand(rng, candidates)
            organism_fold[organism] = fold
            loads[fold] += length(organism_rows[organism])
        end
    end
    sequence_fold = [organism_fold[id] for id in taxonomy]
    return sequence_fold, organism_fold
end

function safe_cor(x, y)
    std(x) <= eps() || std(y) <= eps() ? NaN : cor(x, y)
end

function detailed_metrics(actual, inferred, grid)
    errors = inferred .- actual
    nearest = [grid[argmin(abs.(grid .- t))] for t in actual]
    return (n=length(actual), pearson=safe_cor(actual, inferred), spearman=safe_cor(tied_ranks(actual), tied_ranks(inferred)),
            mae=mean(abs.(errors)), rmse=sqrt(mean(errors .^ 2)), bias=mean(errors),
            exact=mean(inferred .== nearest), within1=mean(abs.(errors) .<= 1),
            within2=mean(abs.(errors) .<= 2), within5=mean(abs.(errors) .<= 5))
end

function aggregate_organisms(taxonomy, temperatures, likelihoods)
    organisms = unique(taxonomy)
    actual = Vector{Float64}(undef, length(organisms))
    profiles = Matrix{Float64}(undef, length(organisms), size(likelihoods, 2))
    counts = Vector{Int}(undef, length(organisms))
    for (i, organism) in enumerate(organisms)
        rows = findall(==(organism), taxonomy)
        values = unique(temperatures[rows])
        length(values) == 1 || error("Organism $organism has multiple temperatures")
        actual[i] = only(values)
        profiles[i, :] .= vec(mean(likelihoods[rows, :]; dims=1))
        counts[i] = length(rows)
    end
    return organisms, actual, profiles, counts
end

function write_fold_assignment(path, ids, taxonomy, temperatures, sequence_fold)
    open(path, "w") do io
        println(io, "sequence_id,taxonomy_id,Topt_ave,fixed_bin_center,fold")
        for i in eachindex(ids)
            @printf(io, "%s,%s,%.6f,%.0f,%d\n", ids[i], taxonomy[i], temperatures[i],
                    BIN_CENTERS[fixed_bin_index(temperatures[i])], sequence_fold[i])
        end
    end
end

function write_fold_counts(path, taxonomy, temperatures, sequence_fold, nfolds)
    open(path, "w") do io
        println(io, "fold,bucket_center,sequences,organisms")
        for fold in 1:nfolds, bin in eachindex(BIN_CENTERS)
            rows = findall(i -> sequence_fold[i] == fold && fixed_bin_index(temperatures[i]) == bin, eachindex(temperatures))
            @printf(io, "%d,%.0f,%d,%d\n", fold, BIN_CENTERS[bin], length(rows), length(unique(taxonomy[rows])))
        end
    end
end

function write_compact_recovery(path, ids, reference, inferred_list, labels)
    open(path, "w") do io
        println(io, join(["id", "reference_temperature", ["inferred_" * label for label in labels]...], ','))
        for m in eachindex(ids)
            println(io, join(Any[ids[m], reference[m], [inferred[m] for inferred in inferred_list]...], ','))
        end
    end
end

function write_profile_summary(path, groups_list, labels)
    open(path, "w") do io
        println(io, "unit,bucket_center,n,mean_reference_temperature,mean_profile_argmax")
        for (label, groups) in zip(labels, groups_list), group in groups
            @printf(io, "%s,%.6f,%d,%.6f,%.6f\n", label, group.center, group.n,
                    group.mean_temperature, group.peak)
        end
    end
end

function write_metrics(path, sequence_t, sequence_fine, sequence_coarse, organism_t, organism_fine, organism_coarse,
                       fine_grid, coarse_grid)
    open(path, "w") do io
        println(io, "unit,grid,bin_center,n,pearson,spearman,mae_C,rmse_C,bias_C,exact_fraction,within_1C,within_2C,within_5C")
        for (unit, actual, fine, coarse) in (("sequence", sequence_t, sequence_fine, sequence_coarse),
                                             ("organism", organism_t, organism_fine, organism_coarse))
            for (grid_name, inferred, grid) in (("0.5C", fine, fine_grid), ("10C", coarse, coarse_grid))
                for (bin_label, rows) in vcat([(0.0, collect(eachindex(actual)))],
                                               [(BIN_CENTERS[b], findall(t -> fixed_bin_index(t) == b, actual)) for b in eachindex(BIN_CENTERS)])
                    isempty(rows) && continue
                    m = detailed_metrics(actual[rows], inferred[rows], grid)
                    @printf(io, "%s,%s,%.0f,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                            unit, grid_name, bin_label, m.n, m.pearson, m.spearman, m.mae, m.rmse, m.bias,
                            m.exact, m.within1, m.within2, m.within5)
                end
            end
        end
    end
end

function write_macro_metrics(path, sequence_t, sequence_fine, sequence_coarse, organism_t, organism_fine, organism_coarse,
                             fine_grid, coarse_grid)
    open(path, "w") do io
        println(io, "unit,grid,macro_mae_C,macro_rmse_C,macro_exact_fraction,macro_within_1C,macro_within_2C,macro_within_5C")
        for (unit, actual, fine, coarse) in (("sequence", sequence_t, sequence_fine, sequence_coarse),
                                             ("organism", organism_t, organism_fine, organism_coarse))
            for (grid_name, inferred, grid) in (("0.5C", fine, fine_grid), ("10C", coarse, coarse_grid))
                metrics = [detailed_metrics(actual[findall(t -> fixed_bin_index(t) == b, actual)],
                                            inferred[findall(t -> fixed_bin_index(t) == b, actual)], grid)
                           for b in eachindex(BIN_CENTERS)]
                @printf(io, "%s,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n", unit, grid_name,
                        mean(m.mae for m in metrics), mean(m.rmse for m in metrics), mean(m.exact for m in metrics),
                        mean(m.within1 for m in metrics), mean(m.within2 for m in metrics), mean(m.within5 for m in metrics))
            end
        end
    end
end

function fixed_profile_summaries(temperatures, likelihoods, grid)
    summaries = NamedTuple[]
    for bin in eachindex(BIN_CENTERS)
        rows = findall(t -> fixed_bin_index(t) == bin, temperatures)
        profile = vec(mean(likelihoods[rows, :]; dims=1)) ./ 62
        push!(summaries, (center=BIN_CENTERS[bin], n=length(rows), mean_temperature=mean(temperatures[rows]),
                          profile=profile, peak=grid[argmax(profile)]))
    end
    return summaries
end

function draw_fixed_heatmap(path, summaries, grid; relative=false)
    values = reduce(vcat, [reshape(s.profile, 1, :) for s in summaries])
    if relative
        values .-= maximum(values; dims=2)
    end
    fig, ax = subplots(figsize=(10.2, 5.8))
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.14, top=0.90)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=relative ? "viridis" : "magma",
                      extent=[first(grid), last(grid), 0.5, length(summaries) + 0.5])
    labels = [@sprintf("%.0f °C   n=%d", s.center, s.n) for s in summaries]
    ax.set_yticks(1:length(summaries)); ax.set_yticklabels(labels)
    ax.set_xlabel("Temperature used for likelihood evaluation (degrees C)")
    ax.set_ylabel("Natural Topt_ave bin")
    ax.set_title(relative ? "Five-fold out-of-fold profiles relative to each row maximum" :
                            "Five-fold out-of-fold mean log-likelihood profiles")
    for (row, summary) in enumerate(summaries)
        ax.scatter([summary.peak], [row], marker="s", s=42, facecolors="none", edgecolors="cyan", linewidths=1.2)
    end
    fig.colorbar(image, ax=ax, label=relative ? "Log-likelihood per site relative to row maximum" : "Mean log-likelihood per site")
    fig.savefig(path, dpi=300); close(fig)
end

function plot_fold_distribution(path, taxonomy, temperatures, sequence_fold, nfolds)
    counts = zeros(Int, length(BIN_CENTERS), nfolds)
    for i in eachindex(temperatures)
        counts[fixed_bin_index(temperatures[i]), sequence_fold[i]] += 1
    end
    fig, ax = subplots(figsize=(9.2, 5.4), constrained_layout=true)
    bottom = zeros(Int, length(BIN_CENTERS))
    colors = PyPlot.cm.Set2(range(0, 1, length=nfolds))
    for fold in 1:nfolds
        ax.bar(BIN_CENTERS, counts[:, fold], bottom=bottom, width=7.5, color=colors[fold, :], label="Fold $fold")
        bottom .+= counts[:, fold]
    end
    ax.set_yscale("log")
    ax.set_xticks(BIN_CENTERS)
    ax.set_xlabel("Natural Topt_ave bin (degrees C)")
    ax.set_ylabel("Sequences, logarithmic scale")
    ax.set_title("Fixed-bin stratified organism-level folds")
    ax.legend(frameon=false, ncol=5)
    ax.grid(true, axis="y", color="#d8dde3", linewidth=0.5, alpha=0.7)
    fig.savefig(path, dpi=300); close(fig)
end

function main_cv(args=ARGS)
    options = cv_args(args)
    mkpath(options["output_dir"])
    metadata = read_metadata(METADATA)
    _, Z, L, M, q, annotations = FeatureDCA.read_annotated_fasta(FASTA; remove_dups=false, max_gap_fraction=1.0)
    ids = sequence_id.(annotations)
    temperatures = [parse(Float64, metadata[id]["Topt_ave"]) for id in ids]
    taxonomy = [metadata[id]["taxonomy_id"] for id in ids]
    sequence_fold, _ = build_folds(taxonomy, temperatures, options["folds"], options["seed"])
    write_fold_assignment(joinpath(options["output_dir"], "fold_assignments.csv"), ids, taxonomy, temperatures, sequence_fold)
    write_fold_counts(joinpath(options["output_dir"], "fold_counts.csv"), taxonomy, temperatures, sequence_fold, options["folds"])
    plot_fold_distribution(joinpath(options["output_dir"], "fold_distribution.pdf"), taxonomy, temperatures, sequence_fold, options["folds"])

    grid = temperature_grid(temperatures, options["grid_step"])
    pooled = Matrix{Float64}(undef, M, length(grid))
    fold_times = zeros(options["folds"])
    for fold in 1:options["folds"]
        checkpoint = joinpath(options["output_dir"], "fold$(fold)_likelihoods.jls")
        if options["resume"] && isfile(checkpoint)
            saved = deserialize(checkpoint)
            pooled[saved.test_indices, :] .= saved.likelihoods
            fold_times[fold] = saved.training_seconds
            println("Fold $fold checkpoint loaded")
            continue
        end
        test = findall(==(fold), sequence_fold)
        train = findall(!=(fold), sequence_fold)
        weights, _ = FeatureDCA.compute_weights(Z[:, train], :auto)
        normalized = weights ./ sum(weights)
        mean_t = sum(temperatures[train] .* normalized)
        scale_t = sqrt(sum(((temperatures[train] .- mean_t) .^ 2) .* normalized))
        Y = reshape((temperatures[train] .- mean_t) ./ scale_t, 1, :)
        println("Fold $fold: training on $(length(train)) sequences, testing on $(length(test))...")
        network, arvar, objective, elapsed = fit_network(Z[:, train], Y, weights, options)
        likelihoods = likelihood_grid_temperature_only(network, Z[:, test], grid, mean_t, scale_t)
        pooled[test, :] .= likelihoods
        fold_times[fold] = elapsed
        serialize(joinpath(options["output_dir"], "fold$(fold)_model.jls"), (
            network=network, objective=objective, feature_mean=mean_t, feature_scale=scale_t,
            train_sequence_ids=ids[train], test_sequence_ids=ids[test], weights=arvar.weights,
            training_seconds=elapsed, seed=options["seed"], fold=fold))
        serialize(checkpoint, (test_indices=test, likelihoods=likelihoods, training_seconds=elapsed))
    end

    fine_inferred = infer(pooled, grid)
    coarse_grid = collect(10.0:10.0:80.0)
    coarse_columns = [argmin(abs.(grid .- t)) for t in coarse_grid]
    coarse_likelihoods = pooled[:, coarse_columns]
    coarse_inferred = infer(coarse_likelihoods, coarse_grid)
    organisms, organism_t, organism_profiles, homolog_counts = aggregate_organisms(taxonomy, temperatures, pooled)
    organism_fine = infer(organism_profiles, grid)
    organism_coarse = infer(organism_profiles[:, coarse_columns], coarse_grid)

    write_compact_recovery(joinpath(options["output_dir"], "out_of_fold_recovery.csv"), ids, temperatures,
                           [fine_inferred, coarse_inferred], ["fine_0p5C", "coarse_10C"])
    write_compact_recovery(joinpath(options["output_dir"], "out_of_fold_organism_recovery.csv"), organisms, organism_t,
                           [organism_fine, organism_coarse], ["fine_0p5C", "coarse_10C"])
    write_metrics(joinpath(options["output_dir"], "metrics.csv"), temperatures, fine_inferred, coarse_inferred,
                  organism_t, organism_fine, organism_coarse, grid, coarse_grid)
    write_macro_metrics(joinpath(options["output_dir"], "macro_metrics.csv"), temperatures, fine_inferred, coarse_inferred,
                        organism_t, organism_fine, organism_coarse, grid, coarse_grid)
    sequence_groups = fixed_profile_summaries(temperatures, pooled, grid)
    organism_groups = fixed_profile_summaries(organism_t, organism_profiles, grid)
    write_profile_summary(joinpath(options["output_dir"], "likelihood_profile_summary.csv"),
                          [sequence_groups, organism_groups], ["sequence", "organism"])
    draw_fixed_heatmap(joinpath(options["output_dir"], "likelihood_heatmap.pdf"), sequence_groups, grid)
    draw_fixed_heatmap(joinpath(options["output_dir"], "likelihood_heatmap_relative.pdf"), sequence_groups, grid; relative=true)
    plot_profile_comparison(joinpath(options["output_dir"], "likelihood_profiles.pdf"),
                            [sequence_groups, organism_groups], ["Individual sequences", "Homologs combined by organism"],
                            grid, "Five-fold out-of-fold conditional likelihood profiles")
    plot_recovery(joinpath(options["output_dir"], "temperature_recovery.pdf"), temperatures,
                  [fine_inferred, coarse_inferred], ["0.5 °C grid", "10 °C grid"],
                  "Five-fold out-of-fold natural temperature recovery")
    open(joinpath(options["output_dir"], "summary.txt"), "w") do io
        @printf(io, "folds=%d\nseed=%d\nsequences=%d\norganisms=%d\nalignment_length=%d\n", options["folds"], options["seed"], M, length(organisms), L)
        @printf(io, "training_seconds_total=%.3f\nmean_homologs_per_organism=%.10f\n", sum(fold_times), mean(homolog_counts))
        for (prefix, actual, inferred, metric_grid) in (("sequence_fine", temperatures, fine_inferred, grid),
                                                        ("sequence_coarse", temperatures, coarse_inferred, coarse_grid),
                                                        ("organism_fine", organism_t, organism_fine, grid),
                                                        ("organism_coarse", organism_t, organism_coarse, coarse_grid))
            m = detailed_metrics(actual, inferred, metric_grid)
            @printf(io, "%s_pearson=%.10f\n%s_spearman=%.10f\n%s_mae_C=%.10f\n%s_rmse_C=%.10f\n%s_exact_fraction=%.10f\n",
                    prefix, m.pearson, prefix, m.spearman, prefix, m.mae, prefix, m.rmse, prefix, m.exact)
        end
    end
    println("Five-fold cross-validation written to $(options["output_dir"])")
end

main_cv()
