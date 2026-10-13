#!/usr/bin/env julia

include(joinpath(@__DIR__, "run_temperature_pca_recovery.jl"))

const SWEEP_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_temperature_pca_dimension_sweep")

function sweep_args(args)
    options = Dict{String,Any}(
        "pc_counts" => [2, 4, 8, 16, 32, 64], "mode" => "all", "output_dir" => SWEEP_OUTPUT,
        "seed" => 31, "grid_step" => 0.5, "lambda_h" => 0.01, "lambda_g" => 0.01,
        "lambda_j" => 0.01, "epsconv" => 1e-5, "maxit" => 1000,
        "verbose" => false, "resume" => true,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/sweep_temperature_pca_dimensions.jl [--pc-counts 2,4,8,16,32,64] [--mode all|full|test] [--no-resume]")
            exit(0)
        elseif arg == "--verbose"
            options["verbose"] = true
        elseif arg == "--no-resume"
            options["resume"] = false
        elseif arg in ("--pc-counts", "--mode", "--output-dir", "--seed", "--grid-step", "--maxit")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key == "pc_counts" ? parse.(Int, split(value, ',')) :
                           key in ("seed", "maxit") ? parse(Int, value) :
                           key == "grid_step" ? parse(Float64, value) :
                           key == "output_dir" ? abspath(value) : value
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    options["mode"] in ("all", "full", "test") || error("--mode must be all, full, or test")
    issorted(options["pc_counts"]) || error("PC counts must be sorted")
    all(>(0), options["pc_counts"]) || error("PC counts must be positive")
    return options
end

function write_compact_recovery(path, ids, reference, inferred_list, labels)
    open(path, "w") do io
        println(io, join(["sequence_id", "reference_temperature", ["inferred_" * label for label in labels]...], ','))
        for m in eachindex(ids)
            println(io, join(Any[ids[m], reference[m], [inferred[m] for inferred in inferred_list]...], ','))
        end
    end
end

function write_profile_summary(path, groups_list, labels)
    open(path, "w") do io
        println(io, "protocol,bucket_center,n,mean_reference_temperature,mean_profile_argmax")
        for (label, groups) in zip(labels, groups_list), group in groups
            @printf(io, "%s,%.6f,%d,%.6f,%.6f\n", label, group.center, group.n, group.mean_temperature, group.peak)
        end
    end
end

function sweep_full(Z, ids, temperatures, weights, options, pca_map, pc_scores, grid)
    root = joinpath(options["output_dir"], "full_generated")
    mkpath(root)
    comparison = NamedTuple[]
    for k in options["pc_counts"]
        output = joinpath(root, "pc$(k)")
        summary_path = joinpath(output, "summary.txt")
        if options["resume"] && isfile(summary_path)
            println("[full PC$k] existing result found; skipping")
            continue
        end
        mkpath(output)
        pcs = pc_scores[1:k, :]
        Y_raw = vcat(reshape(temperatures, 1, :), pcs)
        Y, means, scales = weighted_standardize(Y_raw, weights)
        println("[full PC$k] training FeatureDCA with $(k + 1) features...")
        network, arvar, objective, elapsed = fit_network(Z, Y, weights, options)
        Random.seed!(options["seed"])
        println("[full PC$k] generating $(length(ids)) sequences...")
        generated = FeatureDCA.sample(network, Y)
        resulting_pcs = Matrix{Float64}(predict(pca_map, FeatureDCA.one_hot(generated; q=21)))[1:k, :]
        println("[full PC$k] scanning with original PCs...")
        likelihood_original = likelihood_grid(network, generated, grid, means, scales, pcs)
        println("[full PC$k] scanning with resulting PCs...")
        likelihood_resulting = likelihood_grid(network, generated, grid, means, scales, resulting_pcs)
        inferred_original, inferred_resulting = infer(likelihood_original, grid), infer(likelihood_resulting, grid)
        metric_original = recovery_metrics(temperatures, inferred_original)
        metric_resulting = recovery_metrics(temperatures, inferred_resulting)
        pc_correlations = [cor(pcs[p, :], resulting_pcs[p, :]) for p in 1:k]

        serialize(joinpath(output, "model.jls"), (
            network=network, objective=objective, pca_count=k, feature_means=means, feature_scales=scales,
            sequence_ids=ids, weights=arvar.weights, elapsed_seconds=elapsed, seed=options["seed"]))
        write_compact_recovery(joinpath(output, "recovery.csv"), ids, temperatures,
                               [inferred_original, inferred_resulting], ["original_pc", "resulting_pc"])
        groups_original = profile_groups(temperatures, likelihood_original, grid)
        groups_resulting = profile_groups(temperatures, likelihood_resulting, grid)
        write_profile_summary(joinpath(output, "likelihood_profile_summary.csv"),
                              [groups_original, groups_resulting], ["original_pc", "resulting_pc"])
        plot_recovery(joinpath(output, "temperature_recovery.pdf"), temperatures,
                      [inferred_original, inferred_resulting], ["Original conditioning PCs", "Generated sequence PCs"],
                      "Generated-sequence recovery with temperature plus $k PCs")
        plot_profile_comparison(joinpath(output, "likelihood_profiles.pdf"),
                                [groups_original, groups_resulting], ["Original conditioning PCs", "Generated sequence PCs"],
                                grid, "Generated likelihood profiles with temperature plus $k PCs")
        open(summary_path, "w") do io
            @printf(io, "pca_components=%d\ntraining_sequences=%d\ntraining_seconds=%.3f\ngrid_step_C=%.6f\n",
                    k, length(ids), elapsed, options["grid_step"])
            print_metrics(io, "original_pc", temperatures, inferred_original)
            print_metrics(io, "resulting_pc", temperatures, inferred_resulting)
            @printf(io, "mean_PC_original_resulting_correlation=%.10f\nminimum_PC_original_resulting_correlation=%.10f\n",
                    mean(pc_correlations), minimum(pc_correlations))
        end
        push!(comparison, (k=k, original=metric_original, resulting=metric_resulting))
        likelihood_original = likelihood_resulting = nothing
        GC.gc()
    end
    return comparison
end

function sweep_test(Z, ids, temperatures, options, pca_map, pc_train_all, pc_test_all, train, test, weights, grid)
    root = joinpath(options["output_dir"], "heldout_natural")
    mkpath(root)
    comparison = NamedTuple[]
    for k in options["pc_counts"]
        output = joinpath(root, "pc$(k)")
        summary_path = joinpath(output, "summary.txt")
        if options["resume"] && isfile(summary_path)
            println("[test PC$k] existing result found; skipping")
            continue
        end
        mkpath(output)
        pc_train, pc_test = pc_train_all[1:k, :], pc_test_all[1:k, :]
        Y_raw = vcat(reshape(temperatures[train], 1, :), pc_train)
        Y, means, scales = weighted_standardize(Y_raw, weights)
        println("[test PC$k] training FeatureDCA with $(k + 1) features...")
        network, arvar, objective, elapsed = fit_network(Z[:, train], Y, weights, options)
        println("[test PC$k] scanning $(length(test)) held-out sequences...")
        likelihoods = likelihood_grid(network, Z[:, test], grid, means, scales, pc_test)
        inferred = infer(likelihoods, grid)
        metric = recovery_metrics(temperatures[test], inferred)

        serialize(joinpath(output, "model.jls"), (
            network=network, objective=objective, pca_count=k, feature_means=means, feature_scales=scales,
            train_sequence_ids=ids[train], test_sequence_ids=ids[test], weights=arvar.weights,
            elapsed_seconds=elapsed, seed=options["seed"]))
        write_compact_recovery(joinpath(output, "recovery.csv"), ids[test], temperatures[test], [inferred], ["test_sequence_pc"])
        groups = profile_groups(temperatures[test], likelihoods, grid)
        write_profile_summary(joinpath(output, "likelihood_profile_summary.csv"), [groups], ["test_sequence_pc"])
        plot_recovery(joinpath(output, "temperature_recovery.pdf"), temperatures[test], [inferred],
                      ["Test sequence PCs"], "Held-out natural recovery with temperature plus $k PCs")
        plot_profile_comparison(joinpath(output, "likelihood_profiles.pdf"), [groups], ["Held-out natural sequences"],
                                grid, "Natural test likelihood profiles with temperature plus $k PCs")
        open(summary_path, "w") do io
            @printf(io, "pca_components=%d\ntraining_sequences=%d\ntest_sequences=%d\ntraining_seconds=%.3f\ngrid_step_C=%.6f\n",
                    k, length(train), length(test), elapsed, options["grid_step"])
            print_metrics(io, "test", temperatures[test], inferred)
        end
        push!(comparison, (k=k, metric=metric))
        likelihoods = nothing
        GC.gc()
    end
    return comparison
end

function read_summary_value(path, key)
    for line in eachline(path)
        startswith(line, key * "=") && return parse(Float64, split(line, '='; limit=2)[2])
    end
    error("Missing $key in $path")
end

function collate_results(output_dir, pc_counts)
    path = joinpath(output_dir, "dimension_comparison.csv")
    open(path, "w") do io
        println(io, "pca_components,dataset,pc_protocol,pearson,spearman,mae_C,rmse_C,bias_C,within_1C,within_2C,within_5C")
        for k in pc_counts
            full = joinpath(output_dir, "full_generated", "pc$(k)", "summary.txt")
            test = joinpath(output_dir, "heldout_natural", "pc$(k)", "summary.txt")
            if isfile(full)
                for protocol in ("original_pc", "resulting_pc")
                    values = [read_summary_value(full, protocol * "_" * metric) for metric in
                              ("pearson", "spearman", "mae_C", "rmse_C", "bias_C", "within_1C", "within_2C", "within_5C")]
                    println(io, join(Any[k, "generated", protocol, values...], ','))
                end
            end
            if isfile(test)
                values = [read_summary_value(test, "test_" * metric) for metric in
                          ("pearson", "spearman", "mae_C", "rmse_C", "bias_C", "within_1C", "within_2C", "within_5C")]
                println(io, join(Any[k, "heldout_natural", "sequence_pc", values...], ','))
            end
        end
    end
    println("Comparison table written to $path")
end

function plot_dimension_comparison(output_dir)
    path = joinpath(output_dir, "dimension_comparison.csv")
    rows = split.(readlines(path)[2:end], ',')
    protocols = [
        ("generated", "original_pc", "Generated, original PCs", "#D7634C", "o"),
        ("generated", "resulting_pc", "Generated, resulting PCs", "#D5A129", "s"),
        ("heldout_natural", "sequence_pc", "Held-out natural", "#0E6B6F", "^")]
    fig, axes = subplots(1, 3, figsize=(14.2, 4.7), constrained_layout=true)
    metric_columns = [("Pearson", 4), ("Spearman", 5), ("MAE (degrees C)", 6)]
    for (ax, (label, column)) in zip(axes, metric_columns)
        for (dataset, protocol, legend, color, marker) in protocols
            selected = filter(row -> row[2] == dataset && row[3] == protocol, rows)
            x = parse.(Int, getindex.(selected, 1))
            y = parse.(Float64, getindex.(selected, column))
            ax.plot(x, y, marker=marker, linewidth=1.8, markersize=5, color=color, label=legend)
        end
        if label == "Pearson"
            ax.axhline(0.4204866001, color="#D7634C", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axhline(0.3854953141, color="#0E6B6F", linestyle="--", linewidth=0.9, alpha=0.7)
        elseif label == "Spearman"
            ax.axhline(0.3346333697, color="#D7634C", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axhline(0.3124998631, color="#0E6B6F", linestyle="--", linewidth=0.9, alpha=0.7)
        else
            ax.axhline(11.5441046574, color="#D7634C", linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axhline(11.9287974684, color="#0E6B6F", linestyle="--", linewidth=0.9, alpha=0.7)
        end
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
        ax.set_xlabel("Number of PCA components")
        ax.set_ylabel(label)
        ax.grid(true, color="#d8dde3", linewidth=0.5, alpha=0.7)
    end
    axes[1].legend(frameon=false, fontsize=8)
    fig.suptitle("Temperature recovery degrades as PCA conditioning dimension increases", fontsize=14)
    fig.savefig(joinpath(output_dir, "dimension_comparison.pdf"), dpi=300)
    close(fig)
end

function main_sweep(args=ARGS)
    options = sweep_args(args)
    metadata = read_metadata(METADATA)
    weights_all, Z, _, _, _, annotations = FeatureDCA.read_annotated_fasta(FASTA; remove_dups=false, max_gap_fraction=1.0)
    ids = sequence_id.(annotations)
    temperatures = [parse(Float64, metadata[id]["Topt_ave"]) for id in ids]
    max_pcs = maximum(options["pc_counts"])
    grid = temperature_grid(temperatures, options["grid_step"])
    mkpath(options["output_dir"])

    if options["mode"] in ("all", "full")
        println("Fitting $max_pcs-component PCA on the full MSA...")
        encoded = FeatureDCA.one_hot(Z; q=21)
        pca_full = fit(PCA, encoded, maxoutdim=max_pcs)
        pc_full = Matrix{Float64}(predict(pca_full, encoded))
        serialize(joinpath(options["output_dir"], "full_pca$(max_pcs).jls"), pca_full)
        sweep_full(Z, ids, temperatures, weights_all, options, pca_full, pc_full, grid)
        encoded = pc_full = nothing
        GC.gc()
    end

    if options["mode"] in ("all", "test")
        split = deserialize(SPLIT_ARTIFACT)
        train_set, test_set = Set(split.train_sequence_ids), Set(split.test_sequence_ids)
        train, test = findall(id -> id in train_set, ids), findall(id -> id in test_set, ids)
        weights, _ = FeatureDCA.compute_weights(Z[:, train], :auto)
        println("Fitting leakage-free $max_pcs-component PCA on the training MSA...")
        encoded_train = FeatureDCA.one_hot(Z[:, train]; q=21)
        pca_train = fit(PCA, encoded_train, maxoutdim=max_pcs)
        pc_train = Matrix{Float64}(predict(pca_train, encoded_train))
        pc_test = Matrix{Float64}(predict(pca_train, FeatureDCA.one_hot(Z[:, test]; q=21)))
        serialize(joinpath(options["output_dir"], "train_pca$(max_pcs).jls"), pca_train)
        sweep_test(Z, ids, temperatures, options, pca_train, pc_train, pc_test, train, test, weights, grid)
    end
    collate_results(options["output_dir"], options["pc_counts"])
    plot_dimension_comparison(options["output_dir"])
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main_sweep()
end
