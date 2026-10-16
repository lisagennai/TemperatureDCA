#!/usr/bin/env julia
#
# Maximum-likelihood temperature of the training sequences of a temperature-conditioned FeatureDCA model.
#
# For every training sequence, log P(sequence | T) is evaluated on a temperature grid. The grid temperature
# with the highest likelihood is compared with the natural temperature of the source organism (scatter plot,
# Pearson and Spearman correlation, MAE, RMSE). The same is repeated per organism after averaging the
# likelihood profiles of its homologs, as in crossvalidate_fixed_temperature_bins.jl.
#
# Run from the TemperatureDCA folder:
#   JULIA_NUM_THREADS=auto julia --project=. analysis/infer_training_temperatures.jl --model results/featuredca/<model>.jls

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

using FeatureDCA      # needed to deserialize the ArNet stored in the model
using Printf
using PyPlot
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_MODEL = joinpath(ROOT, "results", "featuredca", "jdomain_topt_ave.jls")
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_METADATA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.metadata.maxgap50.csv")

# Plot colors: one-hue sequential ramp for point density (light = few), neutral ink for text and axes.
const BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const REFERENCE_LINE = "#898781"
const GRID_LINE = "#e4e3df"

function usage(io=stdout)
    println(io, """
Maximum-likelihood temperature of the training sequences of a FeatureDCA model, compared with the
natural temperature of their source organisms.

Usage:
  JULIA_NUM_THREADS=auto julia --project=. analysis/infer_training_temperatures.jl [options]

Options:
  --model PATH         Serialized model (default: results/featuredca/jdomain_topt_ave.jls). Works with
                       train_jdomain_featuredca.jl models (sequence_ids) and with the model.jls / foldN_model.jls
                       of the held-out scripts (train_sequence_ids).
  --fasta PATH         Aligned FASTA the model was trained on (default: path stored in the model, else the
                       J-domain maxgap50 MSA)
  --metadata PATH      Matching metadata CSV (default: path stored in the model, else the J-domain metadata)
  --feature NAME       Model feature to scan (default: Topt_ave, else the first feature). Other features of a
                       multi-feature model are held at each sequence's own metadata values.
  --grid-min VALUE     Lowest grid temperature (default: training minimum rounded down to a grid step)
  --grid-max VALUE     Highest grid temperature (default: training maximum rounded up to a grid step)
  --grid-step VALUE    Grid spacing in degrees C (default: 0.5)
  --output-dir PATH    Output folder (default: <model name>_training_temperatures next to the model)
  -h, --help           Show this help

Outputs (in --output-dir):
  training_likelihoods.csv        one row per training sequence: natural and inferred temperature, error and the
                                  log-likelihood at every grid temperature (same layout as test_likelihoods.csv)
  organism_temperatures.csv       the same per organism (homolog profiles averaged)
  training_temperature_recovery.pdf/.png
  summary.txt
""")
end

function parse_args(args)
    options = Dict{String,Any}(
        "model" => DEFAULT_MODEL, "fasta" => nothing, "metadata" => nothing, "feature" => nothing,
        "grid_min" => nothing, "grid_max" => nothing, "grid_step" => 0.5, "output_dir" => nothing,
    )
    value_options = Set(["--model", "--fasta", "--metadata", "--feature", "--grid-min", "--grid-max",
                         "--grid-step", "--output-dir"])
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif arg in value_options
            i == length(args) && error("Missing value after $arg")
            key, value = replace(arg[3:end], "-" => "_"), args[i + 1]
            if key in ("grid_min", "grid_max", "grid_step")
                options[key] = parse(Float64, value)
            elseif key == "feature"
                options[key] = String(value)
            else
                options[key] = abspath(value)
            end
            i += 1
        else
            error("Unknown option: $arg (see --help)")
        end
        i += 1
    end
    options["grid_step"] > 0 || error("--grid-step must be positive")
    return options
end

# ------------------------------------------------------------------------------------------------
# Metadata (same CSV reader as train_jdomain_featuredca.jl)
# ------------------------------------------------------------------------------------------------

function parse_csv(text::AbstractString)
    rows = Vector{Vector{String}}()
    fields = String[]
    buffer = IOBuffer()
    in_quotes = false
    i = firstindex(text)
    while i <= lastindex(text)
        char = text[i]
        if char == '"'
            next_i = nextind(text, i)
            if in_quotes && next_i <= lastindex(text) && text[next_i] == '"'
                print(buffer, '"')
                i = next_i
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
        push!(fields, String(take!(buffer)))
        push!(rows, fields)
    end
    in_quotes && error("Unterminated quoted field in CSV")
    return rows
end

function read_metadata(path)
    parsed = parse_csv(read(path, String))
    isempty(parsed) && error("Metadata file is empty: $path")
    header = parsed[1]
    "sequence_id" in header || error("Metadata has no sequence_id column: $path")
    rows = Dict{String,Dict{String,String}}()
    for values in parsed[2:end]
        length(values) == length(header) || error("Malformed metadata row with $(length(values)) fields")
        row = Dict(header[i] => values[i] for i in eachindex(header))
        haskey(rows, row["sequence_id"]) && error("Duplicate metadata sequence_id: $(row["sequence_id"])")
        rows[row["sequence_id"]] = row
    end
    return rows, header
end

function feature_value(metadata, id, name)
    text = strip(get(metadata[id], name, ""))
    isempty(text) && error("Sequence $id has no $name value in the metadata")
    return parse(Float64, text)
end

annotation_text(annotation::AbstractString) = annotation
annotation_text(annotation::Tuple) = only(annotation)
sequence_id(annotation) = String(first(split(strip(annotation_text(annotation)))))

# ------------------------------------------------------------------------------------------------
# What the training scripts store in the model
# ------------------------------------------------------------------------------------------------

function stored_vector(model, plural::Symbol, singular::Symbol)
    hasproperty(model, plural) && return Float64.(collect(getproperty(model, plural)))
    hasproperty(model, singular) && return Float64[getproperty(model, singular)]
    error("The model stores neither `$plural` nor `$singular`")
end

function model_description(model)
    d = size(model.network.G[1], 2)
    ids = if hasproperty(model, :train_sequence_ids)      # evaluate_natural_temperature_recovery, CV folds
        model.train_sequence_ids
    elseif hasproperty(model, :sequence_ids)              # train_jdomain_featuredca and full-data models
        model.sequence_ids
    else
        error("The model has neither `train_sequence_ids` nor `sequence_ids`, so its training set is unknown")
    end
    names = if hasproperty(model, :feature_names)
        String.(collect(model.feature_names))
    elseif d == 1
        ["Topt_ave"]                                      # CV fold models store only feature_mean/feature_scale
    else
        error("The model has $d features but no `feature_names`")
    end
    means = stored_vector(model, :feature_means, :feature_mean)
    scales = stored_vector(model, :feature_scales, :feature_scale)
    length(names) == length(means) == length(scales) == d ||
        error("The model has $d features but stores $(length(names)) names, $(length(means)) means, $(length(scales)) scales")
    return String.(collect(ids)), names, means, scales
end

function input_path(option, model, field::Symbol, default, flag)
    option === nothing || return option
    if hasproperty(model, field)
        path = String(getproperty(model, field))
        isfile(path) && return path
        @warn "The file stored in the model ($path) doesn't exist; using $default instead (choose another with --$flag)"
    end
    return default
end

# ------------------------------------------------------------------------------------------------
# Likelihood profiles
# ------------------------------------------------------------------------------------------------

function logsumexp(x::AbstractVector)
    m = maximum(x)
    s = 0.0
    @inbounds for v in x
        s += exp(v - m)
    end
    return m + log(s)
end

# Straightforward autoregressive log-likelihood (as in evaluate_natural_temperature_recovery.jl); used as a check.
function reference_loglikelihood(network, sequence, y)
    H, G, J = network.H, network.G, network.J
    logits = H[1] .+ G[1] * y
    total = logits[sequence[1]] - logsumexp(logits)
    for site in 2:length(H)
        logits .= H[site] .+ G[site] * y
        coupling = J[site - 1]
        @inbounds for previous in 1:site-1, state in eachindex(logits)
            logits[state] += coupling[previous, state, sequence[previous]]
        end
        total += logits[sequence[site]] - logsumexp(logits)
    end
    return total
end

# log P(sequence | y) for every sequence (rows) and grid value of feature k (columns).
# The site logits are H + G*y + couplings to the preceding residues; only G[:, k] * y_k changes along the grid,
# so the rest is computed once per sequence.
function likelihood_profiles(network, Z, k, y_fixed, standardized_grid)
    H, G, J = network.H, network.G, network.J
    L, q = length(H), length(H[1])
    slope = reduce(hcat, [G[site][:, k] for site in 1:L])          # q x L
    profiles = Matrix{Float64}(undef, size(Z, 2), length(standardized_grid))
    Threads.@threads for n in axes(Z, 2)
        sequence = view(Z, :, n)
        base = Matrix{Float64}(undef, q, L)
        for site in 1:L
            b = view(base, :, site)
            b .= H[site] .+ G[site] * view(y_fixed, :, n)
            if site > 1
                coupling = J[site - 1]
                @inbounds for previous in 1:site-1, state in 1:q
                    b[state] += coupling[previous, state, sequence[previous]]
                end
            end
        end
        logits = Vector{Float64}(undef, q)
        for (t, y) in enumerate(standardized_grid)
            total = 0.0
            for site in 1:L
                @inbounds for state in 1:q
                    logits[state] = base[state, site] + slope[state, site] * y
                end
                total += logits[sequence[site]] - logsumexp(logits)
            end
            profiles[n, t] = total
        end
    end
    return profiles
end

function check_against_reference(network, Z, k, y_fixed, standardized_grid, profiles)
    T = length(standardized_grid)
    for n in 1:min(size(Z, 2), 5), t in unique([1, cld(T, 2), T])
        y = y_fixed[:, n]
        y[k] = standardized_grid[t]
        expected = reference_loglikelihood(network, view(Z, :, n), y)
        isapprox(profiles[n, t], expected; rtol=1e-9, atol=1e-9) ||
            error("Likelihood check failed for sequence $n, grid point $t: $(profiles[n, t]) vs $expected")
    end
end

argmax_rows(profiles) = [argmax(view(profiles, n, :)) for n in axes(profiles, 1)]

# ------------------------------------------------------------------------------------------------
# Metrics and output
# ------------------------------------------------------------------------------------------------

function tied_ranks(values)
    order = sortperm(values)
    ranks = Vector{Float64}(undef, length(values))
    first_index = 1
    while first_index <= length(order)
        last_index = first_index
        while last_index < length(order) && values[order[last_index + 1]] == values[order[first_index]]
            last_index += 1
        end
        ranks[order[first_index:last_index]] .= (first_index + last_index) / 2
        first_index = last_index + 1
    end
    return ranks
end

spearman_correlation(x, y) = cor(tied_ranks(x), tied_ranks(y))

function recovery_metrics(natural, inferred, grid)
    errors = inferred .- natural
    at_edge = (inferred .== first(grid)) .| (inferred .== last(grid))
    return (n=length(natural), pearson=cor(natural, inferred), spearman=spearman_correlation(natural, inferred),
            mae=mean(abs.(errors)), rmse=sqrt(mean(abs2.(errors))), bias=mean(errors),
            within_2C=mean(abs.(errors) .<= 2), within_5C=mean(abs.(errors) .<= 5), at_grid_edge=mean(at_edge))
end

with_commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

function csv_escape(value)
    text = string(value)
    any(c -> occursin(c, text), [',', '"', '\n']) || return text
    return "\"" * replace(text, "\"" => "\"\"") * "\""
end

grid_label(t) = "loglik_T" * replace(string(t), "." => "p")

function write_sequence_table(path, ids, taxa, natural, inferred, grid, profiles)
    winners = argmax_rows(profiles)
    nearest = [argmin(abs.(grid .- t)) for t in natural]
    open(path, "w") do io
        println(io, join(["sequence_id", "taxonomy_id", "natural_temperature", "inferred_temperature", "error",
                          "absolute_error", "true_grid_is_argmax", "argmax_at_grid_edge", grid_label.(grid)...], ','))
        for n in eachindex(ids)
            error_n = inferred[n] - natural[n]
            at_edge = winners[n] == 1 || winners[n] == length(grid)
            println(io, join(Any[ids[n], taxa[n], natural[n], inferred[n], error_n, abs(error_n),
                                 nearest[n] == winners[n], at_edge, view(profiles, n, :)...], ','))
        end
    end
end

function write_organism_table(path, taxa, names, sizes, natural, inferred)
    open(path, "w") do io
        println(io, "taxonomy_id,organism,n_sequences,natural_temperature,inferred_temperature,error,absolute_error")
        for g in eachindex(taxa)
            error_g = inferred[g] - natural[g]
            println(io, join(csv_escape.(Any[taxa[g], names[g], sizes[g], natural[g], inferred[g], error_g, abs(error_g)]), ','))
        end
    end
end

function metrics_text(m)
    return @sprintf("Pearson r = %.3f\nSpearman ρ = %.3f\nMAE = %.2f °C, RMSE = %.2f °C\nn = %s",
                    m.pearson, m.spearman, m.mae, m.rmse, with_commas(m.n))
end

function plot_recovery(stem, panels, feature, step)
    cmap = PyPlot.matplotlib.colors.LinearSegmentedColormap.from_list("sequential_blue", BLUES)
    lo = minimum(min(minimum(p.natural), minimum(p.inferred)) for p in panels)
    hi = maximum(max(maximum(p.natural), maximum(p.inferred)) for p in panels)
    pad = max(0.03 * (hi - lo), step)
    limits = (lo - pad, hi + pad)
    fig = figure(figsize=(5.6 * length(panels), 5.0), constrained_layout=true)
    for (i, p) in enumerate(panels)
        ax = fig.add_subplot(1, length(panels), i)
        hb = ax.hexbin(p.natural, p.inferred; gridsize=40, mincnt=1, bins="log", cmap=cmap,
                       extent=(limits..., limits...), linewidths=0.0)
        ax.plot(collect(limits), collect(limits); linestyle="--", color=REFERENCE_LINE, linewidth=1.0, zorder=3)
        ax.text(0.04, 0.96, metrics_text(p.metrics); transform=ax.transAxes, ha="left", va="top", fontsize=9,
                color=INK_SECONDARY, zorder=4,
                bbox=Dict("boxstyle" => "round,pad=0.4", "facecolor" => "white", "edgecolor" => "none", "alpha" => 0.85))
        ax.text(0.97, 0.03, "y = x"; transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=REFERENCE_LINE)
        ax.set_xlim(limits...)
        ax.set_ylim(limits...)
        ax.set_aspect("equal")
        ax.set_xlabel("Natural $feature (°C)"; color=INK)
        ax.set_ylabel("Maximum-likelihood $feature (°C)"; color=INK)
        ax.set_title(p.title; loc="left", fontsize=11, color=INK)
        ax.grid(true; color=GRID_LINE, linewidth=0.6)
        ax.set_axisbelow(true)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        for side in ("top", "right")
            ax.spines[side].set_visible(false)
        end
        for side in ("left", "bottom")
            ax.spines[side].set_color(REFERENCE_LINE)
        end
        cb = fig.colorbar(hb; ax=ax, shrink=0.85)
        cb.set_label(p.count_label; color=INK_SECONDARY)
        cb.outline.set_visible(false)
        cb.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        cb.ax.yaxis.set_major_formatter(PyPlot.matplotlib.ticker.StrMethodFormatter("{x:g}"))   # 1, 10, 100 instead of 10^n
        cb.ax.yaxis.set_minor_formatter(PyPlot.matplotlib.ticker.NullFormatter())
    end
    fig.savefig(stem * ".pdf")
    fig.savefig(stem * ".png"; dpi=200)
    close(fig)
end

function write_metrics(io, prefix, m)
    for key in keys(m)
        value = getproperty(m, key)
        value isa Integer ? println(io, "$(prefix)$(key)=$(value)") : @printf(io, "%s%s=%.10f\n", prefix, key, value)
    end
end

function print_metrics(label, m)
    @printf("%s (n = %s)\n", label, with_commas(m.n))
    @printf("  Pearson r = %.3f   Spearman rho = %.3f\n", m.pearson, m.spearman)
    @printf("  MAE = %.2f C   RMSE = %.2f C   bias = %+.2f C\n", m.mae, m.rmse, m.bias)
    @printf("  within 2 C: %.1f%%   within 5 C: %.1f%%   argmax at a grid edge: %.1f%%\n",
            100 * m.within_2C, 100 * m.within_5C, 100 * m.at_grid_edge)
end

# ------------------------------------------------------------------------------------------------

function main(args=ARGS)
    options = parse_args(args)
    model_path = options["model"]
    isfile(model_path) || error("Model not found: $model_path")
    model = deserialize(model_path)
    hasproperty(model, :network) || error("$model_path has no `network`; expected a model saved by the TemperatureDCA scripts")
    network = model.network
    train_ids, names, means, scales = model_description(model)

    feature = something(options["feature"], "Topt_ave" in names ? "Topt_ave" : first(names))
    k = findfirst(==(feature), names)
    k === nothing && error("$feature is not a feature of the model (features: $(join(names, ", ")))")

    fasta = input_path(options["fasta"], model, :source_fasta, DEFAULT_FASTA, "fasta")
    metadata_path = input_path(options["metadata"], model, :source_metadata, DEFAULT_METADATA, "metadata")
    isfile(fasta) || error("FASTA not found: $fasta")
    isfile(metadata_path) || error("Metadata not found: $metadata_path")
    model_name = splitext(basename(model_path))[1]
    output_dir = something(options["output_dir"], joinpath(dirname(model_path), model_name * "_training_temperatures"))
    mkpath(output_dir)

    # Training sequences, read the way the training scripts read them
    metadata, metadata_columns = read_metadata(metadata_path)
    _, Z, L, _, _, annotations = FeatureDCA.read_annotated_fasta(fasta; remove_dups=false, max_gap_fraction=1.0)
    column_of = Dict(sequence_id(a) => m for (m, a) in enumerate(annotations))
    not_in_fasta = filter(id -> !haskey(column_of, id), train_ids)
    isempty(not_in_fasta) ||
        error("$(length(not_in_fasta)) training sequences of the model are not in $fasta (e.g. $(first(not_in_fasta)))")
    not_in_metadata = filter(id -> !haskey(metadata, id), train_ids)
    isempty(not_in_metadata) ||
        error("$(length(not_in_metadata)) training sequences are not in $metadata_path (e.g. $(first(not_in_metadata)))")
    L == length(network.H) || error("The FASTA has $L alignment columns but the model has $(length(network.H)) sites")
    Z_train = Z[:, [column_of[id] for id in train_ids]]
    q = length(network.H[1])
    maximum(Z_train) <= q || error("The training sequences use more states than the model's q = $q")

    # Natural values; other features of a multi-feature model stay at each sequence's own standardized value
    natural = [feature_value(metadata, id, feature) for id in train_ids]
    y_fixed = zeros(length(names), length(train_ids))
    for j in eachindex(names)
        j == k && continue
        names[j] in metadata_columns ||
            error("The model also conditions on $(names[j]), which isn't a metadata column; it can't be held at the natural value")
        for (n, id) in enumerate(train_ids)
            y_fixed[j, n] = (feature_value(metadata, id, names[j]) - means[j]) / scales[j]
        end
    end

    step = options["grid_step"]
    grid_min = something(options["grid_min"], floor(minimum(natural) / step) * step)
    grid_max = something(options["grid_max"], ceil(maximum(natural) / step) * step)
    grid_min <= grid_max || error("--grid-min must not exceed --grid-max")
    grid = collect(grid_min:step:grid_max)
    standardized_grid = (grid .- means[k]) ./ scales[k]

    @printf("Model: %s (features: %s)\n", model_path, join(names, ", "))
    length(names) > 1 && println("  scanning $feature; $(join(setdiff(names, [feature]), ", ")) held at the natural values")
    @printf("Training set: %s sequences x %d sites from %s\n", with_commas(length(train_ids)), L, fasta)
    @printf("Grid: %.2f to %.2f C in steps of %.2f C (%d temperatures), %d threads\n",
            first(grid), last(grid), step, length(grid), Threads.nthreads())

    started = time()
    profiles = likelihood_profiles(network, Z_train, k, y_fixed, standardized_grid)
    check_against_reference(network, Z_train, k, y_fixed, standardized_grid, profiles)
    @printf("Likelihoods computed in %.1f s\n\n", time() - started)
    inferred = grid[argmax_rows(profiles)]
    sequence_metrics = recovery_metrics(natural, inferred, grid)

    # Organisms: average the profiles of the homologs from the same taxonomy ID (same argmax as their sum)
    taxa = [String(strip(get(metadata[id], "taxonomy_id", ""))) for id in train_ids]
    groups = Dict{String,Vector{Int}}()
    for (n, taxon) in enumerate(taxa)
        isempty(taxon) || push!(get!(groups, taxon, Int[]), n)
    end
    organism_taxa = sort!(collect(keys(groups)))
    members = [groups[t] for t in organism_taxa]
    organism_profiles = Matrix{Float64}(undef, length(members), length(grid))
    for (g, rows) in enumerate(members)
        organism_profiles[g, :] .= vec(mean(view(profiles, rows, :); dims=1))
    end
    organism_natural = [mean(natural[rows]) for rows in members]
    organism_inferred = grid[argmax_rows(organism_profiles)]
    organism_names = [get(metadata[train_ids[first(rows)]], "organism", "") for rows in members]
    conflicting = count(rows -> length(unique(natural[rows])) > 1, members)
    conflicting > 0 && @warn "$conflicting organisms have homologs with different $feature values; their mean is used"
    with_organisms = length(members) >= 3

    print_metrics("Training sequences", sequence_metrics)
    panels = [(natural=natural, inferred=inferred, metrics=sequence_metrics,
               title="Training sequences", count_label="Sequences per hexagon")]
    organism_metrics = nothing
    if with_organisms
        organism_metrics = recovery_metrics(organism_natural, organism_inferred, grid)
        print_metrics("Training organisms, homolog profiles averaged", organism_metrics)
        push!(panels, (natural=organism_natural, inferred=organism_inferred, metrics=organism_metrics,
                       title="Training organisms (homologs averaged)", count_label="Organisms per hexagon"))
        any(isempty, taxa) &&
            println("  ($(count(isempty, taxa)) sequences without a taxonomy_id are left out at the organism level)")
    else
        println("Organism level skipped: fewer than 3 organisms with a taxonomy_id")
    end
    if sequence_metrics.at_grid_edge > 0.01
        println("\nNote: the likelihood peaks at the edge of the grid for $(round(100 * sequence_metrics.at_grid_edge; digits=1))% " *
                "of the sequences; widen it with --grid-min/--grid-max to see where those maxima are.")
    end

    write_sequence_table(joinpath(output_dir, "training_likelihoods.csv"), train_ids, taxa, natural, inferred, grid, profiles)
    with_organisms && write_organism_table(joinpath(output_dir, "organism_temperatures.csv"), organism_taxa,
                                           organism_names, length.(members), organism_natural, organism_inferred)
    plot_recovery(joinpath(output_dir, "training_temperature_recovery"), panels, feature, step)
    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "model=$model_path\nfasta=$fasta\nmetadata=$metadata_path\nfeature=$feature")
        println(io, "model_features=$(join(names, ','))")
        @printf(io, "grid_min=%.6f\ngrid_max=%.6f\ngrid_step=%.6f\n", first(grid), last(grid), step)
        @printf(io, "feature_mean=%.10f\nfeature_scale=%.10f\n", means[k], scales[k])
        write_metrics(io, "sequence_", sequence_metrics)
        organism_metrics === nothing || write_metrics(io, "organism_", organism_metrics)
    end
    println("\nResults written to $output_dir")
end

main()
