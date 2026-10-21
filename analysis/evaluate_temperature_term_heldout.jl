#!/usr/bin/env julia
#
# Held-out test of the temperature term in FeatureDCA training.
#
# Models are trained (Adam, analytic gradient) on
#
#     loss = sum_s w_s [ -log P(s | y_s) - beta * log P(T_s | s) ] / sum_s w_s
#            + lambda_h |H|^2 + lambda_g |G|^2 + lambda_j |J|^2
#
# where P(T | s) is the softmax of log P(s | T) over a training temperature grid (uniform prior) and w_s are
# phylogenetic weights; beta = 0 is the plain conditional maximum-likelihood objective.
#
# Protocol (organisms, i.e. taxonomy_id, are never shared between training, validation and test):
#   1. training organisms (those of --split-from, or a random split) and test organisms; a fraction of the training
#      organisms is set aside as validation
#   2. pretrain a beta = 0 model on the training organisms without the validation ones (frequency initialization)
#   3. for every beta of --betas, continue training from it and keep the epoch with the best validation score
#   4. choose beta on validation only
#   5. refit beta = 0 and the chosen beta on training + validation organisms with the chosen epochs, and evaluate them
#      on the test organisms, together with the --split-from model (the native FeatureDCA fit) when given
#
# The test evaluation is the one of infer_training_temperatures.jl: maximum-likelihood temperature on a grid, per
# sequence and per organism (homolog profiles averaged).
#
# Run from the TemperatureDCA folder:
#   JULIA_NUM_THREADS=auto julia --project=. analysis/evaluate_temperature_term_heldout.jl --split-from <held-out model.jls>

ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])

using FeatureDCA      # read_annotated_fasta, and to deserialize the ArNet stored in --split-from
using LinearAlgebra
using Printf
using PyPlot
using Random
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_METADATA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.metadata.maxgap50.csv")
const DEFAULT_OUTPUT_ROOT = joinpath(ROOT, "results", "featuredca")

# Plot colors (as infer_training_temperatures.jl)
const BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const REFERENCE_LINE = "#898781"
const GRID_LINE = "#e4e3df"

function usage(io=stdout)
    println(io, """
Held-out test of the temperature term: FeatureDCA models are trained with
    loss = weighted mean over sequences of [ -log P(s | y_s) - beta * log P(T_s | s) ] + L2,
beta is chosen on validation organisms, and the chosen model and the beta = 0 model are evaluated on test organisms
(with the native FeatureDCA model of --split-from as reference).

Usage:
  JULIA_NUM_THREADS=auto julia --project=. analysis/evaluate_temperature_term_heldout.jl [options]

Data and split (organisms, i.e. taxonomy_id, are never shared between training, validation and test):
  --split-from PATH        Held-out model of the existing scripts (model.jls / foldN_model.jls). Its training
                           sequences are the training organisms; its test_sequence_ids (else every other sequence of
                           the FASTA with the features in the metadata, minus training organisms) are the test set; its
                           features and standardization are reused, and it is evaluated on the test set as the native
                           FeatureDCA reference. Without it, organisms are split at random with --test-fraction.
  --fasta PATH             Aligned FASTA (default: stored in --split-from, else the J-domain maxgap50 MSA)
  --metadata PATH          Metadata CSV (default: stored in --split-from, else the J-domain metadata)
  --features LIST          Comma-separated model features when there is no --split-from (default: Topt_ave)
  --feature NAME           Feature that gets the temperature term and is scanned (default: Topt_ave, else the first)
  --test-fraction F        Fraction of the organisms in the test set without --split-from (default: 0.2)
  --val-fraction F         Fraction of the training organisms held out to choose beta and the epoch (default: 0.15)
  --seed N                 Splits and batch order (default: 1)

Training:
  --betas LIST             Comma-separated absolute beta values; 0 is always added (default: 0,0.1,0.3,1,3).
                           beta weights the temperature term against the generative log-likelihood. Although the
                           generative term is a sum over L sites, the temperature term only has to move the feature
                           slopes G, where the two gradients are of the same order regardless of L, so beta ~ 1
                           already balances them and the useful range is O(1), not O(L). Widen the grid if the best
                           value sits at an end.
  --lambda-h VALUE         L2 on the fields H (default: 1e-6)
  --lambda-g VALUE         L2 on the feature slopes G (default: 1e-6)
  --lambda-j VALUE         L2 on the couplings J (default: 1e-4)
                           Set the three to the values your FeatureDCA training uses.
  --theta VALUE            Phylogenetic reweighting: sequences with identity >= 1 - theta share their weight
                           (default: 0.2; 0 = no reweighting)
  --eta VALUE              Adam learning rate (default: 0.005)
  --batch-size N           Sequences per gradient step (default: 256)
  --pretrain-epochs N      beta = 0 epochs before the beta sweep (default: 40)
  --epochs N               Epochs per beta after the pretraining (default: 40)
  --report-every N         Validation evaluation interval in epochs (default: 5)
  --select-by NAME         Validation score for the epoch and beta: mae, organism_mae or mean_ce (default: mae)
  --train-grid-step VALUE  Grid spacing in degrees C of the temperature term during training (default: 2)
  --no-refit               Evaluate the models trained without the validation organisms instead of refitting
  --gradcheck              Compare the analytic gradient with finite differences and exit

Evaluation grid (as infer_training_temperatures.jl; it covers the training and the test temperatures):
  --grid-min VALUE         Lowest grid temperature (default: minimum rounded down to a grid step)
  --grid-max VALUE         Highest grid temperature (default: maximum rounded up to a grid step)
  --grid-step VALUE        Grid spacing in degrees C (default: 0.5)
  --output-dir PATH        Output folder (default: <split-from name>_temperature_term next to --split-from, else
                           results/featuredca/temperature_term_heldout_<feature>)
  -h, --help               Show this help

Outputs (in --output-dir):
  split.csv                              sequence_id, taxonomy_id, set (train / validation / test), weight
  training_log.csv                       validation metrics during pretraining and the beta sweep
  beta_sweep.csv                         best epoch and validation metrics per beta
  sweep_models/beta<b>.jls               the best-epoch model of every beta (trained without validation organisms)
  model_beta0.jls, model_beta<b>.jls     the models evaluated on the test set (readable by
                                         infer_training_temperatures.jl)
  test_likelihoods_<model>.csv           same layout as training_likelihoods.csv, for the test sequences
  test_organism_temperatures_<model>.csv
  test_recovery_<model>.pdf/.png
  summary.txt
""")
end

const FLOAT_OPTIONS = Set(["grid_min", "grid_max", "grid_step", "train_grid_step", "test_fraction", "val_fraction",
                           "lambda_h", "lambda_g", "lambda_j", "theta", "eta"])
const INT_OPTIONS = Set(["seed", "batch_size", "pretrain_epochs", "epochs", "report_every"])
const PATH_OPTIONS = Set(["split_from", "fasta", "metadata", "output_dir"])
const STRING_OPTIONS = Set(["features", "feature", "betas", "select_by"])
const VALUE_OPTIONS = union(FLOAT_OPTIONS, INT_OPTIONS, PATH_OPTIONS, STRING_OPTIONS)
const FLAG_OPTIONS = Set(["gradcheck", "no_refit"])

function parse_args(args)
    options = Dict{String,Any}(
        "split_from" => nothing, "fasta" => nothing, "metadata" => nothing, "features" => "Topt_ave",
        "feature" => nothing, "test_fraction" => 0.2, "val_fraction" => 0.15, "seed" => 1,
        "betas" => nothing, "lambda_h" => 1e-6, "lambda_g" => 1e-6, "lambda_j" => 1e-4, "theta" => 0.2,
        "eta" => 0.005, "batch_size" => 256, "pretrain_epochs" => 40, "epochs" => 40, "report_every" => 5,
        "select_by" => "mae", "train_grid_step" => 2.0,
        "grid_min" => nothing, "grid_max" => nothing, "grid_step" => 0.5, "output_dir" => nothing,
        "gradcheck" => false, "no_refit" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        key = startswith(arg, "--") ? replace(arg[3:end], "-" => "_") : ""
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif key in FLAG_OPTIONS
            options[key] = true
        elseif key in VALUE_OPTIONS
            i == length(args) && error("Missing value after $arg")
            value = args[i + 1]
            options[key] = key in FLOAT_OPTIONS ? parse(Float64, value) :
                           key in INT_OPTIONS ? parse(Int, value) :
                           key in PATH_OPTIONS ? abspath(value) : String(value)
            i += 1
        else
            error("Unknown option: $arg (see --help)")
        end
        i += 1
    end
    options["grid_step"] > 0 || error("--grid-step must be positive")
    options["train_grid_step"] > 0 || error("--train-grid-step must be positive")
    0 < options["test_fraction"] < 1 || error("--test-fraction must be between 0 and 1")
    0 < options["val_fraction"] < 1 || error("--val-fraction must be between 0 and 1")
    0 <= options["theta"] < 1 || error("--theta must be in [0, 1)")
    min(options["lambda_h"], options["lambda_g"], options["lambda_j"]) >= 0 || error("lambdas must not be negative")
    options["eta"] > 0 || error("--eta must be positive")
    options["batch_size"] >= 1 || error("--batch-size must be at least 1")
    min(options["pretrain_epochs"], options["epochs"]) >= 0 || error("epochs must not be negative")
    options["report_every"] >= 1 || error("--report-every must be at least 1")
    options["select_by"] in ("mae", "organism_mae", "mean_ce") || error("--select-by must be mae, organism_mae or mean_ce")
    return options
end

# ------------------------------------------------------------------------------------------------
# Metadata and model (same as infer_training_temperatures.jl)
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

# Value of a metadata column, or nothing if it is empty or not a number
function parsed_value(metadata, id, name)
    text = strip(get(metadata[id], name, ""))
    isempty(text) && return nothing
    return tryparse(Float64, text)
end

taxonomy_of(metadata, id) = String(strip(get(metadata[id], "taxonomy_id", "")))

# Unit of the splits: the organism, or the sequence itself when it has no taxonomy_id
function split_group(metadata, id)
    taxon = taxonomy_of(metadata, id)
    return isempty(taxon) ? "sequence:" * id : taxon
end

annotation_text(annotation::AbstractString) = annotation
annotation_text(annotation::Tuple) = only(annotation)
sequence_id(annotation) = String(first(split(strip(annotation_text(annotation)))))

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
# Likelihoods (the same model as infer_training_temperatures.jl)
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

# Site logits without the scanned feature: H + G * y_fixed (y_fixed[k] = 0) + couplings to the preceding residues.
function fill_base!(base, network, sequence, y_fixed)
    H, G, J = network.H, network.G, network.J
    q = length(H[1])
    for site in eachindex(H)
        b = view(base, :, site)
        mul!(b, G[site], y_fixed)
        b .+= H[site]
        if site > 1
            coupling = J[site - 1]
            @inbounds for previous in 1:site-1, state in 1:q
                b[state] += coupling[previous, state, sequence[previous]]
            end
        end
    end
    return base
end

# log P(sequence | y) with feature k at each of `values` (standardized) and the other features at y_fixed.
function profile_values(network, sequence, y_fixed, k, values)
    H, G = network.H, network.G
    L, q = length(H), length(H[1])
    base = fill_base!(Matrix{Float64}(undef, q, L), network, sequence, y_fixed)
    logits = Vector{Float64}(undef, q)
    out = Vector{Float64}(undef, length(values))
    for (t, v) in enumerate(values)
        total = 0.0
        for site in 1:L
            slope = G[site]
            @inbounds for state in 1:q
                logits[state] = base[state, site] + slope[state, k] * v
            end
            total += logits[sequence[site]] - logsumexp(logits)
        end
        out[t] = total
    end
    return out
end

function check_against_reference(network, data, rows)
    T = length(data.grid)
    for n in rows[1:min(length(rows), 5)], t in unique([1, cld(T, 2), T])
        y = data.y_fixed[:, n]
        y[data.k] = data.standardized_grid[t]
        expected = reference_loglikelihood(network, view(data.Z, :, n), y)
        got = profile_values(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                             [data.standardized_grid[t]])[1]
        isapprox(got, expected; rtol=1e-9, atol=1e-9) ||
            error("Likelihood check failed for sequence $n, grid point $t: $got vs $expected")
    end
end

argmax_rows(profiles) = [argmax(view(profiles, n, :)) for n in axes(profiles, 1)]

# ------------------------------------------------------------------------------------------------
# Networks, weights, initialization
# ------------------------------------------------------------------------------------------------

# A network with all parameters at zero: a copy of `template` (keeps the ArNet type of --split-from) or a NamedTuple
# with the same fields H, G, J (all that infer_training_temperatures.jl uses).
function blank_network(template, L, q, d)
    if template !== nothing
        network = deepcopy(template)
        foreach(x -> fill!(x, 0.0), Iterators.flatten((network.H, network.G, network.J)))
        return network
    end
    return (H=[zeros(q) for _ in 1:L], G=[zeros(q, d) for _ in 1:L], J=[zeros(i, q, q) for i in 1:L-1])
end

# Phylogenetic weights: 1 / number of sequences of `rows` with identity >= 1 - theta (itself included)
function sequence_weights(Z, rows, theta)
    n = length(rows)
    theta <= 0 && return ones(n)
    L = size(Z, 1)
    max_mismatches = floor(Int, theta * L)
    counts = zeros(Int, n)
    Threads.@threads for a in 1:n
        sa = view(Z, :, rows[a])
        c = 0
        for b in 1:n
            sb = view(Z, :, rows[b])
            mismatches = 0
            @inbounds for i in 1:L
                mismatches += sa[i] != sb[i]
                mismatches > max_mismatches && break
            end
            c += mismatches <= max_mismatches
        end
        counts[a] = c
    end
    return 1 ./ counts
end

# Fields from the weighted single-site frequencies (the maximum-likelihood fields when G = J = 0), G = J = 0
function init_frequencies!(network, Z, rows, weights; pseudocount=0.01)
    q = length(network.H[1])
    total = sum(weights)
    for site in eachindex(network.H)
        f = zeros(q)
        for (r, n) in enumerate(rows)
            f[Z[site, n]] += weights[r]
        end
        f ./= total
        @. f = (1 - pseudocount) * f + pseudocount / q
        network.H[site] .= log.(f)
    end
    foreach(x -> fill!(x, 0.0), network.G)
    foreach(x -> fill!(x, 0.0), network.J)
    return network
end

zero_gradient(network) = (H=[zero(x) for x in network.H], G=[zero(x) for x in network.G], J=[zero(x) for x in network.J])

# ------------------------------------------------------------------------------------------------
# Loss and analytic gradient
# ------------------------------------------------------------------------------------------------

const NO_GRID = Float64[]

function make_workspace(network, n_grid)
    L, q = length(network.H), length(network.H[1])
    return (gH=[zero(x) for x in network.H], gG=[zero(x) for x in network.G], gJ=[zero(x) for x in network.J],
            base=Matrix{Float64}(undef, q, L), probs=Array{Float64}(undef, q, n_grid, L),
            ll=Vector{Float64}(undef, n_grid), P=Vector{Float64}(undef, n_grid),
            A=Vector{Float64}(undef, q), B=Vector{Float64}(undef, q), e=Vector{Float64}(undef, q))
end

# Adds the gradient of   w * [ -log P(s | y_natural)  -  beta * log P(T_true | s) ]   to ws.gH, ws.gG, ws.gJ.
#
# With P(t | s) = softmax_t log P(s | T_t), the second term's gradient is  beta * w * sum_t (P_t - [t == true]) g(T_t),
# where g(T) = d log P(s | T). Every g(T) is, per site, (onehot(s_i) - softmax(site logits at T)) mapped onto the
# parameters: H_i directly, G_i[:, k] times T, G_i[:, f] times y_fixed[f], J_{i-1}[j, :, s_j] directly. So all the
# terms are summed per site into A (everything except the scanned slope) and B (the scanned slope) and applied once.
# With an empty grid (beta = 0) only the generative term is computed.
function accumulate_sequence!(ws, network, sequence, y_fixed, k, y_natural, true_bin, standardized_grid, w, beta)
    H, G = network.H, network.G
    L, q, T = length(H), length(H[1]), length(standardized_grid)
    base, probs, ll, P, A, B, e = ws.base, ws.probs, ws.ll, ws.P, ws.A, ws.B, ws.e
    fill_base!(base, network, sequence, y_fixed)

    # Pass 1: softmax of every site at every grid temperature, and log P(s | T_t)
    if T > 0
        fill!(ll, 0.0)
        @inbounds for site in 1:L
            slope = G[site]
            s_i = sequence[site]
            for t in 1:T
                v = standardized_grid[t]
                mx = -Inf
                for a in 1:q
                    x = base[a, site] + slope[a, k] * v
                    e[a] = x
                    mx = max(mx, x)
                end
                z = 0.0
                for a in 1:q
                    e[a] = exp(e[a] - mx)
                    z += e[a]
                end
                for a in 1:q
                    probs[a, t, site] = e[a] / z
                end
                ll[t] += (base[s_i, site] + slope[s_i, k] * v) - mx - log(z)
            end
        end
        top = maximum(ll)
        total = 0.0
        @inbounds for t in 1:T
            P[t] = exp(ll[t] - top)
            total += P[t]
        end
        P ./= total
    end

    # Pass 2: per-site sums of the coefficient-weighted (onehot - softmax) terms
    @inbounds for site in 1:L
        slope = G[site]
        s_i = sequence[site]
        fill!(A, 0.0)
        fill!(B, 0.0)

        # generative term at the natural value, coefficient -w
        v = y_natural
        mx = -Inf
        for a in 1:q
            x = base[a, site] + slope[a, k] * v
            e[a] = x
            mx = max(mx, x)
        end
        z = 0.0
        for a in 1:q
            e[a] = exp(e[a] - mx)
            z += e[a]
        end
        c = -w
        for a in 1:q
            p = e[a] / z
            A[a] -= c * p
            B[a] -= c * p * v
        end
        A[s_i] += c
        B[s_i] += c * v

        # temperature term, coefficient beta * w * (P_t - [t == true]) at grid point t
        for t in 1:T
            c = beta * w * (P[t] - (t == true_bin))
            c == 0.0 && continue
            v = standardized_grid[t]
            for a in 1:q
                p = probs[a, t, site]
                A[a] -= c * p
                B[a] -= c * p * v
            end
            A[s_i] += c
            B[s_i] += c * v
        end

        # map onto the parameters
        gH = ws.gH[site]
        for a in 1:q
            gH[a] += A[a]
        end
        gG = ws.gG[site]
        for f in axes(gG, 2)
            if f == k
                for a in 1:q
                    gG[a, f] += B[a]
                end
            elseif y_fixed[f] != 0
                yf = y_fixed[f]
                for a in 1:q
                    gG[a, f] += yf * A[a]
                end
            end
        end
        if site > 1
            gJ = ws.gJ[site - 1]
            for previous in 1:site-1
                b = sequence[previous]
                for a in 1:q
                    gJ[previous, a, b] += A[a]
                end
            end
        end
    end
    return nothing
end

# Fills `grad` with the gradient of the weighted mean loss over `batch` plus the L2 terms.
# `weights` is indexed by data column.
function loss_and_gradient!(grad, workspaces, network, data, batch, weights, beta, hyper)
    total_weight = sum(weights[n] for n in batch)
    grid = beta == 0 ? NO_GRID : data.train_grid_std
    n_chunks = min(length(workspaces), length(batch))
    chunks = collect(Iterators.partition(batch, cld(length(batch), n_chunks)))
    @sync for (c, chunk) in enumerate(chunks)
        Threads.@spawn begin
            ws = workspaces[c]
            foreach(x -> fill!(x, 0.0), ws.gH)
            foreach(x -> fill!(x, 0.0), ws.gG)
            foreach(x -> fill!(x, 0.0), ws.gJ)
            for n in chunk
                accumulate_sequence!(ws, network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                                     data.y_natural[n], beta == 0 ? 0 : data.train_nearest[n], grid,
                                     weights[n] / total_weight, beta)
            end
        end
    end
    for (g, p) in zip(grad.H, network.H); g .= (2 * hyper.lambda_h) .* p; end
    for (g, p) in zip(grad.G, network.G); g .= (2 * hyper.lambda_g) .* p; end
    for (g, p) in zip(grad.J, network.J); g .= (2 * hyper.lambda_j) .* p; end
    for c in eachindex(chunks)
        ws = workspaces[c]
        for (g, x) in zip(grad.H, ws.gH); g .+= x; end
        for (g, x) in zip(grad.G, ws.gG); g .+= x; end
        for (g, x) in zip(grad.J, ws.gJ); g .+= x; end
    end
    return grad
end

# The same loss, computed directly (for the gradient check).
function loss_value(network, data, batch, weights, beta, hyper)
    total_weight = sum(weights[n] for n in batch)
    grid = beta == 0 ? NO_GRID : data.train_grid_std
    total = 0.0
    for n in batch
        w = weights[n] / total_weight
        values = profile_values(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k, [data.y_natural[n]; grid])
        total -= w * values[1]
        if beta != 0
            grid_ll = view(values, 2:length(values))
            total -= beta * w * (grid_ll[data.train_nearest[n]] - logsumexp(grid_ll))
        end
    end
    return total + hyper.lambda_h * sum(sum(abs2, x) for x in network.H) +
           hyper.lambda_g * sum(sum(abs2, x) for x in network.G) +
           hyper.lambda_j * sum(sum(abs2, x) for x in network.J)
end

# Central differences at three step sizes, keeping the closest: with a large beta the loss is large, so round-off
# dominates small steps and truncation large ones.
function gradient_check(network, data, rows, weights, beta, hyper; rng, n_checks=18, epsilons=(1e-4, 1e-5, 1e-6))
    grad = zero_gradient(network)
    workspaces = [make_workspace(network, beta == 0 ? 0 : length(data.train_grid_std)) for _ in 1:Threads.nthreads()]
    loss_and_gradient!(grad, workspaces, network, data, rows, weights, beta, hyper)
    L, q = length(network.H), length(network.H[1])
    worst = 0.0
    for c in 1:n_checks
        kind = (:H, :G, :J)[mod1(c, 3)]
        sequence = view(data.Z, :, rand(rng, rows))
        if kind == :H
            site = rand(rng, 1:L)
            array, g, index = network.H[site], grad.H[site], CartesianIndex(rand(rng, 1:q))
            label = "H[$site][$(index[1])]"
        elseif kind == :G
            site = rand(rng, 1:L)
            index = CartesianIndex(rand(rng, 1:q), rand(rng, axes(network.G[site], 2)))
            array, g = network.G[site], grad.G[site]
            label = "G[$site][$(index[1]),$(index[2])]"
        else
            site = rand(rng, 2:L)
            previous = rand(rng, 1:site-1)
            index = CartesianIndex(previous, rand(rng, 1:q), Int(sequence[previous]))   # an entry the batch uses
            array, g = network.J[site - 1], grad.J[site - 1]
            label = "J[$(site - 1)][$(index[1]),$(index[2]),$(index[3])]"
        end
        original = array[index]
        analytic = g[index]
        numeric, relative = NaN, Inf
        for epsilon in epsilons
            array[index] = original + epsilon
            plus = loss_value(network, data, rows, weights, beta, hyper)
            array[index] = original - epsilon
            minus = loss_value(network, data, rows, weights, beta, hyper)
            array[index] = original
            estimate = (plus - minus) / (2 * epsilon)
            mismatch = abs(estimate - analytic) / max(abs(estimate) + abs(analytic), 1e-8)
            if mismatch < relative
                numeric, relative = estimate, mismatch
            end
        end
        worst = max(worst, relative)
        @printf("  %-24s analytic % .6e   numeric % .6e   relative error %.1e\n", label, analytic, numeric, relative)
    end
    @printf("Largest relative error: %.1e (%s)\n", worst, worst < 1e-4 ? "gradient OK" : "CHECK THE GRADIENT")
    return worst
end

# ------------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------------

mutable struct AdamOptimizer
    params::Vector{Any}
    grads::Vector{Any}
    m::Vector{Any}
    v::Vector{Any}
    t::Int
end

function AdamOptimizer(network, grad)
    params = Any[]                                   # the network's own arrays, updated in place
    grads = Any[]
    for (p, g) in ((network.H, grad.H), (network.G, grad.G), (network.J, grad.J))
        append!(params, p)
        append!(grads, g)
    end
    return AdamOptimizer(params, grads, Any[zero(p) for p in params], Any[zero(p) for p in params], 0)
end

function adam_update!(p, g, m, v, eta, b1, b2, eps, c1, c2)
    @. m = b1 * m + (1 - b1) * g
    @. v = b2 * v + (1 - b2) * g^2
    @. p = p - eta * (m / c1) / (sqrt(v / c2) + eps)
    return nothing
end

function adam_step!(opt::AdamOptimizer, eta; b1=0.9, b2=0.999, eps=1e-8)
    opt.t += 1
    c1, c2 = 1 - b1^opt.t, 1 - b2^opt.t
    for i in eachindex(opt.params)
        adam_update!(opt.params[i], opt.grads[i], opt.m[i], opt.v[i], eta, b1, b2, eps, c1, c2)
    end
end

# `weights` is indexed by data column; on_epoch(epoch) is called after every epoch.
function train_epochs!(network, data, rows, weights, beta, hyper, rng; epochs, on_epoch=nothing)
    epochs <= 0 && return network
    grad = zero_gradient(network)
    n_grid = beta == 0 ? 0 : length(data.train_grid_std)
    workspaces = [make_workspace(network, n_grid) for _ in 1:Threads.nthreads()]
    opt = AdamOptimizer(network, grad)
    for epoch in 1:epochs
        for batch in Iterators.partition(shuffle(rng, rows), hyper.batch_size)
            loss_and_gradient!(grad, workspaces, network, data, batch, weights, beta, hyper)
            adam_step!(opt, hyper.eta)
        end
        on_epoch === nothing || on_epoch(epoch)
    end
    return network
end

# ------------------------------------------------------------------------------------------------
# Evaluation and output
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

# Likelihood profiles on the evaluation grid, per-sequence and per-organism recovery (homolog profiles averaged,
# sequences without taxonomy_id left out at the organism level), mean_ce and the log-likelihood at the natural value.
function evaluate_rows(network, data, rows)
    profiles = Matrix{Float64}(undef, length(rows), length(data.grid))
    natural_ll = Vector{Float64}(undef, length(rows))
    Threads.@threads for r in eachindex(rows)
        n = rows[r]
        values = profile_values(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                                [data.y_natural[n]; data.standardized_grid])
        natural_ll[r] = values[1]
        profiles[r, :] .= view(values, 2:length(values))
    end
    natural = data.natural[rows]
    sequence = recovery_metrics(natural, data.grid[argmax_rows(profiles)], data.grid)
    mean_ce = mean(logsumexp(view(profiles, r, :)) - profiles[r, data.nearest[rows[r]]] for r in eachindex(rows))

    groups = Dict{String,Vector{Int}}()
    for (r, n) in enumerate(rows)
        isempty(data.taxa[n]) || push!(get!(groups, data.taxa[n], Int[]), r)
    end
    taxa = sort!(collect(keys(groups)))
    members = [groups[t] for t in taxa]
    organism = nothing
    organism_natural = Float64[]
    organism_inferred = Float64[]
    if length(members) >= 3
        organism_profiles = reduce(vcat, [mean(view(profiles, m, :); dims=1) for m in members])
        organism_natural = [mean(natural[m]) for m in members]
        organism_inferred = data.grid[argmax_rows(organism_profiles)]
        organism = recovery_metrics(organism_natural, organism_inferred, data.grid)
    end
    return (profiles=profiles, natural_ll=natural_ll, sequence=sequence, organism=organism, mean_ce=mean_ce,
            loglik_per_site=mean(natural_ll) / data.L, organism_taxa=taxa, organism_members=members,
            organism_natural=organism_natural, organism_inferred=organism_inferred)
end

function selection_score(ev, select_by)
    select_by == "mean_ce" && return ev.mean_ce
    select_by == "organism_mae" && ev.organism !== nothing && return ev.organism.mae
    return ev.sequence.mae
end

fmt(x) = @sprintf("%g", x)
beta_label(beta) = "beta" * replace(fmt(beta), "." => "p")
with_commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

function csv_escape(value)
    text = string(value)
    any(c -> occursin(c, text), [',', '"', '\n']) || return text
    return "\"" * replace(text, "\"" => "\"\"") * "\""
end

function log_validation(io, stage, beta, epoch, seconds, ev)
    org = ev.organism
    @printf("  %-8s beta %-7s epoch %4d   val MAE %5.2f C  Spearman %.3f  organism MAE %s  mean_ce %6.3f  loglik/site %.4f  %6.0f s\n",
            stage, fmt(beta), epoch, ev.sequence.mae, ev.sequence.spearman,
            org === nothing ? "   n/a " : @sprintf("%5.2f C", org.mae), ev.mean_ce, ev.loglik_per_site, seconds)
    println(io, join(Any[stage, beta, epoch, round(seconds; digits=1), ev.sequence.mae, ev.sequence.spearman,
                         org === nothing ? "" : org.mae, org === nothing ? "" : org.spearman, ev.mean_ce,
                         ev.loglik_per_site], ','))
    flush(io)
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

function model_record(network, train_ids, validation_ids, test_ids, names, means, scales, fasta, metadata_path, settings)
    return (network=network, train_sequence_ids=train_ids, validation_sequence_ids=validation_ids,
            test_sequence_ids=test_ids, feature_names=names, feature_means=means, feature_scales=scales,
            source_fasta=fasta, source_metadata=metadata_path, temperature_term=settings)
end

# ------------------------------------------------------------------------------------------------

function main(args=ARGS)
    options = parse_args(args)
    rng = MersenneTwister(options["seed"])

    # --split-from: training set, features, standardization, input files, native reference
    split_model = nothing
    pool_from_model = String[]
    names, means, scales = String[], nothing, nothing
    if options["split_from"] !== nothing
        isfile(options["split_from"]) || error("Model not found: $(options["split_from"])")
        split_model = deserialize(options["split_from"])
        hasproperty(split_model, :network) ||
            error("$(options["split_from"]) has no `network`; expected a model saved by the TemperatureDCA scripts")
        pool_from_model, names, means, scales = model_description(split_model)
    else
        names = String.(strip.(split(options["features"], ',')))
    end
    feature = something(options["feature"], "Topt_ave" in names ? "Topt_ave" : first(names))
    k = findfirst(==(feature), names)
    k === nothing && error("$feature is not a model feature (features: $(join(names, ", ")))")
    d = length(names)

    if split_model === nothing
        fasta = something(options["fasta"], DEFAULT_FASTA)
        metadata_path = something(options["metadata"], DEFAULT_METADATA)
    else
        fasta = input_path(options["fasta"], split_model, :source_fasta, DEFAULT_FASTA, "fasta")
        metadata_path = input_path(options["metadata"], split_model, :source_metadata, DEFAULT_METADATA, "metadata")
    end
    isfile(fasta) || error("FASTA not found: $fasta")
    isfile(metadata_path) || error("Metadata not found: $metadata_path")
    if options["output_dir"] !== nothing
        output_dir = options["output_dir"]
    elseif split_model === nothing
        output_dir = joinpath(DEFAULT_OUTPUT_ROOT, "temperature_term_heldout_" * lowercase(feature))
    else
        stem = splitext(basename(options["split_from"]))[1]
        output_dir = joinpath(dirname(options["split_from"]), stem * "_temperature_term")
    end
    mkpath(output_dir)
    mkpath(joinpath(output_dir, "sweep_models"))

    # Sequences, read the way the training scripts read them
    metadata, metadata_columns = read_metadata(metadata_path)
    for name in names
        name in metadata_columns || error("The metadata has no $name column")
    end
    _, Z, L, _, _, annotations = FeatureDCA.read_annotated_fasta(fasta; remove_dups=false, max_gap_fraction=1.0)
    fasta_ids = [sequence_id(a) for a in annotations]
    column_of = Dict(id => m for (m, id) in enumerate(fasta_ids))
    length(column_of) == length(fasta_ids) || error("The FASTA has duplicate sequence IDs")
    has_features(id) = haskey(metadata, id) && all(name -> parsed_value(metadata, id, name) !== nothing, names)
    usable = filter(has_features, fasta_ids)
    usable_set = Set(usable)

    # Training pool (training + validation organisms) and test set
    if split_model === nothing
        groups = unique([split_group(metadata, id) for id in usable])
        shuffle!(rng, groups)
        n_test = round(Int, options["test_fraction"] * length(groups))
        1 <= n_test < length(groups) || error("--test-fraction leaves no training or no test organisms")
        test_groups = Set(groups[1:n_test])
        pool_ids = [id for id in usable if !(split_group(metadata, id) in test_groups)]
        test_ids = [id for id in usable if split_group(metadata, id) in test_groups]
    else
        missing_pool = filter(id -> !(id in usable_set), pool_from_model)
        isempty(missing_pool) || error("$(length(missing_pool)) training sequences of --split-from are not in the FASTA " *
                                       "or lack a feature value in the metadata (e.g. $(first(missing_pool)))")
        pool_ids = pool_from_model
        pool_set = Set(pool_ids)
        candidates = hasproperty(split_model, :test_sequence_ids) ? String.(collect(split_model.test_sequence_ids)) :
                     [id for id in usable if !(id in pool_set)]
        candidates = filter(id -> id in usable_set && !(id in pool_set), candidates)
        pool_groups = Set(split_group(metadata, id) for id in pool_ids)
        shared = count(id -> split_group(metadata, id) in pool_groups, candidates)
        shared > 0 && @warn "$shared test sequences come from training organisms and are left out"
        test_ids = filter(id -> !(split_group(metadata, id) in pool_groups), candidates)
    end
    isempty(test_ids) && error("The test set is empty")

    # Validation organisms, out of the training pool
    pool_group_list = unique([split_group(metadata, id) for id in pool_ids])
    shuffle!(rng, pool_group_list)
    n_val = round(Int, options["val_fraction"] * length(pool_group_list))
    1 <= n_val < length(pool_group_list) || error("--val-fraction leaves no validation or no training organisms")
    val_groups = Set(pool_group_list[1:n_val])
    val_ids = [id for id in pool_ids if split_group(metadata, id) in val_groups]
    fit_ids = [id for id in pool_ids if !(split_group(metadata, id) in val_groups)]

    # Data columns: fit, then validation, then test
    all_ids = vcat(fit_ids, val_ids, test_ids)
    Z_used = Z[:, [column_of[id] for id in all_ids]]
    fit_rows = collect(1:length(fit_ids))
    val_rows = collect(length(fit_ids) .+ (1:length(val_ids)))
    test_rows = collect((length(fit_ids) + length(val_ids)) .+ (1:length(test_ids)))
    pool_rows = vcat(fit_rows, val_rows)

    if split_model !== nothing
        L == length(split_model.network.H) ||
            error("The FASTA has $L alignment columns but --split-from has $(length(split_model.network.H)) sites")
        q = length(split_model.network.H[1])
    else
        q = Int(maximum(Z_used))
    end
    maximum(Z_used) <= q || error("The sequences use more states than q = $q")

    # Features: natural values, standardization (from --split-from, else from the training pool)
    values = Float64[parsed_value(metadata, id, name) for name in names, id in all_ids]   # d x N
    if means === nothing
        means = [mean(values[j, pool_rows]) for j in 1:d]
        scales = [std(values[j, pool_rows]) for j in 1:d]
        any(iszero, scales) && error("A feature is constant over the training organisms")
    end
    standardized = (values .- means) ./ scales
    y_fixed = copy(standardized)
    y_fixed[k, :] .= 0
    natural = values[k, :]

    step = options["grid_step"]
    grid_min = something(options["grid_min"], floor(minimum(natural) / step) * step)
    grid_max = something(options["grid_max"], ceil(maximum(natural) / step) * step)
    grid_min <= grid_max || error("--grid-min must not exceed --grid-max")
    grid = collect(grid_min:step:grid_max)
    train_step = options["train_grid_step"]
    train_lo = floor(minimum(natural[pool_rows]) / train_step) * train_step
    train_hi = ceil(maximum(natural[pool_rows]) / train_step) * train_step
    train_grid = collect(train_lo:train_step:train_hi)

    data = (Z=Z_used, y_fixed=y_fixed, k=k, y_natural=standardized[k, :], natural=natural, L=L, q=q,
            taxa=[taxonomy_of(metadata, id) for id in all_ids],
            grid=grid, standardized_grid=(grid .- means[k]) ./ scales[k], nearest=[argmin(abs.(grid .- t)) for t in natural],
            train_grid_std=(train_grid .- means[k]) ./ scales[k], train_nearest=[argmin(abs.(train_grid .- t)) for t in natural])

    betas = options["betas"] === nothing ? Float64[0, 0.1, 0.3, 1, 3] :
            parse.(Float64, strip.(split(options["betas"], ',')))
    any(<(0), betas) && error("--betas must not be negative")
    betas = sort!(unique!(push!(betas, 0.0)))
    hyper = (lambda_h=options["lambda_h"], lambda_g=options["lambda_g"], lambda_j=options["lambda_j"],
             eta=options["eta"], batch_size=options["batch_size"])
    theta = options["theta"]
    E1, E2, report_every, select_by = options["pretrain_epochs"], options["epochs"], options["report_every"], options["select_by"]
    template = split_model === nothing ? nothing : split_model.network

    n_organisms(ids) = length(unique([split_group(metadata, id) for id in ids]))
    @printf("FASTA: %s (%d sites, q = %d)\nFeatures: %s (temperature term on %s)\n", fasta, L, q, join(names, ", "), feature)
    split_model === nothing || println("Split, standardization and native reference: $(options["split_from"])")
    @printf("Training:   %s sequences, %s organisms\n", with_commas(length(fit_ids)), with_commas(n_organisms(fit_ids)))
    @printf("Validation: %s sequences, %s organisms\n", with_commas(length(val_ids)), with_commas(n_organisms(val_ids)))
    @printf("Test:       %s sequences, %s organisms\n", with_commas(length(test_ids)), with_commas(n_organisms(test_ids)))
    @printf("Training grid: %s to %s C in steps of %s C (%d points); evaluation grid: %s to %s C in steps of %s C (%d points)\n",
            fmt(first(train_grid)), fmt(last(train_grid)), fmt(train_step), length(train_grid),
            fmt(first(grid)), fmt(last(grid)), fmt(step), length(grid))
    @printf("betas: %s; lambda_h = %g, lambda_g = %g, lambda_j = %g, theta = %g, eta = %g, batch %d, %d threads\n\n",
            join(fmt.(betas), ", "), hyper.lambda_h, hyper.lambda_g, hyper.lambda_j, theta, hyper.eta, hyper.batch_size,
            Threads.nthreads())

    split_model === nothing || check_against_reference(split_model.network, data, test_rows)

    started = time()
    weights_fit = zeros(length(all_ids))
    weights_fit[fit_rows] .= sequence_weights(Z_used, fit_rows, theta)
    @printf("Weights: effective number of training sequences %.1f (%.1f s)\n", sum(weights_fit), time() - started)

    if options["gradcheck"]
        network = init_frequencies!(blank_network(template, L, q, d), Z_used, fit_rows, weights_fit[fit_rows])
        for x in Iterators.flatten((network.G, network.J))
            randn!(rng, x)
            x .*= 0.1
        end
        rows = fit_rows[1:min(8, length(fit_rows))]
        println("Gradient check (analytic vs central finite differences) on $(length(rows)) sequences, beta = $(fmt(maximum(betas))):")
        gradient_check(network, data, rows, weights_fit, maximum(betas), hyper; rng=rng)
        return
    end

    open(joinpath(output_dir, "split.csv"), "w") do io
        println(io, "sequence_id,taxonomy_id,set,weight")
        sets = vcat(fill("train", length(fit_ids)), fill("validation", length(val_ids)), fill("test", length(test_ids)))
        for (n, id) in enumerate(all_ids)
            println(io, join(csv_escape.(Any[id, data.taxa[n], sets[n], sets[n] == "train" ? weights_fit[n] : ""]), ','))
        end
    end

    log_io = open(joinpath(output_dir, "training_log.csv"), "w")
    println(log_io, "stage,beta,epoch,seconds,val_mae,val_spearman,val_organism_mae,val_organism_spearman," *
                    "val_mean_ce,val_loglik_per_site")

    # 2. beta = 0 pretraining on the training organisms
    println("\nPretraining with beta = 0 ($E1 epochs):")
    base = init_frequencies!(blank_network(template, L, q, d), Z_used, fit_rows, weights_fit[fit_rows])
    pretrain_started = time()
    pretrain_report = epoch -> begin
        if epoch % report_every == 0 || epoch == E1
            log_validation(log_io, "pretrain", 0.0, epoch, time() - pretrain_started, evaluate_rows(base, data, val_rows))
        end
        nothing
    end
    train_epochs!(base, data, fit_rows, weights_fit, 0.0, hyper, rng; epochs=E1, on_epoch=pretrain_report)
    base_eval = evaluate_rows(base, data, val_rows)

    # 3. beta sweep from the pretrained model, best validation epoch per beta
    results = Dict{Float64,Any}()
    for beta in betas
        println("\nbeta = $(fmt(beta)) ($E2 epochs from the pretrained model):")
        network = deepcopy(base)
        best = Ref{Any}((score=selection_score(base_eval, select_by), epoch=0, network=deepcopy(base), evaluation=base_eval))
        sweep_started = time()
        sweep_report = epoch -> begin
            if epoch % report_every == 0 || epoch == E2
                ev = evaluate_rows(network, data, val_rows)
                log_validation(log_io, "sweep", beta, epoch, time() - sweep_started, ev)
                score = selection_score(ev, select_by)
                if score < best[].score
                    best[] = (score=score, epoch=epoch, network=deepcopy(network), evaluation=ev)
                end
            end
            nothing
        end
        train_epochs!(network, data, fit_rows, weights_fit, beta, hyper, rng; epochs=E2, on_epoch=sweep_report)
        results[beta] = best[]
        @printf("  best: epoch %d, validation %s = %.4f\n", results[beta].epoch, select_by, results[beta].score)
    end
    close(log_io)

    # 4. beta chosen on validation (ties: the smallest beta)
    best_beta = betas[argmin([results[b].score for b in betas])]

    settings_for(beta, epochs_beta, refit) = (beta=beta, feature=feature, pretrain_epochs=E1, beta_epochs=epochs_beta,
        lambda_h=hyper.lambda_h, lambda_g=hyper.lambda_g, lambda_j=hyper.lambda_j, theta=theta, eta=hyper.eta,
        batch_size=hyper.batch_size, train_grid_step=train_step, select_by=select_by, refit=refit,
        seed=options["seed"], split_from=something(options["split_from"], ""))

    open(joinpath(output_dir, "beta_sweep.csv"), "w") do io
        println(io, "beta,best_epoch,chosen,val_mae,val_spearman,val_organism_mae,val_organism_spearman,val_mean_ce,val_loglik_per_site")
        for b in betas
            ev = results[b].evaluation
            org = ev.organism
            println(io, join(Any[b, results[b].epoch, b == best_beta, ev.sequence.mae, ev.sequence.spearman,
                                 org === nothing ? "" : org.mae, org === nothing ? "" : org.spearman, ev.mean_ce,
                                 ev.loglik_per_site], ','))
            serialize(joinpath(output_dir, "sweep_models", beta_label(b) * ".jls"),
                      model_record(results[b].network, fit_ids, val_ids, test_ids, names, means, scales, fasta,
                                   metadata_path, settings_for(b, results[b].epoch, false)))
        end
    end

    println("\nValidation (best epoch per beta; chosen by $select_by):")
    @printf("  %-8s %6s %9s %9s %13s %9s %12s\n", "beta", "epoch", "MAE", "Spearman", "organism MAE", "mean_ce", "loglik/site")
    for b in betas
        ev = results[b].evaluation
        @printf("  %-8s %6d %7.2f C %9.3f %13s %9.3f %12.4f%s\n", fmt(b), results[b].epoch, ev.sequence.mae,
                ev.sequence.spearman, ev.organism === nothing ? "n/a" : @sprintf("%.2f C", ev.organism.mae),
                ev.mean_ce, ev.loglik_per_site, b == best_beta ? "   <- chosen" : "")
    end

    # 5. refit on training + validation organisms (or keep the sweep models), evaluate on the test organisms
    final_betas = unique([0.0, best_beta])
    final = Dict{Float64,Any}()
    if options["no_refit"]
        for b in final_betas
            final[b] = results[b].network
        end
        final_train_ids, final_val_ids = fit_ids, val_ids
    else
        weights_pool = zeros(length(all_ids))
        weights_pool[pool_rows] .= sequence_weights(Z_used, pool_rows, theta)
        for b in final_betas
            println("\nRefitting beta = $(fmt(b)) on training + validation organisms ($E1 + $(results[b].epoch) epochs) ...")
            network = init_frequencies!(blank_network(template, L, q, d), Z_used, pool_rows, weights_pool[pool_rows])
            train_epochs!(network, data, pool_rows, weights_pool, 0.0, hyper, rng; epochs=E1)
            train_epochs!(network, data, pool_rows, weights_pool, b, hyper, rng; epochs=results[b].epoch)
            final[b] = network
        end
        final_train_ids, final_val_ids = vcat(fit_ids, val_ids), String[]
    end

    entries = Any[]
    split_model === nothing || push!(entries, (label="native", title="Native FeatureDCA (--split-from)",
                                               network=split_model.network, beta=nothing))
    for b in final_betas
        title = b == 0 ? "beta = 0" : "beta = $(fmt(b)) (chosen on validation)"
        push!(entries, (label=beta_label(b), title=title, network=final[b], beta=b))
    end

    test_ids_ordered = all_ids[test_rows]
    test_evaluations = Dict{String,Any}()
    for entry in entries
        ev = evaluate_rows(entry.network, data, test_rows)
        test_evaluations[entry.label] = ev
        write_sequence_table(joinpath(output_dir, "test_likelihoods_$(entry.label).csv"), test_ids_ordered,
                             data.taxa[test_rows], natural[test_rows], grid[argmax_rows(ev.profiles)], grid, ev.profiles)
        panels = [(natural=natural[test_rows], inferred=grid[argmax_rows(ev.profiles)], metrics=ev.sequence,
                   title="Test sequences — $(entry.title)", count_label="Sequences per hexagon")]
        if ev.organism !== nothing
            organism_names = [get(metadata[test_ids_ordered[first(m)]], "organism", "") for m in ev.organism_members]
            write_organism_table(joinpath(output_dir, "test_organism_temperatures_$(entry.label).csv"), ev.organism_taxa,
                                 organism_names, length.(ev.organism_members), ev.organism_natural, ev.organism_inferred)
            push!(panels, (natural=ev.organism_natural, inferred=ev.organism_inferred, metrics=ev.organism,
                           title="Test organisms (homologs averaged)", count_label="Organisms per hexagon"))
        end
        plot_recovery(joinpath(output_dir, "test_recovery_$(entry.label)"), panels, feature, step)
        if entry.beta !== nothing
            serialize(joinpath(output_dir, "model_$(entry.label).jls"),
                      model_record(entry.network, final_train_ids, final_val_ids, test_ids, names, means, scales, fasta,
                                   metadata_path, settings_for(entry.beta, results[entry.beta].epoch, !options["no_refit"])))
        end
    end

    println("\nTest organisms ($(with_commas(length(test_ids))) sequences):")
    @printf("  %-34s %8s %9s %13s %18s %9s %12s\n", "model", "MAE", "Spearman", "organism MAE", "organism Spearman",
            "mean_ce", "loglik/site")
    for entry in entries
        ev = test_evaluations[entry.label]
        org = ev.organism
        @printf("  %-34s %6.2f C %9.3f %13s %18s %9.3f %12.4f\n", entry.title, ev.sequence.mae, ev.sequence.spearman,
                org === nothing ? "n/a" : @sprintf("%.2f C", org.mae), org === nothing ? "n/a" : @sprintf("%.3f", org.spearman),
                ev.mean_ce, ev.loglik_per_site)
    end
    if best_beta == 0
        println("\nbeta = 0 had the best validation score: the temperature term did not improve held-out recovery here.")
    end

    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "fasta=$fasta\nmetadata=$metadata_path\nfeatures=$(join(names, ','))\nfeature=$feature")
        println(io, "split_from=$(something(options["split_from"], ""))")
        println(io, "n_train=$(length(fit_ids))\nn_validation=$(length(val_ids))\nn_test=$(length(test_ids))")
        println(io, "betas=$(join(fmt.(betas), ','))\nchosen_beta=$(fmt(best_beta))\nselect_by=$select_by")
        println(io, "refit=$(!options["no_refit"])")
        for key in (:lambda_h, :lambda_g, :lambda_j, :eta, :batch_size)
            println(io, "$(key)=$(getproperty(hyper, key))")
        end
        println(io, "theta=$theta\npretrain_epochs=$E1\nepochs=$E2\ntrain_grid_step=$train_step\ngrid_step=$step")
        @printf(io, "grid_min=%.6f\ngrid_max=%.6f\nfeature_mean=%.10f\nfeature_scale=%.10f\n", first(grid), last(grid),
                means[k], scales[k])
        for b in betas
            println(io, "validation_$(beta_label(b))_best_epoch=$(results[b].epoch)")
            write_metrics(io, "validation_$(beta_label(b))_sequence_", results[b].evaluation.sequence)
        end
        for entry in entries
            ev = test_evaluations[entry.label]
            write_metrics(io, "test_$(entry.label)_sequence_", ev.sequence)
            ev.organism === nothing || write_metrics(io, "test_$(entry.label)_organism_", ev.organism)
            @printf(io, "test_%s_mean_ce=%.10f\ntest_%s_loglik_per_site=%.10f\n", entry.label, ev.mean_ce, entry.label,
                    ev.loglik_per_site)
        end
    end
    println("\nResults written to $output_dir")
end

main()
