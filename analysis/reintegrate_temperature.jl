#!/usr/bin/env julia
#
# Reintegration of the temperature feature by retraining ONLY the feature slopes G of a
# temperature-conditioned FeatureDCA model (fields H and couplings J are frozen).
#
# Idea (Calvanese et al., NAR 2025, adapted). For every training sequence the maximum-likelihood
# temperature is read off a grid exactly as in infer_training_temperatures.jl and compared with the
# organism's real temperature. ALL training sequences are then used to re-tune G with a signed
# compatibility term: sequences the model already places well (predicted temperature close to the real
# one) are REINFORCED, sequences it places badly are SUPPRESSED (treated as unreliable).
#
# Temperature-compatibility score (user's definition):
#     C(a, T) = log P(a | T) - log Pbar(a),     Pbar(a) = sum_t pi(T_t) P(a | T_t)   (uniform pi on the grid)
#
# Objective minimized (only G moves; H, J fixed):
#     Q(G) = L(G | H, J)  -  lambda * R(G)
#     L(G | H, J) = phylo-weighted mean over all training sequences of  -log P(s | y_s)   (generative loss)
#     R(G) = sum_b w(b) * C(b, T_real(b)),   w(b) = w_phylo(b) * ( |T_pred(b) - T_real(b)| <= tau ? +w_plus : -w_minus )
# A sequence with w(b) > 0 has its true-temperature compatibility pushed up (reinforced); one with
# w(b) < 0 has it pushed down (suppressed). The generative anchor L keeps G from drifting.
#
# Why this reuses the existing gradient. With a uniform grid prior,
#     log P(T | a) = log P(a | T) - logsumexp_t log P(a | T_t) = C(a, T) + const,
# so C and the grid log-posterior share the same gradient in H, G, J. The gradient of C(b, T_bin) w.r.t.
# the parameters is therefore exactly the temperature-term gradient of capacity_test_temperature.jl /
# evaluate_temperature_term_heldout.jl: per site, coefficient (P_t - [t == bin]) applied to
# (onehot(s_i) - softmax_i(T_t)), mapped onto H (directly), G[:, k] (times T_t), G[:, f] (times y_fixed[f])
# and J (directly). Here only the G part is kept (H and J are frozen), and the per-sequence coefficient is
# lambda * w(b) at bin = the real temperature.
#
# The close/far split is computed ONCE from the starting model and held fixed during retraining.
#
# Inputs are read exactly as in infer_training_temperatures.jl (model .jls, aligned FASTA, metadata CSV).
# The retrained model is saved so that
#     julia --project=. analysis/infer_training_temperatures.jl --model <out>/reintegrated_model.jls
# plots the before/after recovery on the whole training set.
#
# Run from the TemperatureDCA folder:
#   JULIA_NUM_THREADS=auto julia --project=. analysis/reintegrate_temperature.jl --model results/featuredca/<model>.jls

using FeatureDCA      # needed to deserialize the ArNet stored in the model
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_MODEL = joinpath(ROOT, "results", "featuredca", "jdomain_topt_ave.jls")
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_METADATA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.metadata.maxgap50.csv")

function usage(io=stdout)
    println(io, """
Reintegration of the temperature feature: retrain only the feature slopes G of a FeatureDCA model using
ALL training sequences, with a signed weight per sequence -- positive (reinforce) when the model's
maximum-likelihood temperature is close to the organism's real temperature, negative (suppress) when it
is far. H and J are frozen.

Usage:
  JULIA_NUM_THREADS=auto julia --project=. analysis/reintegrate_temperature.jl [options]

Inputs (as in infer_training_temperatures.jl):
  --model PATH         Serialized model (default: results/featuredca/jdomain_topt_ave.jls). Works with
                       train_jdomain_featuredca.jl models (sequence_ids) and the held-out scripts'
                       model.jls / foldN_model.jls (train_sequence_ids).
  --fasta PATH         Aligned FASTA the model was trained on (default: path stored in the model, else the
                       J-domain maxgap50 MSA)
  --metadata PATH      Matching metadata CSV (default: path stored in the model, else the J-domain metadata)
  --feature NAME       Feature to reintegrate (default: Topt_ave, else the first). Other features of a
                       multi-feature model are held at each sequence's own metadata value.

Signed reintegration weights (applied to every training sequence):
  --close-threshold T  A sequence is "close" (positive weight) if |predicted - real| <= T degrees C,
                       otherwise "far" (negative weight). Default: 2.0
  --w-plus VALUE       Magnitude of the positive weight for close sequences (default: 1.0)
  --w-minus VALUE      Magnitude of the negative weight for far sequences (default: 1.0)
  --lambda VALUE       Weight lambda of R(G) against the generative anchor L (default: 1.0). Both gradients
                       only move G and are of the same order, so lambda ~ 1 balances them.
  --theta VALUE        Phylogenetic reweighting w_phylo(b): sequences with identity >= 1 - theta share
                       their weight (default: 0.2; 0 = uniform weights)
  --l2-g VALUE         Small L2 on G to keep it from drifting (default: 1e-6)

Training (only G is updated):
  --eta VALUE          Adam learning rate (default: 0.01)
  --epochs N           Passes over the training set (default: 60)
  --batch-size N       Sequences per gradient step (default: 64)
  --report-every N     Evaluate (an evaluation subsample) every N epochs (default: 5)
  --seed N             Batch order, phylo subsampling of the progress eval (default: 1)

Grid (one grid is used for the close/far split, for C in the R term, and for the before/after recovery):
  --grid-min VALUE     Lowest grid temperature (default: training minimum rounded down to a grid step)
  --grid-max VALUE     Highest grid temperature (default: training maximum rounded up to a grid step)
  --grid-step VALUE    Grid spacing in degrees C (default: 1.0; 0.5 is finer, 2 is faster). Every sequence
                       now does a grid pass each step, so a coarser grid speeds training up.
  --output-dir PATH    Output folder (default: <model name>_reintegration next to the model)
  --gradcheck          Compare the analytic G-gradient with central finite differences and exit
  -h, --help           Show this help

Outputs (in --output-dir):
  training_log.csv                     evaluation-subsample metrics during retraining
  whole_before.csv, whole_after.csv    infer_training_temperatures-style tables for all training sequences,
                                       with the starting and the retrained G
  reintegrated_model.jls               the retrained model (readable by infer_training_temperatures.jl)
  summary.txt                          settings and before/after metrics (whole set, close subset, far subset)
""")
end

const FLOAT_OPTIONS = Set(["grid_min", "grid_max", "grid_step", "close_threshold", "lambda",
                           "w_plus", "w_minus", "theta", "l2_g", "eta"])
const INT_OPTIONS = Set(["epochs", "batch_size", "report_every", "seed"])
const PATH_OPTIONS = Set(["model", "fasta", "metadata", "output_dir"])
const STRING_OPTIONS = Set(["feature"])
const VALUE_OPTIONS = union(FLOAT_OPTIONS, INT_OPTIONS, PATH_OPTIONS, STRING_OPTIONS)

function parse_args(args)
    options = Dict{String,Any}(
        "model" => DEFAULT_MODEL, "fasta" => nothing, "metadata" => nothing, "feature" => nothing,
        "close_threshold" => 2.0, "w_plus" => 1.0, "w_minus" => 1.0, "lambda" => 1.0, "theta" => 0.2,
        "l2_g" => 1e-6, "eta" => 0.01, "epochs" => 60, "batch_size" => 64, "report_every" => 5, "seed" => 1,
        "grid_min" => nothing, "grid_max" => nothing, "grid_step" => 1.0, "output_dir" => nothing,
        "gradcheck" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            usage(); exit(0)
        elseif arg == "--gradcheck"
            options["gradcheck"] = true
        elseif startswith(arg, "--") && (replace(arg[3:end], "-" => "_") in VALUE_OPTIONS)
            i == length(args) && error("Missing value after $arg")
            key, value = replace(arg[3:end], "-" => "_"), args[i + 1]
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
    options["close_threshold"] >= 0 || error("--close-threshold must not be negative")
    options["lambda"] >= 0 || error("--lambda must not be negative")
    min(options["w_plus"], options["w_minus"]) >= 0 || error("--w-plus and --w-minus must not be negative")
    0 <= options["theta"] < 1 || error("--theta must be in [0, 1)")
    options["l2_g"] >= 0 || error("--l2-g must not be negative")
    options["eta"] > 0 || error("--eta must be positive")
    options["epochs"] >= 0 || error("--epochs must not be negative")
    options["batch_size"] >= 1 || error("--batch-size must be at least 1")
    options["report_every"] >= 1 || error("--report-every must be at least 1")
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

function stored_vector(model, plural::Symbol, singular::Symbol)
    hasproperty(model, plural) && return Float64.(collect(getproperty(model, plural)))
    hasproperty(model, singular) && return Float64[getproperty(model, singular)]
    error("The model stores neither `$plural` nor `$singular`")
end

function model_description(model)
    d = size(model.network.G[1], 2)
    ids = if hasproperty(model, :train_sequence_ids)
        model.train_sequence_ids
    elseif hasproperty(model, :sequence_ids)
        model.sequence_ids
    else
        error("The model has neither `train_sequence_ids` nor `sequence_ids`, so its training set is unknown")
    end
    names = if hasproperty(model, :feature_names)
        String.(collect(model.feature_names))
    elseif d == 1
        ["Topt_ave"]
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

# Straightforward autoregressive log-likelihood; used as a check.
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

# Site logits without the scanned feature: H + G * y_fixed (y_fixed[k] = 0) + couplings to preceding residues.
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

# log P(sequence | y) with feature k at each of `values` (standardized), other features at y_fixed.
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

# log P(sequence | T) for every sequence (rows) and grid value of feature k (columns). Threaded, no gradient.
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
# Reintegration gradient (only G is accumulated; H and J are frozen)
# ------------------------------------------------------------------------------------------------

function make_workspace(network, n_grid)
    L, q = length(network.H), length(network.H[1])
    return (gG=[zero(x) for x in network.G],
            base=Matrix{Float64}(undef, q, L), probs=Array{Float64}(undef, q, max(n_grid, 1), L),
            ll=Vector{Float64}(undef, n_grid), P=Vector{Float64}(undef, n_grid),
            A=Vector{Float64}(undef, q), B=Vector{Float64}(undef, q), e=Vector{Float64}(undef, q))
end

# Adds to ws.gG the G-gradient of
#     w_gen * ( -log P(s | y_natural) )                      [generative anchor; skipped when w_gen == 0]
#   + sum over (bin, coeff) in `targets` of  coeff * ( -log P(T_bin | s) )
# where -d log P(T_bin | s) = sum_t (P_t - [t == bin]) d log P(s | T_t), P_t = softmax_t log P(s | T_t).
# log P(T_bin|s) and C(s, T_bin) share this gradient, so a coeff > 0 pushes C(s, T_bin) up (reinforce),
# coeff < 0 pushes it down (suppress). Only the G entries are written.
function accumulate_sequence!(ws, network, sequence, y_fixed, k, y_natural, standardized_grid, w_gen, targets)
    H, G = network.H, network.G
    L, q, T = length(H), length(H[1]), length(standardized_grid)
    base, probs, ll, P, A, B, e = ws.base, ws.probs, ws.ll, ws.P, ws.A, ws.B, ws.e
    fill_base!(base, network, sequence, y_fixed)
    need_grid = !isempty(targets)

    # Pass 1: grid softmaxes p_i(T_t), grid log-likelihoods, and the posterior P_t = softmax_t log P(s|T_t)
    if need_grid
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

    # Pass 2: per-site A (plain onehot - softmax, maps to G[:, f != k]) and B (scanned slope, maps to G[:, k])
    @inbounds for site in 1:L
        slope = G[site]
        s_i = sequence[site]
        fill!(A, 0.0)
        fill!(B, 0.0)

        if w_gen != 0.0                       # generative term at the natural value, coefficient -w_gen
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
            c = -w_gen
            for a in 1:q
                p = e[a] / z
                A[a] -= c * p
                B[a] -= c * p * v
            end
            A[s_i] += c
            B[s_i] += c * v
        end

        if need_grid                          # reintegration targets
            for (bin, coeff) in targets
                coeff == 0.0 && continue
                for t in 1:T
                    c = coeff * (P[t] - (t == bin))
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
            end
        end

        gG = ws.gG[site]                      # map onto G only (H and J are frozen)
        for a in 1:q
            gG[a, k] += B[a]
        end
        for f in axes(gG, 2)
            f == k && continue
            yf = y_fixed[f]
            yf == 0 && continue
            for a in 1:q
                gG[a, f] += yf * A[a]
            end
        end
    end
    return nothing
end

# Per-sequence generative weight and (single, signed) reintegration target for data column n.
function sequence_terms(data, n)
    w_gen = data.W[n] / data.total_W
    (data.lambda == 0.0) && return w_gen, ()
    coeff = data.lambda * data.w_signed[n] * data.W[n] / data.total_W
    return w_gen, ((data.nat_bin[n], coeff),)
end

# Fills grad.G with the G-gradient of the generative loss over `batch` plus the signed reintegration terms,
# plus the L2 on G.
function loss_and_gradient!(grad, workspaces, network, data, batch)
    n_chunks = min(length(workspaces), length(batch))
    chunks = collect(Iterators.partition(batch, cld(length(batch), n_chunks)))
    @sync for (c, chunk) in enumerate(chunks)
        Threads.@spawn begin
            ws = workspaces[c]
            foreach(x -> fill!(x, 0.0), ws.gG)
            for n in chunk
                w_gen, targets = sequence_terms(data, n)
                accumulate_sequence!(ws, network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                                     data.y_natural[n], data.standardized_grid, w_gen, targets)
            end
        end
    end
    for (g, p) in zip(grad.G, network.G)
        g .= (2 * data.l2_g) .* p
    end
    for c in eachindex(chunks)
        ws = workspaces[c]
        for (g, x) in zip(grad.G, ws.gG)
            g .+= x
        end
    end
    return grad
end

# The same objective, computed directly (for the gradient check). Uses log P(T_bin | s), whose gradient
# equals that of C(s, T_bin); the constant difference is irrelevant to the check.
function loss_value(network, data, rows)
    total = 0.0
    for n in rows
        values = profile_values(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                                [data.y_natural[n]; data.standardized_grid])
        grid_ll = view(values, 2:length(values))
        wn = data.W[n] / data.total_W
        total += wn * (-values[1])
        if data.lambda != 0.0
            lp_nat = grid_ll[data.nat_bin[n]] - logsumexp(grid_ll)
            total += -data.lambda * data.w_signed[n] * wn * lp_nat
        end
    end
    return total + data.l2_g * sum(sum(abs2, x) for x in network.G)
end

function gradient_check(network, data, rows; rng, n_checks=18, epsilons=(1e-4, 1e-5, 1e-6))
    grad = (G=[zero(x) for x in network.G],)
    workspaces = [make_workspace(network, length(data.standardized_grid)) for _ in 1:Threads.nthreads()]
    loss_and_gradient!(grad, workspaces, network, data, rows)
    L, q = length(network.H), length(network.H[1])
    worst = 0.0
    for c in 1:n_checks
        site = rand(rng, 1:L)
        index = CartesianIndex(rand(rng, 1:q), rand(rng, axes(network.G[site], 2)))
        array, g = network.G[site], grad.G[site]
        label = "G[$site][$(index[1]),$(index[2])]"
        original = array[index]
        analytic = g[index]
        numeric, relative = NaN, Inf
        for epsilon in epsilons
            array[index] = original + epsilon
            plus = loss_value(network, data, rows)
            array[index] = original - epsilon
            minus = loss_value(network, data, rows)
            array[index] = original
            estimate = (plus - minus) / (2 * epsilon)
            mismatch = abs(estimate - analytic) / max(abs(estimate) + abs(analytic), 1e-8)
            if mismatch < relative
                numeric, relative = estimate, mismatch
            end
        end
        worst = max(worst, relative)
        @printf("  %-20s analytic % .6e   numeric % .6e   relative error %.1e\n", label, analytic, numeric, relative)
    end
    @printf("Largest relative error: %.1e (%s)\n", worst, worst < 1e-4 ? "gradient OK" : "CHECK THE GRADIENT")
    return worst
end

# ------------------------------------------------------------------------------------------------
# Adam over G only
# ------------------------------------------------------------------------------------------------

mutable struct AdamG
    params::Vector{Any}       # the network's own G matrices, updated in place
    grads::Vector{Any}
    m::Vector{Any}
    v::Vector{Any}
    t::Int
end

AdamG(params, grads) = AdamG(Any[p for p in params], Any[g for g in grads],
                             Any[zero(p) for p in params], Any[zero(p) for p in params], 0)

function adam_step!(opt::AdamG, eta; b1=0.9, b2=0.999, eps=1e-8)
    opt.t += 1
    c1, c2 = 1 - b1^opt.t, 1 - b2^opt.t
    for i in eachindex(opt.params)
        p, g, m, v = opt.params[i], opt.grads[i], opt.m[i], opt.v[i]
        @. m = b1 * m + (1 - b1) * g
        @. v = b2 * v + (1 - b2) * g^2
        @. p = p - eta * (m / c1) / (sqrt(v / c2) + eps)
    end
end

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
    return (n=length(natural),
            pearson=length(natural) > 1 ? cor(natural, inferred) : NaN,
            spearman=length(natural) > 1 ? spearman_correlation(natural, inferred) : NaN,
            mae=mean(abs.(errors)), rmse=sqrt(mean(abs2.(errors))), bias=mean(errors),
            within_2C=mean(abs.(errors) .<= 2), within_5C=mean(abs.(errors) .<= 5), at_grid_edge=mean(at_edge))
end

with_commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

function csv_escape(value)
    text = string(value)
    any(ch -> occursin(ch, text), [',', '"', '\n']) || return text
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

function write_metrics(io, prefix, m)
    for key in keys(m)
        value = getproperty(m, key)
        value isa Integer ? println(io, "$(prefix)$(key)=$(value)") : @printf(io, "%s%s=%.10f\n", prefix, key, value)
    end
end

function print_metrics(label, m)
    @printf("%s (n = %s)\n", label, with_commas(m.n))
    @printf("  MAE = %.2f C   RMSE = %.2f C   bias = %+.2f C   within 2 C: %.1f%%   within 5 C: %.1f%%\n",
            m.mae, m.rmse, m.bias, 100 * m.within_2C, 100 * m.within_5C)
    if m.n > 1
        @printf("  Pearson r = %.3f   Spearman rho = %.3f   argmax at a grid edge: %.1f%%\n",
                m.pearson, m.spearman, 100 * m.at_grid_edge)
    end
end

# Phylogenetic weights: 1 / number of training sequences with identity >= 1 - theta (itself included)
function sequence_weights(Z, theta)
    n = size(Z, 2)
    theta <= 0 && return ones(n)
    L = size(Z, 1)
    max_mismatches = floor(Int, theta * L)
    counts = zeros(Int, n)
    Threads.@threads for a in 1:n
        sa = view(Z, :, a)
        c = 0
        for b in 1:n
            sb = view(Z, :, b)
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

# Recovery metrics of the current G on a set of data columns.
function rows_metrics(network, data, rows)
    profiles = likelihood_profiles(network, data.Z[:, rows], data.k, data.y_fixed[:, rows], data.standardized_grid)
    inferred = data.grid[argmax_rows(profiles)]
    return recovery_metrics(data.natural[rows], inferred, data.grid)
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
    output_dir = something(options["output_dir"], joinpath(dirname(model_path), model_name * "_reintegration"))
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
    nat_bin = [argmin(abs.(grid .- t)) for t in natural]

    N = length(train_ids)
    @printf("Model: %s (features: %s)\n", model_path, join(names, ", "))
    length(names) > 1 && println("  reintegrating $feature; $(join(setdiff(names, [feature]), ", ")) held at the natural values")
    @printf("Training set: %s sequences x %d sites (q = %d) from %s\n", with_commas(N), L, q, fasta)
    @printf("Grid: %.2f to %.2f C in steps of %.2f C (%d temperatures), %d threads\n\n",
            first(grid), last(grid), step, length(grid), Threads.nthreads())

    # Initial maximum-likelihood temperatures (as infer_training_temperatures.jl) and the close/far split
    started = time()
    profiles0 = likelihood_profiles(network, Z_train, k, y_fixed, standardized_grid)
    inferred0 = grid[argmax_rows(profiles0)]
    @printf("Initial likelihoods computed in %.1f s\n", time() - started)

    abs_err = abs.(inferred0 .- natural)
    is_close = abs_err .<= options["close_threshold"]
    is_far = .!is_close
    w_signed = [is_close[n] ? options["w_plus"] : -options["w_minus"] for n in 1:N]
    @printf("Close (|predicted - real| <= %.1f C): %s sequences  (+%.2f)\n",
            options["close_threshold"], with_commas(count(is_close)), options["w_plus"])
    @printf("Far:                                  %s sequences  (-%.2f)\n",
            with_commas(count(is_far)), options["w_minus"])
    (count(is_close) == 0 || count(is_far) == 0) &&
        @warn "All training sequences fall on one side of --close-threshold $(options["close_threshold"]); adjust it for a mix"

    # Phylogenetic weights
    started = time()
    W = sequence_weights(Z_train, options["theta"])
    total_W = sum(W)
    @printf("Weights: effective number of training sequences %.1f (%.1f s)\n\n", total_W, time() - started)

    data = (Z=Z_train, y_fixed=y_fixed, k=k, y_natural=(natural .- means[k]) ./ scales[k], natural=natural,
            grid=grid, standardized_grid=standardized_grid, nat_bin=nat_bin,
            W=W, total_W=total_W, w_signed=w_signed, lambda=options["lambda"], l2_g=options["l2_g"],
            L=L, q=q, step=step)

    check_against_reference(network, data, collect(1:N))

    if options["gradcheck"]
        rng = MersenneTwister(options["seed"])
        close_rows = findall(is_close); far_rows = findall(is_far)
        rows = vcat(close_rows[1:min(5, length(close_rows))], far_rows[1:min(5, length(far_rows))])
        isempty(rows) && (rows = collect(1:min(8, N)))
        println("Gradient check (analytic vs central finite differences) on $(length(rows)) sequences " *
                "($(count(in(close_rows), rows)) close, $(count(in(far_rows), rows)) far):")
        gradient_check(network, data, rows; rng=rng)
        return
    end

    @printf("lambda = %g, w_plus = %g, w_minus = %g, close_threshold = %g, theta = %g, l2_g = %g, eta = %g, %d epochs, batch %d\n",
            options["lambda"], options["w_plus"], options["w_minus"], options["close_threshold"], options["theta"],
            options["l2_g"], options["eta"], options["epochs"], options["batch_size"])

    before_full = recovery_metrics(natural, inferred0, grid)
    before_close = recovery_metrics(natural[is_close], inferred0[is_close], grid)
    before_far = recovery_metrics(natural[is_far], inferred0[is_far], grid)

    # Evaluation subsample for the progress log (all if small, else a fixed random subset)
    rng = MersenneTwister(options["seed"])
    eval_rows = N <= 3000 ? collect(1:N) : sort(randperm(rng, N)[1:3000])

    # Retrain G
    grad = (G=[zero(x) for x in network.G],)
    workspaces = [make_workspace(network, length(standardized_grid)) for _ in 1:Threads.nthreads()]
    opt = AdamG(collect(network.G), grad.G)
    log_io = open(joinpath(output_dir, "training_log.csv"), "w")
    println(log_io, "epoch,seconds,eval_mae,eval_rmse,eval_bias,eval_within_2C,eval_spearman")
    log_eval(epoch, seconds, m) = begin
        @printf("  epoch %4d   eval MAE %5.2f C   within 2 C %5.1f%%   bias %+5.2f C   %6.0f s\n",
                epoch, m.mae, 100 * m.within_2C, m.bias, seconds)
        println(log_io, join(Any[epoch, round(seconds; digits=1), m.mae, m.rmse, m.bias, m.within_2C, m.spearman], ','))
        flush(log_io)
    end
    println("\nRetraining G (H and J frozen):")
    log_eval(0, 0.0, rows_metrics(network, data, eval_rows))
    all_rows = collect(1:N)
    train_started = time()
    for epoch in 1:options["epochs"]
        for batch in Iterators.partition(shuffle(rng, all_rows), options["batch_size"])
            loss_and_gradient!(grad, workspaces, network, data, batch)
            adam_step!(opt, options["eta"])
        end
        if epoch == 1 || epoch % options["report_every"] == 0 || epoch == options["epochs"]
            log_eval(epoch, time() - train_started, rows_metrics(network, data, eval_rows))
        end
    end
    close(log_io)

    # After: recompute on the whole training set and on the (fixed) close / far subsets
    profiles1 = likelihood_profiles(network, Z_train, k, y_fixed, standardized_grid)
    inferred1 = grid[argmax_rows(profiles1)]
    after_full = recovery_metrics(natural, inferred1, grid)
    after_close = recovery_metrics(natural[is_close], inferred1[is_close], grid)
    after_far = recovery_metrics(natural[is_far], inferred1[is_far], grid)

    println()
    print_metrics("Whole set   BEFORE", before_full)
    print_metrics("Whole set   AFTER ", after_full)
    println()
    print_metrics("Close subset BEFORE", before_close)
    print_metrics("Close subset AFTER ", after_close)
    println()
    print_metrics("Far subset   BEFORE", before_far)
    print_metrics("Far subset   AFTER ", after_far)
    println("\n(The far subset is expected to drift away from its labels by construction -- it carries negative weight.)")
    if after_full.mae > before_full.mae + 0.05
        println("Whole-set MAE rose: the suppression outweighed the reinforcement. Lower --lambda or --w-minus, " *
                "raise --close-threshold, or raise --l2-g.")
    end

    # Outputs
    taxa = [String(strip(get(metadata[id], "taxonomy_id", ""))) for id in train_ids]
    write_sequence_table(joinpath(output_dir, "whole_before.csv"), train_ids, taxa, natural, inferred0, grid, profiles0)
    write_sequence_table(joinpath(output_dir, "whole_after.csv"), train_ids, taxa, natural, inferred1, grid, profiles1)

    settings = (source_model=model_path, feature=feature, close_threshold=options["close_threshold"],
                n_close=count(is_close), n_far=count(is_far), w_plus=options["w_plus"], w_minus=options["w_minus"],
                lambda=options["lambda"], theta=options["theta"], l2_g=options["l2_g"], eta=options["eta"],
                epochs=options["epochs"], batch_size=options["batch_size"], seed=options["seed"],
                grid_min=first(grid), grid_max=last(grid), grid_step=step)
    id_field = hasproperty(model, :train_sequence_ids) ? :train_sequence_ids : :sequence_ids
    record = (network=network, feature_names=names, feature_means=means, feature_scales=scales,
              source_fasta=fasta, source_metadata=metadata_path, reintegration=settings)
    record = merge(record, NamedTuple{(id_field,)}((train_ids,)))
    serialize(joinpath(output_dir, "reintegrated_model.jls"), record)

    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "model=$model_path\nfasta=$fasta\nmetadata=$metadata_path\nfeature=$feature")
        println(io, "model_features=$(join(names, ','))")
        for key in keys(settings)
            key == :source_model || println(io, "$(key)=$(getproperty(settings, key))")
        end
        @printf(io, "feature_mean=%.10f\nfeature_scale=%.10f\n", means[k], scales[k])
        write_metrics(io, "full_before_", before_full)
        write_metrics(io, "full_after_", after_full)
        write_metrics(io, "close_before_", before_close)
        write_metrics(io, "close_after_", after_close)
        write_metrics(io, "far_before_", before_far)
        write_metrics(io, "far_after_", after_far)
    end
    println("\nResults written to $output_dir")
    println("Plot the whole-set before/after with:")
    println("  julia --project=. analysis/infer_training_temperatures.jl --model $(joinpath(output_dir, "reintegrated_model.jls"))")
end

main()
