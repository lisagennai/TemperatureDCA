#!/usr/bin/env julia
#
# Capacity test for the temperature pathway of a temperature-conditioned FeatureDCA model.
#
# The model's H, G, J are retrained on a subsample of its own training sequences with the hybrid objective
#
#     loss = mean_s [ -log P(s | y_s)  -  beta * log P(T_s | s) ]  +  lambda * (|G|^2 + |J|^2)
#
# where P(T | s) is the softmax over the temperature grid of log P(s | T) (uniform prior), i.e. the posterior
# whose argmax is the maximum-likelihood temperature of infer_training_temperatures.jl. The gradient is analytic:
#     d log P(T_s | s) = d log P(s | T_s) - sum_t P(t | s) d log P(s | T_t).
#
# Reading the result: if, once the temperature term is pushed (large beta, no regularization, small subsample), the
# grid argmax lands on the natural temperature of the training sequences, the field-only form has the capacity and
# the mismatches of the original model are under-fitting (case a). If it can't even then (and more epochs / other
# seeds don't change that), the form is the bottleneck (case b): temperature has to enter the couplings.
#
# Inputs are read exactly as in infer_training_temperatures.jl (model .jls, aligned FASTA, metadata CSV).
#
# Run from the TemperatureDCA folder:
#   JULIA_NUM_THREADS=auto julia --project=. analysis/capacity_test_temperature.jl --model results/featuredca/<model>.jls

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
Capacity test of the temperature pathway: retrain a FeatureDCA model on a subsample of its training sequences with
an extra term, beta * log P(T_natural | sequence), and check whether the maximum-likelihood temperature of those
sequences can be brought onto their natural temperature.

Usage:
  JULIA_NUM_THREADS=auto julia --project=. analysis/capacity_test_temperature.jl [options]

Inputs (as in infer_training_temperatures.jl):
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
  --grid-step VALUE    Grid spacing in degrees C (default: 0.5). The cost of a training step grows with the number
                       of grid points; --grid-step 1 or 2 is enough for this test.
  --output-dir PATH    Output folder (default: <model name>_capacity_test next to the model)

Training:
  --n-train N          Training sequences in the subsample (default: 300; 0 = all of them)
  --beta VALUE         Weight of the temperature term (default: 10 x number of sites; log P(s|T) is a sum over the
                       sites, log P(T|s) a single categorical, so beta ~ 1 would be invisible)
  --lambda VALUE       L2 regularization of G and J (default: 0, i.e. allowed to overfit)
  --eta VALUE          Adam learning rate (default: 0.01)
  --epochs N           Passes over the subsample (default: 200)
  --batch-size N       Sequences per gradient step (default: 32)
  --init model|random  Start from the model's parameters (default) or from small random ones
  --seed N             Subsample, batch order and random initialization (default: 1)
  --report-every N     Evaluate the subsample every N epochs (default: 10)
  --gradcheck          Compare the analytic gradient with finite differences on the model and exit
  -h, --help           Show this help

Outputs (in --output-dir):
  training_log.csv                       metrics of the subsample during training
  training_likelihoods_before.csv        same layout as infer_training_temperatures.jl, subsample only,
  training_likelihoods_after.csv           with the starting and the retrained parameters
  capacity_model.jls                     the retrained model; its training IDs are the subsample, so
                                         infer_training_temperatures.jl --model .../capacity_model.jls
                                         plots the recovery on exactly these sequences
  summary.txt
""")
end

const FLOAT_OPTIONS = Set(["grid_min", "grid_max", "grid_step", "beta", "lambda", "eta"])
const INT_OPTIONS = Set(["n_train", "epochs", "batch_size", "seed", "report_every"])
const PATH_OPTIONS = Set(["model", "fasta", "metadata", "output_dir"])
const STRING_OPTIONS = Set(["feature", "init"])
const VALUE_OPTIONS = union(FLOAT_OPTIONS, INT_OPTIONS, PATH_OPTIONS, STRING_OPTIONS)

function parse_args(args)
    options = Dict{String,Any}(
        "model" => DEFAULT_MODEL, "fasta" => nothing, "metadata" => nothing, "feature" => nothing,
        "grid_min" => nothing, "grid_max" => nothing, "grid_step" => 0.5, "output_dir" => nothing,
        "n_train" => 300, "beta" => nothing, "lambda" => 0.0, "eta" => 0.01, "epochs" => 200, "batch_size" => 32,
        "init" => "model", "seed" => 1, "report_every" => 10, "gradcheck" => false,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
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
    options["beta"] === nothing || options["beta"] >= 0 || error("--beta must not be negative")
    options["lambda"] >= 0 || error("--lambda must not be negative")
    options["eta"] > 0 || error("--eta must be positive")
    options["epochs"] >= 0 || error("--epochs must not be negative")
    options["batch_size"] >= 1 || error("--batch-size must be at least 1")
    options["n_train"] >= 0 || error("--n-train must not be negative")
    options["report_every"] >= 1 || error("--report-every must be at least 1")
    options["init"] in ("model", "random") || error("--init must be model or random")
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
function capacity_profile(network, sequence, y_fixed, k, values)
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
        got = capacity_profile(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                               [data.standardized_grid[t]])[1]
        isapprox(got, expected; rtol=1e-9, atol=1e-9) ||
            error("Likelihood check failed for sequence $n, grid point $t: $got vs $expected")
    end
end

argmax_rows(profiles) = [argmax(view(profiles, n, :)) for n in axes(profiles, 1)]

# Log-likelihood of the subsample at the natural value (column vector) and at every grid point (matrix).
function capacity_evaluate(network, data, rows)
    profiles = Matrix{Float64}(undef, length(rows), length(data.grid))
    natural_ll = Vector{Float64}(undef, length(rows))
    Threads.@threads for r in eachindex(rows)
        n = rows[r]
        values = capacity_profile(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                                  [data.y_natural[n]; data.standardized_grid])
        natural_ll[r] = values[1]
        profiles[r, :] .= view(values, 2:length(values))
    end
    return profiles, natural_ll
end

# ------------------------------------------------------------------------------------------------
# Loss and analytic gradient
# ------------------------------------------------------------------------------------------------

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
# terms are summed per site into A (everything except the scanned slope) and B (the scanned slope) and applied once;
# the O(L^2 q) coupling update is done once per site, not once per grid point.
function accumulate_sequence!(ws, network, sequence, y_fixed, k, y_natural, true_bin, standardized_grid, w, beta)
    H, G = network.H, network.G
    L, q, T = length(H), length(H[1]), length(standardized_grid)
    base, probs, ll, P, A, B, e = ws.base, ws.probs, ws.ll, ws.P, ws.A, ws.B, ws.e
    fill_base!(base, network, sequence, y_fixed)

    # Pass 1: softmax of every site at every grid temperature, and log P(s | T_t)
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

# Fills `grad` with the gradient of the mean loss over `batch` plus the L2 term (G and J; H is not regularized).
function loss_and_gradient!(grad, workspaces, network, data, batch, beta, lambda)
    n_chunks = min(length(workspaces), length(batch))
    chunks = collect(Iterators.partition(batch, cld(length(batch), n_chunks)))
    w = 1 / length(batch)
    @sync for (c, chunk) in enumerate(chunks)
        Threads.@spawn begin
            ws = workspaces[c]
            foreach(x -> fill!(x, 0.0), ws.gH)
            foreach(x -> fill!(x, 0.0), ws.gG)
            foreach(x -> fill!(x, 0.0), ws.gJ)
            for n in chunk
                accumulate_sequence!(ws, network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                            data.y_natural[n], data.nearest[n], data.standardized_grid, w, beta)
            end
        end
    end
    foreach(x -> fill!(x, 0.0), grad.H)
    for (g, p) in zip(grad.G, network.G)
        g .= (2 * lambda) .* p
    end
    for (g, p) in zip(grad.J, network.J)
        g .= (2 * lambda) .* p
    end
    for c in eachindex(chunks)
        ws = workspaces[c]
        for (g, x) in zip(grad.H, ws.gH); g .+= x; end
        for (g, x) in zip(grad.G, ws.gG); g .+= x; end
        for (g, x) in zip(grad.J, ws.gJ); g .+= x; end
    end
    return grad
end

# The same loss, computed directly (for the gradient check).
function loss_value(network, data, batch, beta, lambda)
    w = 1 / length(batch)
    total = 0.0
    for n in batch
        values = capacity_profile(network, view(data.Z, :, n), view(data.y_fixed, :, n), data.k,
                                  [data.y_natural[n]; data.standardized_grid])
        grid_ll = view(values, 2:length(values))
        total -= w * values[1]
        total -= beta * w * (grid_ll[data.nearest[n]] - logsumexp(grid_ll))
    end
    return total + lambda * (sum(sum(abs2, x) for x in network.G) + sum(sum(abs2, x) for x in network.J))
end

function gradient_check(network, data, rows, beta, lambda; rng, n_checks=18, epsilon=1e-5)
    grad = (H=[zero(x) for x in network.H], G=[zero(x) for x in network.G], J=[zero(x) for x in network.J])
    workspaces = [make_workspace(network, length(data.grid)) for _ in 1:Threads.nthreads()]
    loss_and_gradient!(grad, workspaces, network, data, rows, beta, lambda)
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
        array[index] = original + epsilon
        plus = loss_value(network, data, rows, beta, lambda)
        array[index] = original - epsilon
        minus = loss_value(network, data, rows, beta, lambda)
        array[index] = original
        numeric = (plus - minus) / (2 * epsilon)
        analytic = g[index]
        relative = abs(numeric - analytic) / max(abs(numeric) + abs(analytic), 1e-8)
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

function capacity_train!(network, data, rows; beta, lambda, eta, epochs, batch_size, report_every, rng, log_io=nothing)
    grad = (H=[zero(x) for x in network.H], G=[zero(x) for x in network.G], J=[zero(x) for x in network.J])
    workspaces = [make_workspace(network, length(data.grid)) for _ in 1:Threads.nthreads()]
    opt = AdamOptimizer(network, grad)
    history = Any[]
    started = time()
    for epoch in 1:epochs
        for batch in Iterators.partition(shuffle(rng, rows), batch_size)
            loss_and_gradient!(grad, workspaces, network, data, batch, beta, lambda)
            adam_step!(opt, eta)
        end
        if epoch == 1 || epoch % report_every == 0 || epoch == epochs
            profiles, natural_ll = capacity_evaluate(network, data, rows)
            m = capacity_metrics(data, rows, profiles, natural_ll)
            report_progress(log_io, epoch, time() - started, m)
            push!(history, (epoch, m))
        end
    end
    return history
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

function capacity_metrics(data, rows, profiles, natural_ll)
    winners = argmax_rows(profiles)
    natural = data.natural[rows]
    nearest = data.nearest[rows]
    inferred = data.grid[winners]
    errors = inferred .- natural
    ce = [logsumexp(view(profiles, r, :)) - profiles[r, nearest[r]] for r in axes(profiles, 1)]   # -log P(T_true | s)
    at_edge = (winners .== 1) .| (winners .== length(data.grid))
    return (n=length(rows), mean_ce=mean(ce), argmax_hit=mean(winners .== nearest),
            within_1_step=mean(abs.(errors) .<= data.step + 1e-9), mae=mean(abs.(errors)),
            rmse=sqrt(mean(abs2.(errors))), bias=mean(errors), pearson=cor(natural, inferred),
            spearman=spearman_correlation(natural, inferred), within_2C=mean(abs.(errors) .<= 2),
            at_grid_edge=mean(at_edge), loglik_per_site=mean(natural_ll) / data.L)
end

function report_progress(log_io, epoch, seconds, m)
    @printf("epoch %4d  mean_ce %7.3f  argmax at natural %5.1f%%  within 1 step %5.1f%%  MAE %5.2f C  loglik/site %.4f  %6.0f s\n",
            epoch, m.mean_ce, 100 * m.argmax_hit, 100 * m.within_1_step, m.mae, m.loglik_per_site, seconds)
    if log_io !== nothing
        println(log_io, join(Any[epoch, round(seconds; digits=1), m.mean_ce, m.argmax_hit, m.within_1_step, m.mae,
                                 m.rmse, m.pearson, m.at_grid_edge, m.loglik_per_site], ','))
        flush(log_io)
    end
end

with_commas(n::Integer) = replace(string(n), r"(?<=\d)(?=(\d{3})+$)" => ",")

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
    @printf("  mean_ce = %.3f   argmax at the natural grid point: %.1f%%   within one grid step: %.1f%%\n",
            m.mean_ce, 100 * m.argmax_hit, 100 * m.within_1_step)
    @printf("  MAE = %.2f C   RMSE = %.2f C   bias = %+.2f C   Pearson r = %.3f   argmax at a grid edge: %.1f%%\n",
            m.mae, m.rmse, m.bias, m.pearson, 100 * m.at_grid_edge)
    @printf("  log-likelihood per site at the natural temperature = %.4f\n", m.loglik_per_site)
end

function verdict(after, history)
    still_falling = length(history) >= 2 && history[end][2].mean_ce < 0.98 * history[end - 1][2].mean_ce
    lines = String[]
    if after.within_1_step >= 0.9
        push!(lines, "The retrained parameters put the likelihood maximum within one grid step of the natural temperature " *
                     "for $(round(100 * after.within_1_step; digits=1))% of the subsample: the field-only form CAN place " *
                     "the training temperatures (case a). The mismatches of the original model come from the fit " *
                     "(objective, regularization), not from the form; next step: tune beta (and lambda) on held-out data.")
    else
        push!(lines, "Only $(round(100 * after.within_1_step; digits=1))% of the subsample is within one grid step of " *
                     "its natural temperature, even with the temperature term pushed and no regularization: a case (b) " *
                     "candidate. Before concluding, rerun with another --seed and more --epochs; if it holds, temperature " *
                     "has to enter the couplings, J(y).")
    end
    still_falling && push!(lines, "Note: mean_ce was still falling at the last evaluations; train longer before deciding.")
    return lines
end

# ------------------------------------------------------------------------------------------------

function main(args=ARGS)
    options = parse_args(args)
    model_path = options["model"]
    isfile(model_path) || error("Model not found: $model_path")
    model = deserialize(model_path)
    hasproperty(model, :network) || error("$model_path has no `network`; expected a model saved by the TemperatureDCA scripts")
    train_ids, names, means, scales = model_description(model)

    feature = something(options["feature"], "Topt_ave" in names ? "Topt_ave" : first(names))
    k = findfirst(==(feature), names)
    k === nothing && error("$feature is not a feature of the model (features: $(join(names, ", ")))")

    fasta = input_path(options["fasta"], model, :source_fasta, DEFAULT_FASTA, "fasta")
    metadata_path = input_path(options["metadata"], model, :source_metadata, DEFAULT_METADATA, "metadata")
    isfile(fasta) || error("FASTA not found: $fasta")
    isfile(metadata_path) || error("Metadata not found: $metadata_path")
    model_name = splitext(basename(model_path))[1]
    output_dir = something(options["output_dir"], joinpath(dirname(model_path), model_name * "_capacity_test"))
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
    L == length(model.network.H) || error("The FASTA has $L alignment columns but the model has $(length(model.network.H)) sites")
    Z_train = Z[:, [column_of[id] for id in train_ids]]
    q = length(model.network.H[1])
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

    data = (Z=Z_train, y_fixed=y_fixed, k=k, y_natural=(natural .- means[k]) ./ scales[k], natural=natural,
            grid=grid, standardized_grid=standardized_grid, nearest=[argmin(abs.(grid .- t)) for t in natural],
            step=step, L=L, q=q)

    # Subsample and starting parameters (a copy of the model; its network arrays are trained in place)
    rng = MersenneTwister(options["seed"])
    n_train = options["n_train"] == 0 ? length(train_ids) : min(options["n_train"], length(train_ids))
    rows = sort!(randperm(rng, length(train_ids))[1:n_train])
    trained = deepcopy(model)
    network = trained.network
    if hasproperty(network, :idxperm) && collect(network.idxperm) != collect(1:L)
        @warn "The network stores a site permutation (idxperm); like infer_training_temperatures.jl, this script uses the alignment order"
    end
    if options["init"] == "random"
        for x in Iterators.flatten((network.H, network.G, network.J))
            randn!(rng, x)
            x .*= 0.01
        end
    end
    beta = something(options["beta"], 10.0 * L)
    lambda, eta = options["lambda"], options["eta"]

    @printf("Model: %s (features: %s)\n", model_path, join(names, ", "))
    length(names) > 1 && println("  scanning $feature; $(join(setdiff(names, [feature]), ", ")) held at the natural values")
    @printf("Training set of the model: %s sequences x %d sites (q = %d) from %s\n",
            with_commas(length(train_ids)), L, q, fasta)
    @printf("Subsample: %s sequences (seed %d), starting from %s parameters\n", with_commas(n_train), options["seed"],
            options["init"] == "model" ? "the model's" : "random")
    @printf("Grid: %.2f to %.2f C in steps of %.2f C (%d temperatures), %d threads\n",
            first(grid), last(grid), step, length(grid), Threads.nthreads())
    @printf("beta = %g%s, lambda = %g, eta = %g, %d epochs, batch size %d\n\n", beta,
            options["beta"] === nothing ? " (10 x sites)" : "", lambda, eta, options["epochs"], options["batch_size"])

    check_against_reference(network, data, rows)

    if options["gradcheck"]
        println("Gradient check (analytic vs central finite differences) on $(min(8, n_train)) sequences:")
        gradient_check(network, data, rows[1:min(8, n_train)], beta, lambda; rng=rng)
        return
    end

    before_profiles, before_ll = capacity_evaluate(network, data, rows)
    before = capacity_metrics(data, rows, before_profiles, before_ll)
    log_io = open(joinpath(output_dir, "training_log.csv"), "w")
    println(log_io, "epoch,seconds,mean_ce,argmax_hit,within_1_step,mae,rmse,pearson,at_grid_edge,loglik_per_site")
    report_progress(log_io, 0, 0.0, before)
    history = capacity_train!(network, data, rows; beta=beta, lambda=lambda, eta=eta, epochs=options["epochs"],
                              batch_size=options["batch_size"], report_every=options["report_every"], rng=rng,
                              log_io=log_io)
    close(log_io)
    after_profiles, after_ll = capacity_evaluate(network, data, rows)
    after = capacity_metrics(data, rows, after_profiles, after_ll)

    println()
    print_metrics("Before (starting parameters)", before)
    print_metrics("After retraining with the temperature term", after)
    lines = verdict(after, [(0, before); history])
    println()
    foreach(println, lines)

    # Outputs
    sub_ids = train_ids[rows]
    taxa = [String(strip(get(metadata[id], "taxonomy_id", ""))) for id in sub_ids]
    write_sequence_table(joinpath(output_dir, "training_likelihoods_before.csv"), sub_ids, taxa, natural[rows],
                         grid[argmax_rows(before_profiles)], grid, before_profiles)
    write_sequence_table(joinpath(output_dir, "training_likelihoods_after.csv"), sub_ids, taxa, natural[rows],
                         grid[argmax_rows(after_profiles)], grid, after_profiles)

    settings = (source_model=model_path, feature=feature, n_train=n_train, seed=options["seed"], init=options["init"],
                beta=beta, lambda=lambda, eta=eta, epochs=options["epochs"], batch_size=options["batch_size"],
                grid_min=first(grid), grid_max=last(grid), grid_step=step)
    if trained isa NamedTuple
        id_field = hasproperty(trained, :train_sequence_ids) ? :train_sequence_ids : :sequence_ids
        trained = merge(trained, NamedTuple{(id_field, :capacity_test)}((sub_ids, settings)))
    else
        @warn "The model isn't a NamedTuple, so capacity_model.jls keeps the original training IDs"
    end
    serialize(joinpath(output_dir, "capacity_model.jls"), trained)

    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "model=$model_path\nfasta=$fasta\nmetadata=$metadata_path\nfeature=$feature")
        println(io, "model_features=$(join(names, ','))")
        for key in keys(settings)
            key == :source_model || println(io, "$(key)=$(getproperty(settings, key))")
        end
        @printf(io, "feature_mean=%.10f\nfeature_scale=%.10f\n", means[k], scales[k])
        write_metrics(io, "before_", before)
        write_metrics(io, "after_", after)
        foreach(line -> println(io, "# ", line), lines)
    end
    println("\nResults written to $output_dir")
end

main()
