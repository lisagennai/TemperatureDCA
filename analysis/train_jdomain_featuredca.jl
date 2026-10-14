#!/usr/bin/env julia

using FeatureDCA
using Printf
using Random
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_METADATA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.metadata.maxgap50.csv")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_topt_ave.jls")
const TEMPERATURE_COLUMNS = ["Tmin", "Topt_ave", "Topt_low", "Topt_high", "Tmax", "Tmax_Tmin"]

function usage(io=stdout)
    println(io, """
Train temperature-conditioned FeatureDCA on the filtered J-domain MSA.

Usage:
  julia --project=. analysis/train_jdomain_featuredca.jl [options]

Options:
  --fasta PATH             Aligned FASTA (default: final J-domain maxgap50 MSA)
  --metadata PATH          Matching metadata CSV
  --features NAMES         Comma-separated temperature columns (default: Topt_ave)
                           Use 'all' for all six temperature measures.
  --output PATH            Serialized model path
  --lambda-h VALUE         Field regularization (default: 0.01)
  --lambda-g VALUE         Feature regularization (default: 0.01)
  --lambda-j VALUE         Coupling regularization (default: 0.01)
  --epsconv VALUE          Optimizer tolerance (default: 1e-5)
  --maxit N                Maximum evaluations per site (default: 1000)
  --max-sequences N        Reproducibly subsample N complete records (0 = all)
  --seed N                 Subsampling seed (default: 1)
  --dry-run                Validate and assemble Z, W, and Y without training
  --quiet                  Disable per-site optimizer output
  -h, --help               Show this help
""")
end

function parse_args(args)
    options = Dict{String,Any}(
        "fasta" => DEFAULT_FASTA,
        "metadata" => DEFAULT_METADATA,
        "features" => ["Topt_ave"],
        "output" => DEFAULT_OUTPUT,
        "lambda_h" => 0.01,
        "lambda_g" => 0.01,
        "lambda_j" => 0.01,
        "epsconv" => 1e-5,
        "maxit" => 1000,
        "max_sequences" => 0,
        "seed" => 1,
        "dry_run" => false,
        "verbose" => true,
    )
    value_options = Set(["--fasta", "--metadata", "--features", "--output", "--lambda-h",
                         "--lambda-g", "--lambda-j", "--epsconv", "--maxit", "--max-sequences", "--seed"])
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            usage()
            exit(0)
        elseif arg == "--dry-run"
            options["dry_run"] = true
        elseif arg == "--quiet"
            options["verbose"] = false
        elseif arg in value_options
            i == length(args) && error("Missing value after $arg")
            value = args[i + 1]
            key = replace(arg[3:end], "-" => "_")
            if key in ("lambda_h", "lambda_g", "lambda_j", "epsconv")
                options[key] = parse(Float64, value)
            elseif key in ("maxit", "max_sequences", "seed")
                options[key] = parse(Int, value)
            elseif key == "features"
                options[key] = lowercase(value) == "all" ? copy(TEMPERATURE_COLUMNS) : split(value, ',')
            else
                options[key] = abspath(value)
            end
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    isempty(options["features"]) && error("At least one feature is required")
    unknown = setdiff(options["features"], TEMPERATURE_COLUMNS)
    isempty(unknown) || error("Unknown temperature column(s): $(join(unknown, ", "))")
    options["maxit"] > 0 || error("--maxit must be positive")
    options["max_sequences"] >= 0 || error("--max-sequences cannot be negative")
    return options
end

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
    length(unique(header)) == length(header) || error("Metadata has duplicate column names")
    rows = Dict{String,Dict{String,String}}()
    for values in parsed[2:end]
        length(values) == length(header) || error("Malformed metadata row with $(length(values)) fields")
        row = Dict(header[i] => values[i] for i in eachindex(header))
        id = row["sequence_id"]
        haskey(rows, id) && error("Duplicate metadata sequence_id: $id")
        rows[id] = row
    end
    return rows
end

annotation_text(annotation::AbstractString) = annotation
annotation_text(annotation::Tuple) = only(annotation)
sequence_id(annotation) = first(split(strip(annotation_text(annotation))))

function weighted_standardize(Y, weights)
    normalized_weights = weights ./ sum(weights)
    means = vec(Y * normalized_weights)
    centered = Y .- means
    scales = sqrt.(vec((centered .^ 2) * normalized_weights))
    any(scales .<= eps(Float64)) && error("A selected feature has zero weighted variance")
    return centered ./ scales, means, scales
end

function csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_cohort(path, ids, metadata, features, Y_raw, Y)
    open(path, "w") do io
        println(io, join(vcat(["sequence_id", "accession", "taxonomy_id", "organism"],
                              features, ["standardized_" * f for f in features]), ','))
        for (i, id) in enumerate(ids)
            row = metadata[id]
            values = Any[id, get(row, "accession", ""), get(row, "taxonomy_id", ""), get(row, "organism", "")]
            append!(values, Y_raw[:, i])
            append!(values, Y[:, i])
            println(io, join(csv_escape.(values), ','))
        end
    end
end

function main(args=ARGS)
    options = parse_args(args)
    fasta = options["fasta"]
    metadata_path = options["metadata"]
    isfile(fasta) || error("FASTA not found: $fasta")
    isfile(metadata_path) || error("Metadata not found: $metadata_path")

    metadata = read_metadata(metadata_path)
    weights, Z, L, M, q, annotations = FeatureDCA.read_annotated_fasta(
        fasta; remove_dups=false, max_gap_fraction=1.0
    )
    ids = sequence_id.(annotations)
    length(unique(ids)) == length(ids) || error("FASTA contains duplicate sequence IDs")
    missing_metadata = filter(id -> !haskey(metadata, id), ids)
    isempty(missing_metadata) || error("$(length(missing_metadata)) FASTA records lack metadata")

    features = options["features"]
    complete = [i for i in eachindex(ids) if all(!isempty(strip(get(metadata[ids[i]], f, ""))) for f in features)]
    isempty(complete) && error("No records have complete values for $(join(features, ", "))")
    dropped_missing = M - length(complete)

    max_sequences = options["max_sequences"]
    if max_sequences > 0 && length(complete) > max_sequences
        rng = MersenneTwister(options["seed"])
        complete = sort(complete[randperm(rng, length(complete))[1:max_sequences]])
    end

    if length(complete) != M
        Z = Z[:, complete]
        ids = ids[complete]
        weights, _ = FeatureDCA.compute_weights(Z, :auto)
    end
    weights = ones(length(ids))        # unweighted training
    Y_raw = [parse(Float64, metadata[id][feature]) for feature in features, id in ids]
    Y, means, scales = weighted_standardize(Y_raw, weights)

    @assert size(Z, 2) == length(weights) == size(Y, 2) == length(ids)
    @printf("J-domain FeatureDCA input\n")
    @printf("  alignment: %d positions x %d sequences, q=%d\n", size(Z, 1), size(Z, 2), q)
    @printf("  features: %s\n", join(features, ", "))
    @printf("  incomplete rows excluded: %d\n", dropped_missing)
    for i in eachindex(features)
        @printf("  %-10s mean=%9.4f scale=%9.4f range=[%.2f, %.2f]\n",
                features[i], means[i], scales[i], minimum(Y_raw[i, :]), maximum(Y_raw[i, :]))
    end

    output = options["output"]
    cohort_path = replace(output, r"\.[^.]*$" => "") * ".cohort.csv"
    mkpath(dirname(output))
    write_cohort(cohort_path, ids, metadata, features, Y_raw, Y)
    @printf("  cohort: %s\n", cohort_path)

    if options["dry_run"]
        println("Dry run complete; optimizer was not started.")
        return
    end

    started_at = time()
    network, arvar, objective = trainer(
        Z, Float64.(weights);
        Y=Matrix{Float64}(Y),
        lambdaH=options["lambda_h"],
        lambdaG=options["lambda_g"],
        lambdaJ=options["lambda_j"],
        epsconv=options["epsconv"],
        maxit=options["maxit"],
        verbose=options["verbose"],
    )
    elapsed_seconds = time() - started_at
    artifact = (
        network=network,
        objective=objective,
        feature_names=features,
        feature_means=means,
        feature_scales=scales,
        sequence_ids=ids,
        weights=arvar.weights,
        alignment_length=arvar.L,
        alphabet_size=arvar.q,
        source_fasta=abspath(fasta),
        source_metadata=abspath(metadata_path),
        lambdaH=options["lambda_h"],
        lambdaG=options["lambda_g"],
        lambdaJ=options["lambda_j"],
        epsconv=options["epsconv"],
        maxit=options["maxit"],
        seed=options["seed"],
        elapsed_seconds=elapsed_seconds,
    )
    serialize(output, artifact)
    @printf("Model written to %s (%.1f seconds)\n", output, elapsed_seconds)
end

main()
