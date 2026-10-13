#!/usr/bin/env julia

using FeatureDCA
using MultivariateStats
using Printf
using Serialization
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_FASTA = joinpath(ROOT, "data", "processed_new", "jdomain", "jdomain.aligned.maxgap50.fasta")
const DEFAULT_OUTPUT = joinpath(ROOT, "results", "featuredca", "jdomain_pca2.jls")

function parse_args(args)
    options = Dict{String,Any}(
        "fasta" => DEFAULT_FASTA,
        "output" => DEFAULT_OUTPUT,
        "lambda_h" => 0.01,
        "lambda_g" => 0.01,
        "lambda_j" => 0.01,
        "epsconv" => 1e-5,
        "maxit" => 1000,
        "verbose" => true,
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage: julia --project=. analysis/train_jdomain_pca_featuredca.jl [--fasta PATH] [--output PATH] [--maxit N] [--quiet]")
            exit(0)
        elseif arg == "--quiet"
            options["verbose"] = false
        elseif arg in ("--fasta", "--output", "--lambda-h", "--lambda-g", "--lambda-j", "--epsconv", "--maxit")
            i == length(args) && error("Missing value after $arg")
            key = replace(arg[3:end], "-" => "_")
            value = args[i + 1]
            options[key] = key == "maxit" ? parse(Int, value) : key in ("lambda_h", "lambda_g", "lambda_j", "epsconv") ? parse(Float64, value) : abspath(value)
            i += 1
        else
            error("Unknown option: $arg")
        end
        i += 1
    end
    return options
end

function main(args=ARGS)
    options = parse_args(args)
    fasta = options["fasta"]
    isfile(fasta) || error("FASTA not found: $fasta")

    weights, Z, L, M, q, annotations = FeatureDCA.read_annotated_fasta(
        fasta; remove_dups=false, max_gap_fraction=1.0
    )
    ids = [first(split(strip(only(annotation)))) for annotation in annotations]
    length(unique(ids)) == M || error("FASTA sequence IDs are not unique")

    println("Fitting two-component PCA to the natural one-hot MSA...")
    encoded = FeatureDCA.one_hot(Z; q=q)
    pca_map = fit(PCA, encoded, maxoutdim=2)
    Y = Matrix{Float64}(predict(pca_map, encoded))
    total_variance = sum(var(encoded; dims=2, corrected=true))
    explained = principalvars(pca_map) ./ total_variance
    @printf("J-domain PCA-conditioned FeatureDCA input\n")
    @printf("  alignment: %d positions x %d sequences, q=%d\n", L, M, q)
    @printf("  PC1 explained variance: %.4f%%\n", 100 * explained[1])
    @printf("  PC2 explained variance: %.4f%%\n", 100 * explained[2])

    started_at = time()
    network, arvar, objective = trainer(
        Z, Float64.(weights);
        Y=Y,
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
        pca_map=pca_map,
        Y=Y,
        explained_variance=explained,
        feature_names=["PC1", "PC2"],
        sequence_ids=ids,
        weights=arvar.weights,
        alignment_length=arvar.L,
        alphabet_size=arvar.q,
        source_fasta=abspath(fasta),
        lambdaH=options["lambda_h"],
        lambdaG=options["lambda_g"],
        lambdaJ=options["lambda_j"],
        epsconv=options["epsconv"],
        maxit=options["maxit"],
        elapsed_seconds=elapsed_seconds,
    )
    mkpath(dirname(options["output"]))
    serialize(options["output"], artifact)
    @printf("Model written to %s (%.1f seconds)\n", options["output"], elapsed_seconds)
end

main()
