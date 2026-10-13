#!/usr/bin/env julia

# One-hot MSA PCA colored by available temperature fields, implemented with Julia's
# LinearAlgebra, Statistics, and MultivariateStats. Observations are stored as
# columns, following MultivariateStats convention.

using LinearAlgebra
using MultivariateStats
using Printf
ENV["MPLBACKEND"] = "Agg"
ENV["MPLCONFIGDIR"] = normpath(joinpath(@__DIR__, "..", ".matplotlib-cache"))
mkpath(ENV["MPLCONFIGDIR"])
using PyPlot
using Random
using Statistics

const FAMILIES = ["hsp70", "jdomain"]
const TEMPERATURE_COLUMNS = ["Tmin", "Topt_ave", "Topt_low", "Topt_high", "Tmax", "Tmax_Tmin"]

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
                if next_i <= lastindex(text) && text[next_i] == '\n'
                    i = next_i
                end
            end
            push!(fields, String(take!(buffer)))
            if !(length(fields) == 1 && isempty(fields[1]))
                push!(rows, fields)
            end
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

    return rows
end

function csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function read_metadata(path::AbstractString)
    parsed_rows = parse_csv(read(path, String))
    header = parsed_rows[1]
    rows = Dict{String, Dict{String, String}}()

    for values in parsed_rows[2:end]
        row = Dict(header[i] => get(values, i, "") for i in eachindex(header))
        rows[row["sequence_id"]] = row
    end

    return rows
end

function read_fasta(path::AbstractString)
    records = Vector{Tuple{String, String}}()
    current_id = nothing
    chunks = String[]

    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue

        if startswith(line, ">")
            if current_id !== nothing
                push!(records, (current_id, uppercase(join(chunks))))
            end
            current_id = split(line[2:end])[1]
            empty!(chunks)
        else
            push!(chunks, line)
        end
    end

    if current_id !== nothing
        push!(records, (current_id, uppercase(join(chunks))))
    end

    return records
end

function one_hot_encode(sequences::Vector{String})
    lengths = unique(length.(sequences))
    length(lengths) == 1 || error("Expected equal-length aligned sequences, got $(sort(lengths))")

    alphabet = sort(collect(Set(vcat(collect.(sequences)...))))
    state_index = Dict(char => i for (i, char) in enumerate(alphabet))
    alignment_length = lengths[1]
    n_states = length(alphabet)
    n_sequences = length(sequences)
    encoded = zeros(Float32, alignment_length * n_states, n_sequences)

    for (column, sequence) in enumerate(sequences)
        for (position, char) in enumerate(sequence)
            row = (position - 1) * n_states + state_index[char]
            encoded[row, column] = 1.0
        end
    end

    return encoded, join(alphabet)
end

function randomized_pca_model(encoded; n_components=2, oversamples=8, n_iter=2, seed=0)
    n_features, n_sequences = size(encoded)
    rank = min(n_components + oversamples, n_sequences - 1, n_features)
    mean_vector = vec(mean(encoded; dims=2))
    centered = encoded .- mean_vector

    rng = MersenneTwister(seed)
    omega = randn(rng, Float32, n_sequences, rank)
    sketch = centered * omega

    for _ in 1:n_iter
        sketch = Matrix(qr(sketch).Q)
        sketch = centered * (centered' * sketch)
    end

    q_matrix = Matrix(qr(sketch).Q)
    small = q_matrix' * centered
    decomposition = svd(small; full=false)
    projection = q_matrix * decomposition.U[:, 1:n_components]

    principal_variances = decomposition.S[1:n_components] .^ 2 ./ (n_sequences - 1)
    total_variance = sum(abs2, centered) / (n_sequences - 1)

    return PCA(mean_vector, projection, principal_variances, total_variance)
end

function pearson(x::AbstractVector, y::AbstractVector)
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)
    denom = norm(x_centered) * norm(y_centered)
    denom == 0 && return NaN
    return dot(x_centered, y_centered) / denom
end

function nice_step(raw_step)
    raw_step <= 0 && return 1.0
    exponent = floor(Int, log10(raw_step))
    fraction = raw_step / 10.0^exponent
    nice_fraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10
    return nice_fraction * 10.0^exponent
end

function tick_values(values; n_ticks=7)
    vmin = minimum(values)
    vmax = maximum(values)
    vmax <= vmin && return [vmin]
    step = nice_step((vmax - vmin) / max(n_ticks - 1, 1))
    tick_min = floor(vmin / step) * step
    tick_max = ceil(vmax / step) * step
    tick_min > 0 && (tick_min = 0.0)
    tick_max < 0 && (tick_max = 0.0)
    ticks = collect(tick_min:step:(tick_max + step * 0.5))
    ticks[abs.(ticks) .< step * 1e-8] .= 0.0
    return ticks
end

function write_pdf(path, family, scores, temperatures, explained, temperature_column)
    available = .!isnan.(temperatures)
    plot_scores = scores[available, :]
    plot_temperatures = temperatures[available]
    order = sortperm(plot_temperatures)
    x_ticks = tick_values(scores[:, 1])
    y_ticks = tick_values(scores[:, 2])

    fig, ax = subplots(figsize=(7.2, 5.6), constrained_layout=true)
    scatter = ax.scatter(
        plot_scores[order, 1],
        plot_scores[order, 2],
        c=plot_temperatures[order],
        cmap="viridis",
        s=4,
        alpha=0.78,
        linewidths=0,
        rasterized=true,
    )
    ax.set_title("$family Julia PCA colored by $temperature_column", loc="center", fontsize=13, pad=12)
    ax.set_xlabel(@sprintf("PC1 (%.2f%% variance)", explained[1] * 100))
    ax.set_ylabel(@sprintf("PC2 (%.2f%% variance)", explained[2] * 100))
    ax.set_xlim(first(x_ticks), last(x_ticks))
    ax.set_ylim(first(y_ticks), last(y_ticks))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.xaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.grid(true, color="#d7dce2", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(true)
    for spine_name in ["left", "right", "top", "bottom"]
        spine = ax.spines[spine_name]
        spine.set_linewidth(0.8)
        spine.set_color("#5b6573")
    end

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.025, shrink=0.92)
    colorbar.set_label(temperature_column)
    colorbar.ax.tick_params(labelsize=8, colors="#1f2933")
    colorbar.outline.set_linewidth(0.6)

    fig.savefig(path, format="pdf", dpi=300)
    close(fig)
end

function write_coordinates(path, records, rows, scores)
    header = vcat(["sequence_id", "accession", "organism", "taxonomy_id"], TEMPERATURE_COLUMNS, ["PC1", "PC2"])
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(records)
            sequence_id = records[i][1]
            row = rows[i]
            values = [
                sequence_id,
                get(row, "accession", ""),
                get(row, "organism", ""),
                get(row, "taxonomy_id", ""),
                [get(row, column, "") for column in TEMPERATURE_COLUMNS]...,
                @sprintf("%.8g", scores[i, 1]),
                @sprintf("%.8g", scores[i, 2]),
            ]
            println(io, join(csv_escape.(values), ","))
        end
    end
end

function temperature_array(rows, column)
    values = Float64[]
    for row in rows
        value = get(row, column, "")
        push!(values, isempty(value) ? NaN : parse(Float64, value))
    end
    return values
end

function summarize_temperature(column, temperatures, scores, base_summary)
    available = .!isnan.(temperatures)
    available_temperatures = temperatures[available]
    available_scores = scores[available, :]
    return merge(
        base_summary,
        Dict(
            "temperature_column" => column,
            "temperature_n" => count(available),
            "temperature_min" => minimum(available_temperatures),
            "temperature_median" => median(available_temperatures),
            "temperature_max" => maximum(available_temperatures),
            "pc1_temperature_pearson" => pearson(available_scores[:, 1], available_temperatures),
            "pc2_temperature_pearson" => pearson(available_scores[:, 2], available_temperatures),
        ),
    )
end

function analyze_family(input_dir, output_dir, family)
    fasta_path = joinpath(input_dir, family, "$family.aligned.maxgap50.fasta")
    metadata_path = joinpath(input_dir, family, "$family.metadata.maxgap50.csv")

    metadata = read_metadata(metadata_path)
    fasta_records = read_fasta(fasta_path)
    records = Tuple{String, String}[]
    rows = Dict{String, String}[]

    for (sequence_id, sequence) in fasta_records
        row = get(metadata, sequence_id, nothing)
        if row !== nothing
            push!(records, (sequence_id, sequence))
            push!(rows, row)
        end
    end

    sequences = [sequence for (_, sequence) in records]
    encoded, alphabet = one_hot_encode(sequences)
    model = randomized_pca_model(encoded)
    transformed = transform(model, encoded)
    scores = Matrix(transformed')
    explained = principalvars(model) ./ model.tvar

    mkpath(output_dir)
    write_coordinates(joinpath(output_dir, "$family.pca_temperature.csv"), records, rows, scores)

    base_summary = Dict(
        "family" => family,
        "n_sequences" => length(records),
        "alignment_length" => length(first(sequences)),
        "alphabet" => alphabet,
        "pc1_explained" => explained[1],
        "pc2_explained" => explained[2],
    )

    summaries = Dict{String, Any}[]
    for column in TEMPERATURE_COLUMNS
        temperatures = temperature_array(rows, column)
        if any(.!isnan.(temperatures))
            write_pdf(joinpath(output_dir, "$family.pca_$column.pdf"), family, scores, temperatures, explained, column)
            push!(summaries, summarize_temperature(column, temperatures, scores, base_summary))
        end
    end

    return summaries
end

function write_summary(path, summaries)
    open(path, "w") do io
        println(io, "Julia one-hot MSA PCA temperature-space summary")
        println(io)
        println(io, "Temperature columns: $(join(TEMPERATURE_COLUMNS, ", "))")
        println(io, "Input: data/processed_new/<family>/<family>.aligned.maxgap50.fasta")
        println(io, "PCA: randomized low-rank SVD in Julia, wrapped as MultivariateStats.PCA and projected with MultivariateStats.transform")
        println(io)

        for summary in summaries
            println(io, summary["family"])
            println(io, "  sequences: $(summary["n_sequences"])")
            println(io, "  alignment length: $(summary["alignment_length"])")
            println(io, "  alphabet: $(summary["alphabet"])")
            @printf(io, "  explained variance: PC1 %.2f%%, PC2 %.2f%%\n", summary["pc1_explained"] * 100, summary["pc2_explained"] * 100)
            println(io, "  temperature column: $(summary["temperature_column"]); n=$(summary["temperature_n"])")
            @printf(io, "  %s range: %.1f - %.1f C; median %.1f C\n", summary["temperature_column"], summary["temperature_min"], summary["temperature_max"], summary["temperature_median"])
            @printf(io, "  Pearson corr(%s, PC1): %.3f\n", summary["temperature_column"], summary["pc1_temperature_pearson"])
            @printf(io, "  Pearson corr(%s, PC2): %.3f\n", summary["temperature_column"], summary["pc2_temperature_pearson"])
            println(io)
        end
    end
end

function main()
    input_dir = length(ARGS) >= 1 ? ARGS[1] : "data/processed_new"
    output_dir = length(ARGS) >= 2 ? ARGS[2] : "results/pca_temperature_julia"

    summaries = reduce(vcat, [analyze_family(input_dir, output_dir, family) for family in FAMILIES])
    write_summary(joinpath(output_dir, "summary.txt"), summaries)

    for summary in summaries
        @printf("%s %s: n=%d, PC1=%.2f%%, PC2=%.2f%%\n", summary["family"], summary["temperature_column"], summary["temperature_n"], summary["pc1_explained"] * 100, summary["pc2_explained"] * 100)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
