using UncertaintyQuantification
using DelimitedFiles
using Dates
using Distributed
using Formatting
using Mustache

include("./inputsProcessing.jl")
include("./outputProcessing.jl")

function readxlsxinput(input_file::String)
    parameters_list = xlsx2parameters(input_file)
    variables_list = xlsx2variables(input_file)
    par, numberformats = parameters4UQ(parameters_list)
    total_list = par
    for rv in variables_list
        push!(total_list, rv)
    end
    return variables_list, par, numberformats, total_list
end

function get_default_inputs_and_format(input_file::String)
    return Dict{String,Vector}(
        "UQInputs" => readxlsxinput(input_file)[4],
        "FormatSpec" => [Dict(k => v) for (k, v) in readxlsxinput(input_file)[3]]
    )
end

function update_input_list(inputs_list::Union{Array{<:UQInput},UQInput}, to_be_update::Union{Array{<:UQInput},UQInput})
    to_be_update_names = [i.name for i in to_be_update]
    for input in inputs_list
        if input.name ∉ to_be_update_names
            push!(to_be_update, input)
        end
    end
    return to_be_update
end

function update_input_list(inputs_list::Vector{Dict{Symbol,FormatSpec}}, to_be_update::Vector{Dict{Symbol,FormatSpec}})
    to_be_update_names = [collect(keys(to_be_update[i]))[1] for i in range(1, length(to_be_update))]
    for format in inputs_list
        if collect(keys(format))[1] ∉ to_be_update_names
            push!(to_be_update, format)
        end
    end
    return to_be_update
end

function get_workdir(input_file::Union{Array{<:UQInput},UQInput}, sourcedir::String)
    workdir = []
    sim_days = []
    for element in input_file
        if element.name == :sim_duration
            push!(sim_days, element.value[1])
        end
    end
    for element in input_file
        if element.name == :specstorage && element.value == 0
            val = sim_days[1]
            push!(workdir, joinpath(sourcedir, "SteadyState_GWflow", "$(val)_days"))
        elseif element.name == :specstorage && element.value != 0
            val = sim_days[1]
            push!(workdir, joinpath(sourcedir, "Transient_GWflow", "$(val)_days"))
        end
    end
    workdir = workdir[1]
    return workdir
end

function _build_concentration_extractor2D(output_file::String)
    regexs = Dict(
        "variable_regex" => r"(?<=\")[^,]*?(?=\")",
        "day_regex" => r"\d*\.\d{2,5}",
        "x_regex" => r"(?<=i=).*?(?=[,j=])",
        "z_regex" => r"(?<=j=).*?(?=,)",
    )
    extractors = Extractor(
        base -> begin
            file = joinpath(base, output_file)
            result, var, x, z = concentrationplt2dict(
                file,
                regexs["variable_regex"],
                regexs["day_regex"],
                regexs["x_regex"],
                regexs["z_regex"],
                Symbol("concentration")
            )
            return result
        end,
        Symbol("concentration"),
    )
    return extractors
end

function _build_temperature_extractor2D(output_file::String)
    regexs = Dict(
        "variable_regex" => r"(?<=\")[^,]*?(?=\")",
        "day_regex" => r"\d*\.\d{2,5}",
        "x_regex" => r"(?<=i=).*?(?=[,j=])",
        "z_regex" => r"(?<=j=).*?(?=,)",
    )
    extractors = Extractor(
        base -> begin
            file = joinpath(base, output_file)
            result, var, x, z = tempertatureplt2dict(
                file,
                regexs["variable_regex"],
                regexs["day_regex"],
                regexs["x_regex"],
                regexs["z_regex"],
                Symbol("temperature")
            )
            return result
        end,
        Symbol("temperature"),
    )
    return extractors
end


function _build_head_extractor2D(output_file::String)
    regexs = Dict(
        "variable_regex" => r"(?<=\")[^,]*?(?=\")",
        "day_regex" => r"\d*\.\d{2,5}",
        "x_regex" => r"(?<=i=).*?(?=[,j=])",
        "z_regex" => r"(?<=j=).*?(?=,)",
    )
    extractors = Extractor(
        base -> begin
            file = joinpath(base, output_file)
            result, var, x, z = headplt2dict(
                file,
                regexs["variable_regex"],
                regexs["day_regex"],
                regexs["x_regex"],
                regexs["z_regex"],
                Symbol("head")
            )
            return result
        end,
        Symbol("head"),
    )
    return extractors
end

# function build_specific_extractor(outputfile::String, x_range::Vector{Int64}, z_range::Vector{Int64}, qtyofinterest::String)
#     regexs = Dict(
#         "variable_regex" => r"(?<=\")[^,]*?(?=\")",
#         "day_regex" => r"\d*\.\d{2,5}",
#         "x_regex" => r"(?<=i=).*?(?=[,j=])",
#         "z_regex" => r"(?<=j=).*?(?=,)",
#     )
#     extractors = Extractor(
#         base -> begin
#             file = joinpath(base, outputfile)
#             result, var, x, z = concentrationplt2dict(
#                 file,
#                 regexs["variable_regex"],
#                 regexs["day_regex"],
#                 regexs["x_regex"],
#                 regexs["z_regex"],
#             )
#             return [result]
#         end,
#         Symbol("$qtyofinterest"),
#     )
#     return [extractors]
# end


## TODO add 3D concentration and 2/3D Temperature


function build_performances(output_parameters::Dict)
    thresholds = Vector()
    for (key, val) in output_parameters
        if occursin("threshold", key)
            push!(thresholds, (Float64(val[1]), Symbol(val[2])))
        end
    end
    sort!(thresholds)
    performances = Dict{Symbol,Function}()
    for i in range(1, length(thresholds))
        if i != length(thresholds)
            performances[thresholds[i][2]] = df -> max.(
                thresholds[i][1] .- df[!, :concentration],
                df[!, :concentration] .- thresholds[i+1][1])
        else
            performances[thresholds[i][2]] = df ->
                thresholds[i][1] .- df[!, :concentration]
        end
    end
    performances[Symbol("safe")] = df -> df[!, :concentration] .- thresholds[1][1]
    return performances
end

function _get_th_model(sourcedir::String, format_dict::Dict{Symbol,FormatSpec}, uqinputs::Vector{UQInput}, extractor::Vector{Extractor}, cleanup::Bool)
    sourcefile = "smoker.data"
    extras = String[]
    workdir = get_workdir(uqinputs, sourcedir)
    solvername = "smokerV3TC"
    solver = Solver(joinpath(sourcedir, solvername), "", sourcefile)
    return ExternalModel(sourcedir, [sourcefile], extras, format_dict, workdir, extractor, solver, cleanup)
end

function _get_discrete_inputs_mapping_dict(
    parent_vector::Vector{T} where {T<:AbstractNode},
    default_inputs::String,
    sourcedir::String,
    source_file::String,
    extras::Vector{String},
    solvername::String,
    cleanup::Bool
)
    inputs_mapping_dict = Dict{Any,Vector}()
    updated_inputs = Dict{Any,Vector{<:UQInput}}()
    nodes_states, nodes_combinations = get_states_combination(parent_vector)
    output_parameters = xlsx2output_parameter(default_inputs)
    for state in nodes_combinations
        updated_inputs[state] = Vector{UQInput}()
        new_inputs = Vector{UQInput}()
        new_formats = Vector{Dict{Symbol,FormatSpec}}()
        for i in range(1, length(state))
            append!(new_inputs, input for input in parent_vector[collect(keys(nodes_states))[i]].model_input[state[i]]["UQInputs"])
            append!(new_formats, input for input in parent_vector[collect(keys(nodes_states))[i]].model_input[state[i]]["FormatSpec"])
        end
        updated_inputs[state] = update_input_list(get_default_inputs_and_format(default_inputs)["UQInputs"], new_inputs)
        updated_formats_dict = Dict{Symbol,FormatSpec}()
        updated_formats_i = update_input_list(get_default_inputs_and_format(default_inputs)["FormatSpec"], new_formats)
        for el in updated_formats_i
            for (k, v) in el
                updated_formats_dict[k] = v
            end
        end
        inputs_mapping_dict[state] = [
            sourcedir,
            [source_file],
            extras,
            updated_formats_dict,
            get_workdir(update_input_list(get_default_inputs_and_format(default_inputs)["UQInputs"],
                    new_inputs), joinpath(pwd(), "model_TH")),
            _build_concentration_extractor(
                output_parameters["output_filename"],
                [output_parameters["x_min"],
                    output_parameters["x_max"]],
                [output_parameters["z_min"], output_parameters["z_max"]],
                output_parameters["quantity_of_interest"]
            ), Solver(joinpath(sourcedir, solvername), "", source_file),
            cleanup
        ]
    end
    return inputs_mapping_dict, updated_inputs
end

function _get_externalmodels_vector(inputs_mapping_dict::Dict{Any,Vector})
    extmodels_vector = Dict{Any,ExternalModel}()
    for state in collect(keys(inputs_mapping_dict))
        extmodels_vector[state] = ExternalModel(inputs_mapping_dict[state]...)
    end
    return extmodels_vector
end

function evaluate_gen!(m::ExternalModel, df::DataFrame)
    datetime = Dates.format(now(), "YYYY-mm-dd-HH-MM-SS")
    n = size(df, 1)
    digits = ndigits(n)
    results = pmap(1:n) do i
        path = joinpath(m.workdir, datetime, "sample-$(lpad(i, digits, "0"))")
        mkpath(path)
        row = formatinputs(df[i, :], m.formats)
        for file in m.sources
            tokens = Mustache.load(joinpath(m.sourcedir, file))
            open(joinpath(path, file), "w") do io
                render(io, tokens, row)
            end
        end
        for file in m.extras
            cp(joinpath(m.sourcedir, file), joinpath(path, file))
        end
        run(m.solver, path)
        result = map(e -> e.f(path), m.extractors)
        if m.cleanup
            rm(path; recursive=true)
        end
        return result
    end
    results = hcat(results...)
    for (i, name) in enumerate(names(m.extractors))
        df[!, name] = results[i, :]
    end
end

function formatinputs(row::DataFrameRow, formats::Dict{Symbol,FormatSpec})
    names = propertynames(row)
    values = []
    for symbol in names
        if haskey(formats, symbol)
            push!(values, fmt(formats[symbol], row[symbol]))
        elseif haskey(formats, :*)
            push!(values, fmt(formats[:*], row[symbol]))
        else
            push!(values, row[symbol])
        end
    end
    return (; zip(names, values)...)
end