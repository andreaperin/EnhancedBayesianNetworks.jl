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

function _get_th_model(sourcedir::String, format_dict::Dict{Symbol,FormatSpec}, uqinputs::Vector{Q}, extractor::Vector{Extractor}, cleanup::Bool) where {Q<:UQInput}
    sourcefile = "smoker.data"
    extras = String[]
    workdir = get_workdir(uqinputs, sourcedir)
    solvername = "smokerV3TC"
    solver = Solver(joinpath(sourcedir, solvername), "", sourcefile)
    return ExternalModel(sourcedir, [sourcefile], extras, format_dict, workdir, extractor, solver, cleanup)
end