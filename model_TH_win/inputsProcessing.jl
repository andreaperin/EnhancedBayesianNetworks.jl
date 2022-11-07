using XLSX
using Formatting
using DataFrames

function parameters4UQ(param_list::Vector)
    par = UQInput[]
    frmt = Dict{Symbol,FormatSpec}()
    for i in range(1, length(param_list))
        push!(par, Parameter(param_list[i][1]["value"], param_list[i][2]))
        frmt[param_list[i][2]] = FormatSpec(param_list[i][1]["format"])
    end
    return par, frmt
end

function xlsx2parameters(input_file::String)
    df = DataFrame(XLSX.readtable(input_file, "parameter"))
    parameters_list = []
    for i in range(1, size(df)[1])
        push!(
            parameters_list,
            (
                Dict("value" => df[i, "value"], "format" => df[i, "format"]),
                Symbol(df[i, "mustache"]),
            ),
        )
    end
    return parameters_list
end

function xlsx2output_parameter(input_file::String)
    df = DataFrame(XLSX.readtable(input_file, "output"))
    output_parameter = Dict()
    for i in range(1, size(df)[1])
        if typeof(df[i, "state_name"]) == Missing
            output_parameter[df[i, "name"]] = df[i, "value"]
        else
            output_parameter[df[i, "name"]] = (df[i, "value"], df[i, "state_name"])
        end
    end
    return output_parameter
end

function xlsx2variables(input_file::String)
    df = DataFrame(XLSX.readtable(input_file, "randomvariable"))
    variables_list = []
    for i in range(1, size(df)[1])
        if df[i, "distribution"] == "Uniform"
            push!(
                variables_list,
                RandomVariable(Uniform(df[i, "lower"], df[i, "upper"]), Symbol(df[i, "mustache"])),
            )
        elseif df[i, "distribution"] == "Truncated Normal"
            if typeof(df[i, "lower"]) == Missing
                push!(
                    variables_list,
                    RandomVariable(
                        truncated(
                            Normal(df[i, "mean"], df[i, "var"]); upper=df[i, "upper"]
                        ),
                        Symbol(df[i, "mustache"]),
                    ),
                )
            elseif typeof(df[i, "upper"]) == Missing
                push!(
                    variables_list,
                    RandomVariable(
                        truncated(
                            Normal(df[i, "mean"], df[i, "var"]); lower=df[i, "lower"]
                        ),
                        Symbol(df[i, "mustache"]),
                    ),
                )
            end
        elseif df[i, "distribution"] == "LogNormal"
            push!(
                variables_list,
                RandomVariable(
                    LogNormal(df[i, "mean"], df[i, "var"]), Symbol(df[i, "mustache"])
                ),
            )
        else
            dist = df[i."distribution"]
            print("this distribution '$dist' needs to be added in xlsx2variables")
        end
    end
    return variables_list
end