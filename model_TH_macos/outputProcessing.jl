
function plt2df(data, var_regex, day_regex, x_regex, z_regex)
    result = Dict()
    day = []
    x = []
    z = []
    vars = []
    t = 0
    for i in 1:size(data)[1]
        if occursin("variables", string(data[i]))
            push!(vars, [match.match for match in eachmatch(var_regex, data[1])])
            continue
        elseif occursin("text", string(data[i]))
            day = match(day_regex, data[i]).match
            result[day] = []
            continue
        elseif occursin("zone", string(data[i]))
            push!(x, parse(Float64, match(x_regex, data[i]).match))
            push!(z, parse(Float64, match(z_regex, data[i]).match))
            continue
        else
            if typeof(data[i]) == Float64
                push!(result[day], data[i])
            elseif typeof(data[i]) == SubString{String}
                for val in split(data[i])
                    push!(result[day], parse(Float64, val))
                end
            else
                print("ERROR")
            end
        end
    end
    ### Checking matrix dimension
    for (key, val) in result
        t = t + 1
        if length(result[key]) == x[t] * z[t] || length(result[key]) == 3 * x[t] * z[t]
            println("")
        else
            print("error parsing values")
        end
    end
    ### Dataframe Construction
    df = DataFrame()
    t = 0
    for (key, val) in result
        t = t + 1
        if length(result[key]) == 3 * x[t] * z[t]
            df[!, Symbol("t$key")] = result[key][3:3:end]
            df[!, :x] = result[key][1:3:end]
            df[!, :z] = result[key][2:3:end]
        elseif length(result[key]) == x[t] * z[t]
            df[!, Symbol("t$key")] = result[key]
        end
    end
    return df, vars, x[1], z[1]
end

function df2criticalvalue(df, x_val, z_val) ## For now just maximum value
    times = []
    for time in names(df)
        if occursin("t", time)
            push!(times, time)
        end
    end
    filtered = filter(
        [:x, :z] => (x, z) -> x_val[2] <= x <= x_val[2] && z_val[1] <= z <= z_val[2], df
    )[
        !, times
    ]
    maxs = Dict()
    for t in times
        timet = parse(Float64, chop(t; head=1))
        maxs[timet] = []
        push!(maxs[timet], maximum(filtered[!, t]))
    end
    max_conc = 0
    for (key, val) in maxs
        if val[1] > max_conc
            max_conc = val[1]
        end
    end
    return maxs, max_conc
end
