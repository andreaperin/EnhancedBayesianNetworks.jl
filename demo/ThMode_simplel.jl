include("../bn.jl")
Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")

# timescenario = NamedCategorical([:first, :second, :third], [0.34, 0.33, 0.33])
# CPD_timescenario = RootCPD(:time_scenario, timescenario)
# timescenario_node = StdNode(CPD_timescenario)

earthquake = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPD_earthquake = RootCPD(:earthquake, earthquake)
earthquake_node = StdNode(CPD_earthquake)

# extremerain = NamedCategorical([:low, :medium, :high], [0.5, 0.3, 0.2])
# CPD_extremerain = RootCPD(:extreme_rain, extremerain)
# extremerain_node = StdNode(CPD_extremerain)

disp_longv_distribution = truncated(Normal(1, 1), lower=0)
CPD_disp_longv = RootCPD(:disp_longv, disp_longv_distribution)
disp_longv_node = StdNode(CPD_disp_longv)

## Model Node

output_target = :output
output_parents = [earthquake_node, disp_longv_node]
output_parental_ncat = [2]

## Scenario 1
sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
default_file1 = joinpath(pwd(), sourcedir, "inputs", "default_th_values1.xlsx")
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file1)[3]
parameters = readxlsxinput(default_file1)[2]
output_file_conc = "smoker_cxz.plt"
output_file_temp = "smoker_txz.plt"
output_file_head = "smoker_hxz.plt"
extractor1 = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
model1 = _get_th_model(sourcedir, format_dict, parameters, extractor1, true)

## Performance Function case 1
function performance1(df::DataFrame)
    results = []
    for i in range(1, size(df)[1])
        max_temp = []
        for (key, val) in df[i, :temperature]
            append!(max_temp, maximum(val.temperature))
        end
        max_conc = []
        for (key, val) in df[i, :concentration]
            append!(max_conc, maximum(val.concentration))
        end
        max_head = []
        for (key, val) in df[i, :head]
            append!(max_head, maximum(val.head))
        end
        push!(results, min(1 - maximum(max_temp), 0.1 - maximum(max_conc), 10 - maximum(max_head)))
    end
    return results
end

simulation1 = MonteCarlo(20)
srp1 = CPDSystemReliabilityProblem(model1, parameters, performance1, simulation1)
scenario1 = CPDProbabilityDictionaryFunctional((Dict(name(earthquake_node) => 1), srp1))

## Scenario 2
sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
default_file2 = joinpath(pwd(), sourcedir, "inputs", "default_th_values2.xlsx")
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file2)[3]
parameters = readxlsxinput(default_file2)[2]
output_file_conc = "smoker_cxz.plt"
output_file_temp = "smoker_txz.plt"
output_file_head = "smoker_hxz.plt"
extractor2 = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
model2 = _get_th_model(sourcedir, format_dict, parameters, extractor2, true)

function performance2(df::DataFrame)
    results = []
    for i in range(1, size(df)[1])
        max_temp = []
        for (key, val) in df[i, :temperature]
            append!(max_temp, maximum(val.temperature))
        end
        max_conc = []
        for (key, val) in df[i, :concentration]
            append!(max_conc, maximum(val.concentration))
        end
        max_head = []
        for (key, val) in df[i, :head]
            append!(max_head, maximum(val.head))
        end
        push!(results, min(10 - maximum(max_temp), 100 - maximum(max_conc), 100 - maximum(max_head)))
    end
    return results
end

function performance2(df::DataFrame)
    results = []
    for i in range(1, size(df)[1])
        max_temp = []
        for (key, val) in df[i, :temperature]
            append!(max_temp, maximum(val.temperature))
        end
        max_conc = []
        for (key, val) in df[i, :concentration]
            append!(max_conc, maximum(val.concentration))
        end
        max_head = []
        for (key, val) in df[i, :head]
            append!(max_head, maximum(val.head))
        end
        push!(results, min(1 - maximum(max_temp), 1 - maximum(max_conc), 1 - maximum(max_head)))
    end
    return results
end

simulation2 = MonteCarlo(10)
srp2 = CPDSystemReliabilityProblem(model2, parameters, performance2, simulation2)
scenario2 = CPDProbabilityDictionaryFunctional((Dict(name(earthquake_node) => 2), srp2))


prob_dict_output = [scenario1, scenario2]
CPD_output = FunctionalCPD(output_target, name.(output_parents), output_parental_ncat, prob_dict_output)
output_node = FunctionalNode(CPD_output, output_parents, "discrete")


nodes = [earthquake_node, disp_longv_node, output_node]
ebn = EnhancedBayesNet(nodes)
show(ebn)





""" Solving EnhancedBayesNet (No undefined continuous parents case) """
results = Dict()
for prob_dict in output_node.node_prob_dict
    UQInputs = vcat(build_UQInputs_singlecase(output_node, prob_dict), prob_dict.distribution.parameters)
    pf, cov, samples = probability_of_failure(prob_dict.distribution.model, prob_dict.distribution.performance, UQInputs, prob_dict.distribution.simulation)
    samples[!, :performance] = prob_dict.distribution.performance(samples)
    results[prob_dict] = Dict("pf" => pf, "cov" => cov, "samples" => samples)
end
