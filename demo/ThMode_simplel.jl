using EnhancedBayesianNetworks

Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")

timescenario = NamedCategorical([:first, :second, :third], [0.34, 0.33, 0.33])
CPD_timescenario = RootCPD(:time_scenario, timescenario)
timescenario_node = StdNode(CPD_timescenario)

earthquake = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPD_earthquake = RootCPD(:earthquake, [earthquake])
earthquake_node = RootNode(CPD_earthquake)

extremerain = NamedCategorical([:low, :high], [0.5, 0.5])
CPD_extremerain = RootCPD(:extremerain, [extremerain])
extremerain_node = RootNode(CPD_extremerain)

disp_longv_distribution = truncated(Normal(1, 1), lower=0)
CPD_disp_longv = RootCPD(:disp_longv, [disp_longv_distribution])
disp_longv_node = RootNode(CPD_disp_longv)

Kz_parents = [extremerain_node]
Kz_distribution1 = truncated(Normal(1, 1), lower=0)
Kz_distribution2 = truncated(Normal(4, 2), lower=0)
CPD_Kz = StdCPD(:Kz, name.(Kz_parents), [2], [Kz_distribution1, Kz_distribution2])
Kz_node = StdNode(CPD_Kz, Kz_parents)

## Model Node



output_target = :output
output_parents = [earthquake_node, disp_longv_node, Kz_node]
output_parental_ncat = [2, 2]


function max_t(df::DataFrame)
    max_temp = []
    for i in range(1, length(df.temperature))
        max_tempi = []
        for (key, val) in df.temperature[i]
            push!(max_tempi, maximum(df.temperature[i][key].temperature))
        end
        push!(max_temp, maximum(max_tempi))
    end
    return max_temp
end
function max_c(df::DataFrame)
    max_conc = []
    for i in range(1, length(df.concentration))
        max_conci = []
        for (key, val) in df.concentration[i]
            push!(max_conci, maximum(df.concentration[i][key].concentration))
        end
        push!(max_conc, maximum(max_conci))
    end
    return max_conc
end

function max_h(df::DataFrame)
    max_head = []
    for i in range(1, length(df.head))
        max_headi = []
        for (key, val) in df.head[i]
            push!(max_headi, maximum(df.head[i][key].head))
        end
        push!(max_head, maximum(max_headi))
    end
    return max_head
end



## Scenario 1
correlated_nodes1 = name.([disp_longv_node, Kz_node])
copula1 = GaussianCopula([1 0.8; 0.8 1])
name1 = :jd
correlation1 = [CPDCorrelationCopula(correlated_nodes1, copula1, :jd)]

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
models1 = [model1, Model(max_t, :T_max), Model(max_c, :c_max), Model(max_h, :head_max)]

performance1 = df -> 10 .- df.T_max

simulation1 = MonteCarlo(2)
srp1 = CPDSystemReliabilityProblem(models1, parameters, performance1, simulation1)
scenario1 = CPDProbabilityDictionaryFunctional((Dict(name(earthquake_node) => 1, name(extremerain_node) => 1), srp1))

## Scenario 2
correlated_nodes2 = name.([disp_longv_node, Kz_node])
copula2 = GaussianCopula([1 0.8; 0.8 1])
name2 = :jd
correlation2 = [CPDCorrelationCopula(correlated_nodes1, copula1, :jd)]

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
models2 = [model2, Model(max_t, :T_max), Model(max_c, :c_max), Model(max_h, :head_max)]

performance2 = df -> 2 .- df.c_max

simulation2 = MonteCarlo(2)
srp2 = CPDSystemReliabilityProblem(models2, parameters, performance2, correlation2, simulation2)
scenario2 = CPDProbabilityDictionaryFunctional((Dict(name(earthquake_node) => 1, name(extremerain_node) => 2), srp2))

## Scenario 3
correlated_nodes3 = name.([disp_longv_node, Kz_node])
copula3 = GaussianCopula([1 0.8; 0.8 1])
name3 = :jd
correlation3 = [CPDCorrelationCopula(correlated_nodes3, copula3, :jd)]

sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
default_file3 = joinpath(pwd(), sourcedir, "inputs", "default_th_values3.xlsx")
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file3)[3]
parameters = readxlsxinput(default_file3)[2]
output_file_conc = "smoker_cxz.plt"
output_file_temp = "smoker_txz.plt"
output_file_head = "smoker_hxz.plt"
extractor3 = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
model3 = _get_th_model(sourcedir, format_dict, parameters, extractor3, true)
models3 = [model3, Model(max_t, :T_max), Model(max_c, :c_max), Model(max_h, :head_max)]

performance3 = df -> 0.9 .- df.c_max

simulation3 = MonteCarlo(2)
srp3 = CPDSystemReliabilityProblem(models3, parameters, performance3, correlation3, simulation3)
scenario3 = CPDProbabilityDictionaryFunctional((Dict(name(earthquake_node) => 2, name(extremerain_node) => 1), srp3))

## Scenario 4
correlated_nodes4 = name.([disp_longv_node, Kz_node])
copula4 = GaussianCopula([1 0.8; 0.8 1])
name4 = :jd
correlation4 = [CPDCorrelationCopula(correlated_nodes4, copula4, :jd)]

sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
default_file4 = joinpath(pwd(), sourcedir, "inputs", "default_th_values4.xlsx")
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file4)[3]
parameters = readxlsxinput(default_file4)[2]
output_file_conc = "smoker_cxz.plt"
output_file_temp = "smoker_txz.plt"
output_file_head = "smoker_hxz.plt"
extractor4 = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
model4 = _get_th_model(sourcedir, format_dict, parameters, extractor4, true)
models4 = [model4, Model(max_t, :T_max), Model(max_c, :c_max), Model(max_h, :head_max)]

performance4 = df -> 0.9 .- df.c_max

simulation4 = MonteCarlo(2)
srp4 = CPDSystemReliabilityProblem(models4, parameters, performance4, correlation4, simulation4)
scenario4 = CPDProbabilityDictionaryFunctional((Dict(name(earthquake_node) => 2, name(extremerain_node) => 2), srp4))

prob_dict_output = [scenario1, scenario2, scenario3, scenario4]
CPD_output = FunctionalCPD(output_target, name.(output_parents), output_parental_ncat, prob_dict_output)
output_node = FunctionalNode(CPD_output, output_parents, "discrete")


nodes = [earthquake_node, extremerain_node, disp_longv_node, Kz_node, output_node]
ebn = EnhancedBayesNet(nodes)
##TODO add check when ExternalModel with Moustaches (continuous nodename should be among moustaches)
show(ebn)

""" Solving EnhancedBayesNet (No undefined continuous parents case) """
results = Dict()
for prob_dict in output_node.node_prob_dict
    UQInputs = vcat(build_UQInputs_singlecase(output_node, prob_dict), prob_dict.distribution.parameters)
    pf, cov, samples = probability_of_failure(prob_dict.distribution.model, prob_dict.distribution.performance, UQInputs, prob_dict.distribution.simulation)
    samples[!, :performance] = prob_dict.distribution.performance(samples)
    results[prob_dict] = Dict("pf" => pf, "cov" => cov, "samples" => samples)
end
