using EnhancedBayesianNetworks
using GraphRecipes
using Plots

Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")

## EarthquakeNode
parameters_earthquake = Dict(
    :happen => [Parameter(-8.1e-5, :flow)],
    :nothappen => [Parameter(-0.1e-5, :flow)]
)
node_earthquake = DiscreteRootNode(:EQ, Dict(:happen => 0.5, :nothappen => 0.5), parameters_earthquake)

## ExtremeRainNode
node_extremerain = DiscreteRootNode(:ER, Dict(:low => 0.5, :high => 0.5))

## Dispersivity long_vNode
node_disp = ContinuousRootNode(:DSP, truncated(Normal(1, 1), lower=0))

## K_zNode
Kz_parents = [node_extremerain]
Kz_states = Dict(
    [:low] => truncated(Normal(1, 1), lower=0),
    [:high] => truncated(Normal(4, 2), lower=0)
)
Kz_node = ContinuousStandardNode(:KZ, Kz_parents, Kz_states)


## Model Node
# sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
# default_file = joinpath(pwd(), sourcedir, "inputs", "default_th_values.xlsx")
# sourcedir = joinpath(pwd(), sourcedir)
# format_dict = readxlsxinput(default_file)[3]
# uqinputs = readxlsxinput(default_file)[4]
# output_file_conc = "smoker_cxz.plt"
# output_file_temp = "smoker_txz.plt"
# output_file_head = "smoker_hxz.plt"
sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
default_file = joinpath(pwd(), sourcedir, "inputs", "default_th_values.xlsx")
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file)[3]
uqinputs = readxlsxinput(default_file)[4]
output_parameters = xlsx2output_parameter(default_file)

output_file_conc = "smoker_cxz.plt"
output_file_temp = "smoker_txz.plt"
output_file_head = "smoker_hxz.plt"

extractor = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
default_model = _get_th_model(sourcedir, uqinputs, format_dict, extractor, false)

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
performance = df -> 1.2 .- df.c_max


output_target = :output
output_parents = [node_earthquake, node_disp, Kz_node]

extractor1 = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
model1 = _get_th_model(sourcedir, uqinputs, format_dict, extractor1, true)
models1 = [model1, Model(max_t, :T_max), Model(max_c, :c_max), Model(max_h, :head_max), Model(performance, :output)]

output_models = Dict(
    [:happen] => models1,
    [:nothappen] => models1
)
output_simulations = Dict(
    [:happen] => MonteCarlo(4),
    [:nothappen] => MonteCarlo(4)
)
output_performances = Dict(
    [:happen] => df -> 1.2 .- df.c_max,
    [:nothappen] => df -> 1.2 .- df.c_max
)

node_output = DiscreteFunctionalNode(:OUT, output_parents, output_models, output_performances, output_simulations)


nodes = [node_earthquake, node_extremerain, node_disp, Kz_node, node_output]
ebn = EnhancedBayesianNetwork(nodes)

# gr();
# nodesize = 0.1
# fontsize = 18
# EnhancedBayesianNetworks.plot(ebn, :tree, nodesize, fontsize)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/THmodel_ebn.png")

# rbn = reduce_ebn_markov_envelopes(ebn)
# EnhancedBayesianNetworks.plot(rbn[1], :tree, nodesize, fontsize)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/THmodel_rbn.png")

e_ebn = evaluate_ebn(ebn)


# """ Solving EnhancedBayesNet """
# groups = markov_envelopes(ebn)
# rdag, dag_names = _reduce_ebn_to_rbn(ebn)
# graphplot(
#     rdag,
#     method=:tree,
#     names=dag_names,
#     fontsize=9,
#     nodeshape=:ellipse,
#     markercolor=map(x -> x.type == "discrete" ? "lightgreen" : "orange", filter(x -> x.cpd.target ∈ dag_names, ebn.nodes)),
#     linecolor=:darkgrey,
# )

# empty_srp_table_with_evidence = _build_node_evidence_after_reduction(ebn, rdag, dag_names, output_node)
# a = _functional_node_after_reduction(ebn, empty_srp_table_with_evidence, output_node)

# function update_model_parameter_for_given_evidence(default_parameters::Vector{UQInput}, specific_inputs::Vector{UQInput})
#     mes = map(x -> x.name, specific_inputs)
#     updated_uqinputs = filter(x -> x.name ∉ mes, default_parameters)
#     append!(updated_uqinputs, specific_inputs)
#     return updated_uqinputs
# end

# n = 1
# result = Dict()
# for evaluation in a
#     df = UncertaintyQuantification.sample(update_model_parameter_for_given_evidence(uqinputs, evaluation.srp[2].inputs), n)
#     ## evaluating parents dependencies if needed
#     for aux in evaluation.srp[2].ancestors_models
#         evaluate!(aux, df)
#     end
#     model = filter(x -> x.name == evaluation.srp[2].model, output_node.cpd.distributions)[1].model
#     evaluate!(model, df)
#     pf = sum(df.output .< 0) / n
#     result[evaluation.evidence] = pf
# end
