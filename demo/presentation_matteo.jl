using EnhancedBayesianNetworks
using GraphRecipes

Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")

## EarthquakeNode
earthquake = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPD_earthquake = RootCPD(:earthquake, [earthquake])
earthquake1 = [
    ModelParameters(:output, :model1, [Parameter(-8.1e-5, :flow)]),
]
earthquake2 = [
    ModelParameters(:output, :model2, [Parameter(-0.1e-5, :flow)]),
]
parameters_vector = [earthquake1, earthquake2]
earthquake_node = RootNode(CPD_earthquake, parameters_vector)

## ExtremeRainNode
extremerain = NamedCategorical([:low, :high], [0.5, 0.5])
CPD_extremerain = RootCPD(:extremerain, [extremerain])
extremerain_node = RootNode(CPD_extremerain)

## Dispersivity long_vNode
disp_longv_distribution = truncated(Normal(1, 1), lower=0)
CPD_disp_longv = RootCPD(:disp_longv, [disp_longv_distribution])
disp_longv_node = RootNode(CPD_disp_longv)

## K_zNode
Kz_parents = [extremerain_node]
Kz_distribution1 = truncated(Normal(1, 1), lower=0)
Kz_distribution2 = truncated(Normal(4, 2), lower=0)
CPD_Kz = StdCPD(:Kz, name.(Kz_parents), [2], [Kz_distribution1, Kz_distribution2])
Kz_node = StdNode(CPD_Kz, Kz_parents)

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
performance = df -> 2.2 .- df.c_max


output_target = :output
output_parents = [earthquake_node, disp_longv_node, Kz_node]
output_parental_ncat = [2]

extractor1 = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]
model1 = _get_th_model(sourcedir, uqinputs, format_dict, extractor1, true)
models1 = [model1, Model(max_t, :T_max), Model(max_c, :c_max), Model(max_h, :head_max), Model(performance, :output)]
model_with_name1 = ModelWithName(:model1, models1)
model_with_name2 = ModelWithName(:model2, models1)

output_CPD = FunctionalCPD(:output, name.(output_parents), output_parental_ncat, [model_with_name1, model_with_name2])
output_node = FunctionalNode(output_CPD, output_parents, "discrete")

nodes = [earthquake_node, extremerain_node, disp_longv_node, Kz_node, output_node]
ebn = EnhancedBayesNet(nodes)
show(ebn)

""" Solving EnhancedBayesNet """
groups = markov_envelopes(ebn)
rdag, dag_names = _reduce_ebn_to_rbn(ebn)
graphplot(
    rdag,
    method=:tree,
    names=dag_names,
    fontsize=9,
    nodeshape=:ellipse,
    markercolor=map(x -> x.type == "discrete" ? "lightgreen" : "orange", filter(x -> x.cpd.target ∈ dag_names, ebn.nodes)),
    linecolor=:darkgrey,
)

empty_srp_table_with_evidence = _build_node_evidence_after_reduction(ebn, rdag, dag_names, output_node)
a = _functional_node_after_reduction(ebn, empty_srp_table_with_evidence, output_node)

function update_model_parameter_for_given_evidence(default_parameters::Vector{UQInput}, specific_inputs::Vector{UQInput})
    mes = map(x -> x.name, specific_inputs)
    updated_uqinputs = filter(x -> x.name ∉ mes, default_parameters)
    append!(updated_uqinputs, specific_inputs)
    return updated_uqinputs
end

n = 1
result = Dict()
for evaluation in a
    df = UncertaintyQuantification.sample(update_model_parameter_for_given_evidence(uqinputs, evaluation.srp[2].inputs), n)
    ## evaluating parents dependencies if needed
    for aux in evaluation.srp[2].ancestors_models
        evaluate!(aux, df)
    end
    model = filter(x -> x.name == evaluation.srp[2].model, output_node.cpd.distributions)[1].model
    evaluate!(model, df)
    pf = sum(df.output .< 0) / n
    result[evaluation.evidence] = pf
end
