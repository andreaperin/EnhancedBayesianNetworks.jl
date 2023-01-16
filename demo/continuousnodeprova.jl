include("../bn.jl")
Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")
include("../models_probabilities.jl")

a = NamedCategorical([:first, :second, :third], [0.34, 0.33, 0.33])
CPDa = RootCPD(:time_scenario, a)
timescenario = StdNode(CPDa)

b = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPDprova = RootCPD(:prova, b)
prova = StdNode(CPDprova)

c1 = NamedCategorical([:low, :medium, :high], [0.5, 0.3, 0.2])
CPDc = RootCPD(:grandparent, c1)
grandparent = StdNode(CPDc)


parents_dispersivitivy_longv = [timescenario, grandparent]
dispersivitivy_longv1 = NamedCategorical([:disp1, :disp2], [0.9, 0.1])
dispersivitivy_longv2 = NamedCategorical([:disp1, :disp2], [0.8, 0.2])
dispersivitivy_longv3 = NamedCategorical([:disp1, :disp2], [0.7, 0.3])
dispersivitivy_longv4 = NamedCategorical([:disp1, :disp2], [0.6, 0.4])
dispersivitivy_longv5 = NamedCategorical([:disp1, :disp2], [0.5, 0.5])
dispersivitivy_longv6 = NamedCategorical([:disp1, :disp2], [0.4, 0.6])
dispersivitivy_longv7 = NamedCategorical([:disp1, :disp2], [0.3, 0.7])
dispersivitivy_longv8 = NamedCategorical([:disp1, :disp2], [0.2, 0.8])
dispersivitivy_longv9 = NamedCategorical([:disp1, :disp2], [0.1, 0.9])
dist = [dispersivitivy_longv1, dispersivitivy_longv2, dispersivitivy_longv3, dispersivitivy_longv4, dispersivitivy_longv5, dispersivitivy_longv6, dispersivitivy_longv7, dispersivitivy_longv8, dispersivitivy_longv2]
CPDd = CategoricalCPD(:dispersivity, [:time_scenario, :grandparent], [3, 3], dist)

##TODO DiscreteModelInput and ContinuousModelInput need to be changed according to the new ModelNode Definition (WIP)
model_input_disp = [
    DiscreteModelInput(:disp1, Parameter(2.0, :disp_longv), FormatSpec(".8e")),
    DiscreteModelInput(:disp2, Parameter(5.0, :disp_longv), FormatSpec(".8e"))
]
dispersivitivy_longv = StdNode(CPDd, parents_dispersivitivy_longv, model_input_disp)

parents_simduration = [timescenario]
duration1 = NamedCategorical([:day1, :day10, :day100], [1.0, 0.0, 0.0])
duration2 = NamedCategorical([:day1, :day10, :day100], [0.0, 1.0, 0.0])
duration3 = NamedCategorical([:day1, :day10, :day100], [0.0, 0.0, 1.0])
CPDduration = CategoricalCPD(:simduration, name.(parents_simduration), [3], [duration1, duration2, duration3])
model_input_duration = [
    DiscreteModelInput(:day1, Parameter(1, :sim_duration), FormatSpec("d")),
    DiscreteModelInput(:day10, Parameter(10, :sim_duration), FormatSpec("d")),
    DiscreteModelInput(:day100, Parameter(100, :sim_duration), FormatSpec("d"))
]
node_simduration = StdNode(CPDduration, parents_simduration, model_input_duration)

parents_Kz = [timescenario, grandparent, prova]
Kz1_1_1 = truncated(Normal(1, 1), lower=0)
Kz1_1_2 = truncated(Normal(2, 1), lower=0)
Kz1_2_1 = truncated(Normal(2, 2), lower=0.1)
Kz1_2_2 = truncated(Normal(3, 2), lower=0.1)
Kz1_3_1 = truncated(Normal(4, 2), lower=0.1)
Kz1_3_2 = truncated(Normal(5, 2), lower=0.1)
Kz2_1_1 = truncated(Normal(1, 1), lower=0)
Kz2_1_2 = truncated(Normal(1, 1), lower=0)
Kz2_2_1 = truncated(Normal(1, 1), lower=0)
Kz2_2_2 = truncated(Normal(1, 1), lower=0)
kz2_3_1 = truncated(Normal(1, 1), lower=0)
Kz2_3_2 = truncated(Normal(1, 1), lower=0)
Kz3_1_1 = truncated(Normal(1, 1), lower=0)
Kz3_1_2 = truncated(Normal(1, 1), lower=0)
Kz3_2_1 = truncated(Normal(1, 1), lower=0)
Kz3_2_2 = truncated(Normal(1, 1), lower=0)
Kz3_3_1 = truncated(Normal(1, 1), lower=0)
Kz3_3_2 = truncated(Normal(4, 2), lower=0.1)
Kz_cpd = [Kz1_1_1, Kz1_1_2, Kz1_2_1, Kz1_2_2, Kz1_3_1, Kz1_3_2, Kz2_1_1, Kz2_1_2, Kz2_2_1, Kz2_2_2, kz2_3_1, Kz2_3_2, Kz3_1_1, Kz3_1_2, Kz3_2_1, Kz3_2_2, Kz3_3_1, Kz3_3_2]
CPDKz = CategoricalCPD(:Kz, name.(parents_Kz), [3, 3, 2], Kz_cpd)
model_input_Kz = [
    ContinuousModelInput(:K_z, FormatSpec(".8e"))
]

Kz = StdNode(CPDKz, parents_Kz, model_input_Kz)

bn = StdBayesNet([timescenario, grandparent, prova, Kz, dispersivitivy_longv])
show(bn)


sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
default_file = joinpath(pwd(), sourcedir, "inputs", "default_th_values.xlsx")
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file)[3]
uqinputs = readxlsxinput(default_file)[4]
output_parameters = xlsx2output_parameter(default_file)
outputmodel = []
for (k, v) in output_parameters
    push!(outputmodel, OutputModelParameter(k, v))
end
performances_par = filter(i -> typeof(i.definition) == Tuple{Float64,String}, outputmodel)

output_file_conc = "smoker_cxz.plt"
output_file_temp = "smoker_txz.plt"
output_file_head = "smoker_hxz.plt"

extractor = [_build_temperature_extractor2D(output_file_temp), _build_concentration_extractor2D(output_file_conc), _build_head_extractor2D(output_file_head)]

default_model = _get_th_model(sourcedir, format_dict, uqinputs, extractor, true)

samples = UncertaintyQuantification.sample(uqinputs, 1)

evaluate!(default_model, samples)


# #### PROVA Parse output files
# file_path = joinpath(pwd(), "model_TH_win\\SteadyState_GWflow\\1_days\\2022-12-20-15-49-38\\sample-1\\smoker_cxz.plt")
# var_regex = r"(?<=\")[^,]*?(?=\")"
# day_regex = r"\d*\.\d{2,5}"
# x_regex = r"(?<=i=).*?(?=[,j=])"
# z_regex = r"(?<=j=).*?(?=,)"
# result, x, z = concentrationplt2dict(file_path, var_regex, day_regex, x_regex, z_regex)


# th_node = ModelNode(:th_node, parents_th, default_inputs_th, sourcedir, source_file, extras, solvername, output_parameters, performances, true, sim_th)
