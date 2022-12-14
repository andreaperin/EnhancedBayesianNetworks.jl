include("../bn.jl")
include("../model_TH_macos/buildmodel_TH.jl")
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


default_file = "model_TH_macos/inputs/default_th_values.xlsx"
sourcedir = Sys.isapple() ? "model_TH_macos" : "model_TH_win"
sourcedir = joinpath(pwd(), sourcedir)
format_dict = readxlsxinput(default_file)[3]
uqinputs = readxlsxinput(default_file)[4]
output_parameters = xlsx2output_parameter(default_file)
outputmodel = []
for (k, v) in output_parameters
    push!(outputmodel, OutputModelParameter(k, v))
end
performances_par = filter(i -> typeof(i.definition) == Tuple{Float64,String}, outputmodel)

## TODO This Extractor builder do not work properly (Check with 'build_specific_extractor')
extractor = _build_concentration_extractor2D(
    df2criticalvalue_maximum2D,
    [output_parameters["x_min"], output_parameters["x_max"]],
    [output_parameters["z_min"], output_parameters["z_max"]])

extractor2 = build_specific_extractor(output_parameters["output_filename"],
    [output_parameters["x_min"], output_parameters["x_max"]],
    [output_parameters["z_min"], output_parameters["z_max"]],
    output_parameters["quantity_of_interest"])
default_model = _get_th_model(sourcedir, format_dict, uqinputs, extractor, false)
default_model2 = _get_th_model(sourcedir, format_dict, uqinputs, extractor2, true)

samples = UncertaintyQuantification.sample(uqinputs, 4)







# evaluate!(default_model, samples)





parents_th = [node_simduration, dispersivitivy_longv, Kz]



# default_model = _get_default_th_model(default_file, true)

# sim_th = MonteCarlo(2)
# default_inputs_th = get_default_inputs_and_format(default_file)
# sourcedir = joinpath(pwd(), "model_TH_macos")
# source_file = "smoker.data"
# extras = String[]
# solvername = "smokerV3TC"
# output_parameters = xlsx2output_parameter(default_file)
# performances = build_performances(output_parameters)
# workdir = get_workdir(default_inputs_th["UQInputs"], sourcedir)
# extractor = build_specific_extractor(
#     output_parameters["output_filename"],
#     [output_parameters["x_min"], output_parameters["x_max"]],
#     [output_parameters["z_min"], output_parameters["z_max"]],
#     output_parameters["quantity_of_interest"]
# )
# solver = Solver(joinpath(sourcedir, solvername), "", source_file)
# format_dict = Dict{Symbol,FormatSpec}()
# for el in default_inputs_th["FormatSpec"]
#     for (k, v) in el
#         format_dict[k] = v
#     end
# end
# default_model = ExternalModel(sourcedir, [source_file], extras, format_dict, workdir, extractor, solver, true)


# th_node = ModelNode(:th_node, parents_th, default_inputs_th, sourcedir, source_file, extras, solvername, output_parameters, performances, true, sim_th)
