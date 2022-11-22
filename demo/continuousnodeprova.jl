include("../bn.jl")
include("../buildmodel_TH.jl")
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
dispersivitivy_longv1 = truncated(Normal(1, 1), lower=0)
dispersivitivy_longv2 = truncated(Normal(2, 1), lower=0)
dispersivitivy_longv3 = truncated(Normal(2, 2), lower=0.1)
dispersivitivy_longv4 = truncated(Normal(10, 1), lower=0)
dispersivitivy_longv5 = truncated(Normal(20, 1), lower=0)
dispersivitivy_longv6 = truncated(Normal(22, 2), lower=0.1)
dispersivitivy_longv7 = truncated(Normal(22, 2), lower=0.1)
dispersivitivy_longv8 = truncated(Normal(22, 2), lower=0.1)
dispersivitivy_longv9 = truncated(Normal(22, 2), lower=0.1)

dist = [dispersivitivy_longv1, dispersivitivy_longv2, dispersivitivy_longv3, dispersivitivy_longv4, dispersivitivy_longv5, dispersivitivy_longv6, dispersivitivy_longv7, dispersivitivy_longv8, dispersivitivy_longv2]
CPDd = CategoricalCPD(:flow, [:time_scenario, :grandparent], [3, 3], dist)
dispersivitivy_longv = StdNode(CPDd, parents_dispersivitivy_longv)


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
Kz3_3_1 = truncated(Normal(1, 1), lower=0)
Kz3_3_2 = truncated(Normal(1, 1), lower=0)
Kz_cpd = [
    Kz1_1_1,
    Kz1_1_2,
    Kz1_2_1,
    Kz1_2_2,
    Kz1_3_1,
    Kz1_3_2,
    Kz2_1_1,
    Kz2_1_2,
    Kz2_2_1,
    Kz2_2_2,
    kz2_3_1,
    Kz2_3_2,
    Kz3_1_1,
    Kz3_1_2,
    Kz3_2_1,
    Kz3_3_1,
    Kz3_3_2
]

CPDKz = CategoricalCPD{Distribution}(:Kz, [:time_scenario, :grandparent], [3, 3, 2], Kz_cpd)
Kz = StdNode(CPDKz, parents_Kz)

parents_th = [dispersivitivy_longv, Kz]
sim_th = MonteCarlo(2)
default_file = "model_TH/inputs/default_th_values.xlsx"
default_inputs_th = get_default_inputs_and_format(default_file)
sourcedir = joinpath(pwd(), "model_TH")
source_file = "smoker.data"
extras = String[]
solvername = "smokerV3TC"
output_parameters = xlsx2output_parameter(default_file)
performances = build_performances(output_parameters)
th_node = ModelNode(:th_node, parents_th, default_inputs_th, sourcedir, source_file, extras, solvername, output_parameters, performances, true, sim_th)
