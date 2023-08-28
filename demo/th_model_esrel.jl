using EnhancedBayesianNetworks
using Plots
# using JLD2

include("./TH_model/1_model_build/model_builder.jl")
include("./TH_model/1_model_build/heads_model.jl")

## EARTHQUAKE NODES => Porosity RV
earthquake = DiscreteRootNode(:EQ, Dict(:earthquake => 0.0001, :no_earthquake => 0.9999))

porosity_name = :porosity
porosity_parents = [earthquake]
porosity_states = Dict(
    [:earthquake] => truncated(Normal(0.1, 0.05); lower=0, upper=4),
    [:no_earthquake] => truncated(Normal(3.0, 0.05); lower=0, upper=4)
)
porosity_node = ContinuousStandardNode(porosity_name, porosity_parents, porosity_states)

kx = ContinuousRootNode(:KX, truncated(Normal(9.81 * 10^(-6), 10^(-4)); lower=0, upper=0.001))
kz = ContinuousRootNode(:KZ, truncated(Normal(9.81 * 10^(-6), 10^(-4)); lower=0, upper=0.001))
disp_long_hz = ContinuousRootNode(:disp_long_h, Uniform(10, 60))
disp_long_vr = ContinuousRootNode(:disp_long_v, Uniform(1, 6))


## GLOBAL_WARMING => DiffusionCoefficient RV
global_warming = DiscreteRootNode(:GW, Dict(:warming => 0.7, :astoday => 0.1, :cooling => 0.2))

diff_coeff_name = :diff_coefficient
diff_coeff_parents = [global_warming]
diff_coeff_states = Dict(
    [:warming] => truncated(Normal(2 * 10^(-6), 10^(-6)); lower=0, upper=0.00001),
    [:astoday] => truncated(Normal(2 * 10^(-8), 10^(-7)); lower=0, upper=0.00001),
    [:cooling] => truncated(Normal(2 * 10^(-9), 10^(-6)); lower=0, upper=0.00001)
)
diff_coeff_node = ContinuousStandardNode(diff_coeff_name, diff_coeff_parents, diff_coeff_states)


## EXTREMERAIN => head_factor Par
extremerain = DiscreteRootNode(:er, Dict(:extremerain => 0.4, :no_extremerain => 0.6), Dict(:extremerain => [Parameter(1.2, :head_factor)], :no_extremerain => [Parameter(0.8, :head_factor)]))
timescenario = DiscreteRootNode(:time, Dict(:short => 0.5, :long => 0.5), Dict(:short => [Parameter(10000.0, :sim_duration)], :long => [Parameter(100000.0, :sim_duration)]))
flowtype = DiscreteRootNode(:flow_type, Dict(:steadystate => 1, :transient => 0), Dict(:steadystate => [Parameter(0e-5, :specific_storage)], :transient => [Parameter(0.1e-5, :specific_storage)]))

```
        Model Node
```
functional_name = :THC

functional_parents = [porosity_node, kx, kz, disp_long_hz, disp_long_vr, diff_coeff_node, extremerain, timescenario, flowtype]

if Sys.isapple()
    sourcedir = joinpath(pwd(), "demo/TH_model/Projects_Unix/230823_1841")
elseif Sys.islinux()
    sourcedir = joinpath(pwd(), "demo/TH_model/Projects_Unix/230810_1605")
elseif Sys.iswindows()
    sourcedir = joinpath(pwd(), "demo\\TH_model\\Projects_Win\\20230818_1538")
end

#####################################################################################################################
############################################ Standard inputs ########################################################
KX = RandomVariable(truncated(Normal(9.81 * 10^(-6), 10^(-4)); lower=0, upper=0.001), :KX)
KZ = RandomVariable(truncated(Normal(9.81 * 10^(-6), 10^(-4)); lower=0, upper=0.001), :KZ)
disp_long_h = RandomVariable(Uniform(10, 60), :disp_long_h)
disp_long_v = RandomVariable(Uniform(1, 6), :disp_long_v)
diff_coefficient = RandomVariable(truncated(Normal(2(10^(-8)), 10^(-7)); lower=0, upper=0.00001), :diff_coefficient)
porosity = RandomVariable(truncated(Normal(0.15, 0.2); lower=0, upper=4), :porosity)

specific_storage = Parameter(0e-5, :specific_storage)
sim_duration = Parameter(1000.0, :sim_duration)

head_factor = RandomVariable(Uniform(0.8, 1.2), :head_factor)

numberformats = Dict(
    :KX => ".2e",
    :KZ => ".2e",
    :disp_long_v => ".2e",
    :disp_long_h => ".2e",
    :diff_coefficient => ".2e",
    :porosity => ".2f",
    :specific_storage => ".2e",
    :sim_duration => ".1f"
)

inputs = [head_factor, KZ, KX, disp_long_h, disp_long_v, porosity, diff_coefficient, specific_storage, sim_duration]

output_file = "smoker_cxz.plt"

###################################################################################################################

model0 = HeadsModel(df -> heads_values(df))
model1 = build_th_model(sourcedir, inputs, [concentration_radionuclides(output_file)], numberformats, true)
model2 = UncertaintyQuantification.Model(df -> sum_surface_concentration(df), :surface_conc)
models = [model0, model1, model2]
performance = df -> 0.9 .- df.surface_conc


functional_models = Dict(
    [:extremerain, :short, :steadystate] => models,
    [:extremerain, :short, :transient] => models,
    [:extremerain, :long, :steadystate] => models,
    [:extremerain, :long, :transient] => models,
    [:no_extremerain, :short, :steadystate] => models,
    [:no_extremerain, :short, :transient] => models,
    [:no_extremerain, :long, :steadystate] => models,
    [:no_extremerain, :long, :transient] => models
)

functional_simulations = Dict(
    [:extremerain, :short, :steadystate] => MonteCarlo(2),
    [:extremerain, :short, :transient] => MonteCarlo(2),
    [:extremerain, :long, :steadystate] => MonteCarlo(2),
    [:extremerain, :long, :transient] => MonteCarlo(2),
    [:no_extremerain, :short, :steadystate] => MonteCarlo(2),
    [:no_extremerain, :short, :transient] => MonteCarlo(2),
    [:no_extremerain, :long, :steadystate] => MonteCarlo(2),
    [:no_extremerain, :long, :transient] => MonteCarlo(2)
)

functional_performances = Dict(
    [:extremerain, :short, :steadystate] => df -> 0.9 .- df.surface_conc,
    [:extremerain, :short, :transient] => df -> 0.9 .- df.surface_conc,
    [:extremerain, :long, :steadystate] => df -> 0.9 .- df.surface_conc,
    [:extremerain, :long, :transient] => df -> 0.9 .- df.surface_conc,
    [:no_extremerain, :short, :steadystate] => df -> 0.9 .- df.surface_conc,
    [:no_extremerain, :short, :transient] => df -> 0.9 .- df.surface_conc,
    [:no_extremerain, :long, :steadystate] => df -> 0.9 .- df.surface_conc,
    [:no_extremerain, :long, :transient] => df -> 0.9 .- df.surface_conc
)

functional_node = DiscreteFunctionalNode(functional_name, functional_parents, functional_models, functional_performances, functional_simulations)

nodes = [earthquake, global_warming, porosity_node, kz, kx, disp_long_hz, disp_long_vr, diff_coeff_node, extremerain, timescenario, flowtype, functional_node]

ebn = EnhancedBayesianNetwork(nodes)

# EnhancedBayesianNetworks.plot(ebn, :spring, 0.1, 8)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/ebn_salt_dome.png")

# rbn = reduce_ebn_standard(ebn)
# EnhancedBayesianNetworks.plot(rbn, :spring, 0.1, 8)
# Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/Rbn_salt_dome.png")

# a = evaluate_ebn(ebn)

# @save "prova.jld2" a


# rbn = a[1]
# bn = BayesianNetwork(rbn)
# query = [:x]
# e = Dict(:f1 => :f)
# inf = InferenceState(bn, query, e)


# infer(bn, query, e)
