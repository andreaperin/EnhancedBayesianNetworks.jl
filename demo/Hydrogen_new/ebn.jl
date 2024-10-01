using EnhancedBayesianNetworks
# using JDL2
if Sys.isapple()
    include("/Users/andreaperin_macos/Documents/Code/Hydrogen_project/JuliaHyram/wrapper.jl")
elseif Sys.iswindows()
    include("D:/Code/Hydrogen_project/JuliaHyram/wrapper.jl")
else
    error("missing linux python environment for Hyra")
end

node_leak = DiscreteRootNode(:Leak, Dict(:YL => 0.1, :NL => 0.9))
node_detector = DiscreteRootNode(:Detect, Dict(:WD => 0.9, :NWD => 0.1))

release_states = Dict(
    [:YL, :WD] => Dict(:Nor => 1.0, :Sr => 0.0, :Mr => 0.0, :Br => 0.0),
    [:YL, :NWD] => Dict(:Nor => 0.0, :Sr => 0.4, :Mr => 0.3, :Br => 0.3),
    [:NL, :WD] => Dict(:Nor => 1.0, :Sr => 0.0, :Mr => 0.0, :Br => 0.0),
    [:NL, :NWD] => Dict(:Nor => 1.0, :Sr => 0.0, :Mr => 0.0, :Br => 0.0))
node_release = DiscreteChildNode(:Rel, [node_leak, node_detector], release_states)

ignition_states = Dict(
    [:Nor] => Dict(:No => 1.0, :Rel => 0.0, :Imm => 0.0, :Del => 0.0),
    [:Sr] => Dict(:No => 0.0, :Rel => 0.988, :Imm => 0.008, :Del => 0.004),
    [:Mr] => Dict(:No => 0.0, :Rel => 0.92, :Imm => 0.053, :Del => 0.027),
    [:Br] => Dict(:No => 0.0, :Rel => 0.66, :Imm => 0.230, :Del => 0.110),
)
ignition_parameters = Dict(
    :No => [Parameter(3, :scenario)],
    :Rel => [Parameter(0, :scenario)],
    :Imm => [Parameter(1, :scenario)],
    :Del => [Parameter(2, :scenario)]
)
node_ignition = DiscreteChildNode(:Ignition, [node_release], ignition_states, ignition_parameters)

node_time = DiscreteRootNode(:Time, Dict(:long_exp => 0.3, :short_exp => 0.7), Dict(:long_exp => [Parameter(30, :time)], :short_exp => [Parameter(60, :time)]))
node_operator_radius = DiscreteRootNode(:Operators, Dict(:close => 0.1, :far => 0.9), Dict(:close => [Parameter(5, :r_operators)], :far => [Parameter(15, :r_operators)]))

node_t_amb = ContinuousRootNode(:t_amb, Normal(288.15, 20))
node_p_amb = ContinuousRootNode(:p_amb, Normal(101_325, 5_000))
node_t_h2 = ContinuousRootNode(:t_h2, Normal(287.15, 20))
node_p_h2 = ContinuousRootNode(:p_h2, Normal(13_420_000, 50_000))
node_d_or = ContinuousRootNode(:d_or, (Uniform(0.00356 - 0.003, 0.00356 + 0.003)))
node_rel_humidity = ContinuousRootNode(:humidity, Uniform(0.85 - 0.1, 0.85 + 0.1))
node_Θ = ContinuousRootNode(:Θ, Uniform(50 / 180 * π, 130 / 180 * π))


## Model node
x = range(-30, 30, 10)
y = range(-30, 30, 10)
z = range(0, 4, 10)
a = collect(Iterators.product(x, y, z))
loc = vec(a)

model_1 = Model(df -> wrapper_model.(
        df.scenario,
        df.t_amb,
        df.p_amb,
        df.t_h2,
        df.p_h2,
        df.d_or,
        df.Θ,
        1,
        "yuce",
        df.humidity,
        [loc],
        "bst"
    ), :res)

threshold_op = 4000
## 1 percent probability of lethal injury 

threshold_1stlevel_burn = 115 # [(kW/m2 ) 4/3 s]
threshold_2ndlevel_burn = 250 # [(kW/m2 ) 4/3 s]
threshold_3rdlevel_burn = 900 # [(kW/m2 ) 4/3 s]
# dose that would be needed to produce first-degree burns in 1% of the exposed population.

model_mfr = Model(df -> mass_flow_rate_extractor(df), :mfr)
## Plume Specific
model_t_plume = Model(df -> T_extractor_plume(df), :T_plume)
model_massfracs_plume = Model(df -> mass_extractor_plume(df), :massf_plume)
model_molefracs_plume = Model(df -> mole_extractor_plume(df), :molef_plume)
## Fire Specific
th_dose_model = Model(df -> th_dose_extractor(df), :th_dose)
radius_0_1st_level_model = Model(df -> burn_radius_0(df, threshold_1stlevel_burn, [loc]), :burn_r_0_1st)
radius_0_2nd_level_model = Model(df -> burn_radius_0(df, threshold_2ndlevel_burn, [loc]), :burn_r_0_2nd)
radius_0_3rd_level_model = Model(df -> burn_radius_0(df, threshold_3rdlevel_burn, [loc]), :burn_r_0_3rd)

radius_C_1st_level_model = Model(df -> burn_radius_C(df, threshold_1stlevel_burn, [loc]), :burn_r_C_1st)
radius_C_2nd_level_model = Model(df -> burn_radius_C(df, threshold_2ndlevel_burn, [loc]), :burn_r_C_2nd)
radius_C_3th_level_model = Model(df -> burn_radius_C(df, threshold_3rdlevel_burn, [loc]), :burn_r_C_3th)

## Overpressure Specific
overpressures_model = Model(df -> overpressures_extractor(df), :op)
explosion_radius_0_model = Model(df -> explosion_radius_0(df, threshold_op, [loc]), :explosion_r_0)
explosion_radius_c_model = Model(df -> explosion_radius_C(df, threshold_op, [loc]), :explosion_r_C)

function hyram_performance(scenario, r_operators, burn_r_C_3th, explosion_r_C)
    if scenario == 0
        return 1
    elseif scenario == 1
        return r_operators - burn_r_C_3th
    elseif scenario == 2
        return r_operators - explosion_r_C
    elseif scenario == 3
        return 1
    end
end
function overall_performance(df)
    hyram_performance.(df.scenario, df.r_operators, df.burn_r_C_3th, df.explosion_r_C)
end

hyram_models = [model_1, model_mfr, model_t_plume, overpressures_model, th_dose_model, radius_0_1st_level_model, radius_0_2nd_level_model, radius_0_3rd_level_model, radius_C_1st_level_model, radius_C_2nd_level_model, radius_C_3th_level_model, explosion_radius_0_model, explosion_radius_c_model]

hyram_parents = [node_t_amb, node_p_amb, node_t_h2, node_p_h2, node_d_or, node_Θ, node_rel_humidity, node_ignition, node_time, node_operator_radius]

hyram_simulations = MonteCarlo(2)
hyram_perf = df -> overall_performance(df)

node_hyram = DiscreteFunctionalNode(:HYRAM, hyram_parents, hyram_models, hyram_perf, hyram_simulations)

nodes = [node_leak, node_detector, node_release, node_ignition, node_time, node_operator_radius, node_p_amb, node_t_amb, node_p_h2, node_t_h2, node_d_or, node_rel_humidity, node_Θ, node_hyram]

ebn = EnhancedBayesianNetwork(nodes)

eebn = EnhancedBayesianNetworks.evaluate(ebn)

# path = pwd() * "/demo/Hydrogen_new/ebn/"
# save_object(path * "ebn" * string(hyram_simulations.n) * string(typeof(hyram_simulations)) * ".jld2", eebn)