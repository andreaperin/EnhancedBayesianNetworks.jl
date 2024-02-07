using EnhancedBayesianNetworks
using Distributions
using Plots

node_emission = DiscreteRootNode(:em, Dict(:no => 0.0, :yes => 1.0))
node_time = DiscreteRootNode(:time, Dict(:f => 1.0, :s => 0.0, :t => 0.0))
node_windvelocity = DiscreteRootNode(:wv, Dict(:fast => 0.2, :slow => 0.8))

parents_extremeprecipitation = Vector{AbstractNode}([node_emission, node_time])
extremeprecipitation_states = Dict(
    [:no, :f] => Dict(:lev1 => 0.39, :lev2 => 0.37, :lev3 => 0.15, :lev4 => 0.07, :lev5 => 0.02),
    [:no, :s] => Dict(:lev1 => 0.39, :lev2 => 0.52, :lev3 => 0.0, :lev4 => 0.07, :lev5 => 0.02),
    [:no, :t] => Dict(:lev1 => 0.39, :lev2 => 0.52, :lev3 => 0.0, :lev4 => 0.07, :lev5 => 0.02),
    [:yes, :f] => Dict(:lev1 => 0.39, :lev2 => 0.52, :lev3 => 0.0, :lev4 => 0.07, :lev5 => 0.02),
    [:yes, :s] => Dict(:lev1 => 0.39, :lev2 => 0.37, :lev3 => 0.15, :lev4 => 0.07, :lev5 => 0.02),
    [:yes, :t] => Dict(:lev1 => 0.39, :lev2 => 0.37, :lev3 => 0.15, :lev4 => 0.03, :lev5 => 0.06)
)
node_extremeprec = DiscreteStandardNode(:ex_p, parents_extremeprecipitation, extremeprecipitation_states)

parents_waterlevel = Vector{AbstractNode}([node_extremeprec])
waterlevel_states = Dict(
    [:lev1] => Dict(:low => 0.9, :high => 0.1),
    [:lev2] => Dict(:low => 0.8, :high => 0.2),
    [:lev3] => Dict(:low => 0.6, :high => 0.4),
    [:lev4] => Dict(:low => 0.2, :high => 0.8),
    [:lev5] => Dict(:low => 0.1, :high => 0.9)
)
parameters = Dict(
    :low => [Parameter(-296.2, :wl)],
    :high => [Parameter(296.2, :wl)])
node_waterlev = DiscreteStandardNode(:wl, parents_waterlevel, waterlevel_states, parameters)

parents_debrisflow = Vector{AbstractNode}([node_extremeprec])
debrisflow_states = Dict(
    [:lev1] => Dict(:st1 => 0.444, :st2 => 0.519, :st3 => 0.027, :st4 => 0.01),
    [:lev2] => Dict(:st1 => 0.4, :st2 => 0.543, :st3 => 0.04, :st4 => 0.017),
    [:lev3] => Dict(:st1 => 0.2, :st2 => 0.3, :st3 => 0.352, :st4 => 0.148),
    [:lev4] => Dict(:st1 => 0.0, :st2 => 0.0, :st3 => 0.852, :st4 => 0.148),
    [:lev5] => Dict(:st1 => 0.0, :st2 => 0.0, :st3 => 0.0, :st4 => 1.0)
)
parameters_dbf = Dict(
    :st1 => [Parameter(10, :dbf)],
    :st2 => [Parameter(20, :dbf)],
    :st3 => [Parameter(40, :dbf)],
    :st4 => [Parameter(50, :dbf)],
)
node_debrisflow = DiscreteStandardNode(:dbf, parents_debrisflow, debrisflow_states, parameters_dbf)

parents_waveraising = Vector{AbstractNode}([node_windvelocity])
waverising_states = Dict(
    [:slow] => Rayleigh(0.387),
    [:fast] => Rayleigh(2.068)
)
node_waverising = ContinuousStandardNode(:wr, parents_waveraising, waverising_states)

parents_overtopping = Vector{AbstractNode}([node_waverising, node_waterlev])
overtopping_model1 = Model(df -> df.wl .+ df.wr .+ 291, :level)
overtopping_model2 = Model(df -> df.wl .+ df.wr .- 291, :level)
overtopping_models = Dict(
    [:low] => [overtopping_model1],
    [:high] => [overtopping_model2]
)
overtopping_simulation = Dict(
    [:low] => MonteCarlo(400),
    [:high] => MonteCarlo(400)
)
overtopping_performances = Dict(
    [:low] => df -> df.level,
    [:high] => df -> df.level
)
parameters_ov_t = Dict(
    :safe_ov_t => [Parameter(1, :ov_t)],
    :fail_ov_t => [Parameter(-1, :ov_t)]
)

node_overtopping = DiscreteFunctionalNode(:ov_t, parents_overtopping, overtopping_models, overtopping_performances, overtopping_simulation, parameters_ov_t)

parents_st_dmg = Vector{AbstractNode}([node_overtopping, node_debrisflow])
st_dmg_model1 = Model(df -> df.ov_t .* df.dbf .+ 91, :dmg)
st_dmg_model2 = Model(df -> df.ov_t .* df.dbf .- 91, :dmg)
st_dmg_models = Dict(
    [:fail_ov_t, :st1] => [st_dmg_model1],
    [:safe_ov_t, :st1] => [st_dmg_model2],
    [:fail_ov_t, :st2] => [st_dmg_model1],
    [:safe_ov_t, :st2] => [st_dmg_model2],
    [:fail_ov_t, :st3] => [st_dmg_model1],
    [:safe_ov_t, :st3] => [st_dmg_model2],
    [:fail_ov_t, :st4] => [st_dmg_model1],
    [:safe_ov_t, :st4] => [st_dmg_model2]
)
st_dmg_performances = Dict(
    [:fail_ov_t, :st1] => df -> df.dmg,
    [:safe_ov_t, :st1] => df -> df.dmg,
    [:fail_ov_t, :st2] => df -> df.dmg,
    [:safe_ov_t, :st2] => df -> df.dmg,
    [:fail_ov_t, :st3] => df -> df.dmg,
    [:safe_ov_t, :st3] => df -> df.dmg,
    [:fail_ov_t, :st4] => df -> df.dmg,
    [:safe_ov_t, :st4] => df -> df.dmg
)

st_dmg_node = DiscreteFunctionalNode(:st_dmg, parents_st_dmg, st_dmg_models, st_dmg_performances)

nodes = [node_debrisflow, node_emission, node_extremeprec, node_time, node_waterlev, node_waverising, node_windvelocity, node_overtopping, st_dmg_node]

ebn = EnhancedBayesianNetwork(nodes)

gr();
nodesize = 0.1
fontsize = 18
EnhancedBayesianNetworks.plot(ebn, :tree, nodesize, fontsize)
Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/Silvia_ebn.png")

rbn = reduce_ebn_markov_envelopes(ebn)
EnhancedBayesianNetworks.plot(rbn[1], :tree, nodesize, fontsize)
Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/Silvia_rbn.png")

e_ebn = evaluate_ebn(ebn)

