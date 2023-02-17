using DataFrames
include("../CPDs.jl")
include("../nodes.jl")
include("../models_probabilities.jl")
include("../bn.jl")

emission = NamedCategorical([:nothappen, :happen], [0.0, 1.0])
CPD_emission = RootCPD(:emission, emission)
node_emission = StdNode(CPD_emission)

timescenario = NamedCategorical([:first, :second, :third], [1.0, 0.0, 0.0])
CPD_timescenario = RootCPD(:timescenario, timescenario)
node_timescenario = StdNode(CPD_timescenario)

extremeprecipitation1 = NamedCategorical([:lev1, :lev2, :lev3, :lev4, :lev5], [0.39, 0.37, 0.15, 0.07, 0.01])
extremeprecipitation2 = NamedCategorical([:lev1, :lev2, :lev3, :lev4, :lev5], [0.39, 0.52, 0.0, 0.07, 0.01])
extremeprecipitation3 = NamedCategorical([:lev1, :lev2, :lev3, :lev4, :lev5], [0.39, 0.52, 0.0, 0.07, 0.01])
extremeprecipitation4 = NamedCategorical([:lev1, :lev2, :lev3, :lev4, :lev5], [0.39, 0.52, 0.0, 0.07, 0.01])
extremeprecipitation5 = NamedCategorical([:lev1, :lev2, :lev3, :lev4, :lev5], [0.39, 0.37, 0.15, 0.07, 0.01])
extremeprecipitation6 = NamedCategorical([:lev1, :lev2, :lev3, :lev4, :lev5], [0.39, 0.37, 0.15, 0.03, 0.06])
parents_extremeprecipitation = [node_emission, node_timescenario]
CPD_extremeprecipitation = CategoricalCPD(
    :extremeprecipitation, [:emission, :timescenario], [2, 3],
    [extremeprecipitation1,
        extremeprecipitation2,
        extremeprecipitation3,
        extremeprecipitation4,
        extremeprecipitation5,
        extremeprecipitation6]
)
node_extremeprecipitation = StdNode(CPD_extremeprecipitation, parents_extremeprecipitation)

distribution_waterlevel = [
    NamedCategorical([:low, :high], [0.9, 0.1]),
    NamedCategorical([:low, :high], [0.8, 0.2]),
    NamedCategorical([:low, :high], [0.6, 0.4]),
    NamedCategorical([:low, :high], [0.2, 0.8]),
    NamedCategorical([:low, :high], [0.1, 0.9])
]
parents_waterlevel = [node_extremeprecipitation]
parental_ncategories_waterlevel = [5]
CPD_waterlevel = CategoricalCPD(:waterlevel, name.(parents_waterlevel), parental_ncategories_waterlevel, distribution_waterlevel)
node_waterlevel = StdNode(CPD_waterlevel, parents_waterlevel)

debrisflow1 = NamedCategorical([:state1, :state2, :state3, :state4], [0.444, 0.519, 0.027, 0.01])
debrisflow2 = NamedCategorical([:state1, :state2, :state3, :state4], [0.4, 0.543, 0.04, 0.017])
debrisflow3 = NamedCategorical([:state1, :state2, :state3, :state4], [0.2, 0.3, 0.352, 0.148])
debrisflow4 = NamedCategorical([:state1, :state2, :state3, :state4], [0.0, 0.0, 0.852, 0.148])
debrisflow5 = NamedCategorical([:state1, :state2, :state3, :state4], [0.0, 0.0, 0.0, 1.0])
parents_debrisflow = [node_extremeprecipitation]
CPD_debrisflow = CategoricalCPD(
    :debrisflow, name.(parents_debrisflow), [5],
    [debrisflow1,
        debrisflow2,
        debrisflow3,
        debrisflow4,
        debrisflow5]
)
node_debrisflow = StdNode(CPD_debrisflow, parents_debrisflow)

windvelocity = NamedCategorical([:slow, :fast], [0.8, 0.2])
CPD_windvelocity = RootCPD(:windvelocity, windvelocity)
node_windvelocity = StdNode(CPD_windvelocity)
# dict_windvelocity = Dict{Symbol,Dict{String,Vector}}(
#     :slow => Dict(
#         "UQInputs" => [Parameter(1, :windvelocity)],
#         "FormatSpec" => [Dict(:windvelocity => FormatSpec(".8e"))]
#     ),
#     :fast => Dict(
#         "UQInputs" => [Parameter(2, :windvelocity)],
#         "FormatSpec" => [Dict(:windvelocity => FormatSpec(".8e"))]
#     )
# )


waveraising1 = Rayleigh(0.387)
waveraising2 = Rayleigh(2.068)
parents_waveraising = [node_windvelocity]
CPD_waveraising = CategoricalCPD(:waveraising, name.(parents_waveraising), [2], [waveraising1, waveraising2])
node_waveraising = StdNode(CPD_waveraising, parents_waveraising)

child1 = Model(df -> df.waverising .+ df.windvelocity, :child1)
child2 = Model(df -> df.waverising .- df.windvelocity, :child2)
child_model = [child1, child2]
## TODO Child new type of node that accept functional relationship as CPD (Struct model node, add check on parents categories as written in notes)


nodes = [node_windvelocity, node_waveraising, node_emission, node_debrisflow, node_extremeprecipitation, node_timescenario, node_waterlevel]
dag = _build_DiAGraph_from_nodes(nodes)
ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
bn = StdBayesNet(ordered_nodes)
show(bn)











##TODO Review with Jasper, i want just a map function
# discreteparents_states = vec(get_discreteparents_states_combinations(node_child)[2])
# df_dict = Dict()
# for state in discreteparents_states
#     df_dict[state] = DataFrame()
#     evidence = Dict{Symbol,Any}()
#     for i in range(1, length(state))
#         evidence[collect(keys(get_discreteparents_states_combinations(node_child)[1][i]))[1]] = state[i]
#         df_dict[state][!, collect(keys(get_discreteparents_states_combinations(node_child)[1][i]))[1]] = [state[i]]
#     end
#     for continuous_parent in get_continuous_parents(node_child)
#         distribution = evaluate_nodecpd_with_evidence(bn, name(continuous_parent), evidence)
#         df_dict[state][!, name(continuous_parent)] = [collect(values(distribution))[1]]
#     end
# end


## to evaluate the pdf of child node, sample from wave rising dist and evaluate model


# nodes = [node_emission, node_timescenario, node_debrisflow, node_extremeprecipitation, node_waterlevel, node_windvelocity, node_waveraising]
# dag = _build_DiAGraph_from_nodes(nodes)
# ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
# bn = StdBayesNet(ordered_nodes)
# show(bn)

evidence = Assignment(:windvelocity => :fast, :emission => :happen)
a = evaluate_nodecpd_with_evidence(bn, name(node_waveraising), evidence)