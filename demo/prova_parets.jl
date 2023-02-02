include("../bn.jl")
Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")
include("../CPDs.jl")
include("../nodes.jl")
include("../models_probabilities.jl")

a = NamedCategorical([:first, :second, :third], [0.35, 0.33, 0.32])
CPDa = RootCPD(:t, a)
t = StdNode(CPDa)

b = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPDp = RootCPD(:p, b)
p = StdNode(CPDp)

c = Rayleigh(0.387)
CPDc = RootCPD(:w, c)
w = StdNode(CPDc, Vector{AbstractNode}())

wraising1 = Rayleigh(0.387)
wraising2 = Rayleigh(2.068)
parents_wraising = [p]
parents_waverising_name = NodeNames([:p])
CPD_wraising = CategoricalCPD(:h, parents_waverising_name, [2], [wraising1, wraising2])
h = StdNode(CPD_wraising, parents_wraising)


parents_simduration = [t]
duration1 = NamedCategorical([:day1, :day10, :day100], [1.0, 0.0, 0.0])
duration2 = NamedCategorical([:day1, :day10, :day100], [0.0, 1.0, 0.0])
duration3 = NamedCategorical([:day1, :day10, :day100], [0.0, 0.0, 1.0])
CPDduration = CategoricalCPD(:k, [:t], [3], [duration1, duration2, duration3])
k = StdNode(CPDduration, parents_simduration)


wraising1 = Rayleigh(0.5)
wraising2 = Rayleigh(2.5)
wraising3 = Rayleigh(3.5)
parents_m = [t]
CPD_m = CategoricalCPD(:m, name.(parents_m), [3], [wraising1, wraising2, wraising3])
m = StdNode(CPD_m, parents_m)









par = [t, k, m, h]


ancestors_states, parents_states = get_evidences_vectors_for_modelnodes(par)

b = map_state_to_integer.(ancestors_states[1], repeat([ancestors_states[2]], length(ancestors_states[1])))

d = map_state_to_integer.(parents_states[1], repeat([Vector{Node}(parents_states[2])], length(parents_states[1])))

states_vector = d
reference_vector = Vector{Node}(parents_states[2])




bn = StdBayesNet([t, p, k, m, h])

## TODO for each state_comb, valutare quali node cont_nonroot mancano ad ancestors rispetto a parents, valutarli data la state combination e aggiungere la distribuzione!
evidence = Assignment(:t => :first)
a = evaluate_nodecpd_with_evidence(bn, name(m), evidence)



## evaluate_nodecpd_with_evidence requires BN, maybe the function should be changed to just pass the parents(?)

