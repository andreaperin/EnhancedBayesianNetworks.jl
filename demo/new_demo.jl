using EnhancedBayesianNetworks
using Plots

root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8))
root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
root3 = ContinuousRootNode(RandomVariable(Normal(), :z))

standard1_name = :α
standard1_parents = [root1, root2]
standard1_states = OrderedDict(
    [:yes, :yes] => Dict(:a => 0.2, :b => 0.8),
    [:no, :yes] => Dict(:a => 0.3, :b => 0.7),
    [:yes, :no] => Dict(:a => 0.4, :b => 0.6),
    [:no, :no] => Dict(:a => 0.5, :b => 0.5)
)
standard1_node = DiscreteStandardNode(standard1_name, standard1_parents, standard1_states)

standard2_name = :β
standard2_parents = [root1]
standard2_states = OrderedDict(
    [:yes] => Normal(),
    [:no] => Normal(2, 2)
)
standard2_node = ContinuousStandardNode(standard2_name, standard2_parents, standard2_states)

functional1_name = :f1
functional1_parents = [root2, standard2_node]
functional1_model = Model(df -> sqrt.(df.x .^ 2 + df.β .^ 2), :value1)
functional1_performance = Model(df -> 1 .- 2 .* df.value1, :value2)
functional1_models = OrderedDict(
    [:yes] => [functional1_model, functional1_performance],
    [:no] => [functional1_model, functional1_performance],
)
functional1_node = DiscreteFunctionalNode(functional1_name, functional1_parents, functional1_models)

functional2_name = :f2
functional2_parents = [standard1_node, root3]
functional2_model = Model(df -> sqrt.(df.α .^ 2 + df.z .^ 2), :value1)
functional2_performance = Model(df -> 1 .- 2 .* df.value1, :value2)
functional2_models = OrderedDict(
    [:a] => [functional2_model, functional2_performance],
    [:b] => [functional2_model, functional2_performance]
)
functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, functional2_models)


nodes = [standard1_node, root1, root3, root2, functional1_node, functional2_node, standard2_node]
# a = EnhancedBayesianNetworks._build_digraph(nodes)
# b = EnhancedBayesianNetworks._topological_ordered_dag(nodes)[1]
ebn = EnhancedBayesianNetwork(nodes)
# EnhancedBayesianNetworks.plot(ebn)
# childrensr1 = get_children(ebn, root1)
# parentsr1 = get_parents(ebn, root1)

# childrenss1 = get_children(ebn, standard1_node)
# parentss1 = get_parents(ebn, standard1_node)

a = markov_envelope(ebn)

# rdag = copy(ebn.dag)

# rdag = EnhancedBayesianNetworks._invert_link(rdag, 2, 7)