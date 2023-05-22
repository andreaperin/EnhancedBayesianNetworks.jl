using EnhancedBayesianNetworks
using Plots

root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
root3 = ContinuousRootNode(:z, Normal(), [[-0.5, 0.03], [0.03, 0.1]])

standard1_name = :α
standard1_parents = [root1, root2]
standard1_states = OrderedDict(
    [:y, :yes] => Dict(:a => 0.2, :b => 0.8),
    [:n, :yes] => Dict(:a => 0.3, :b => 0.7),
    [:y, :no] => Dict(:a => 0.4, :b => 0.6),
    [:n, :no] => Dict(:a => 0.5, :b => 0.5)
)
standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
standard1_node = DiscreteStandardNode(standard1_name, standard1_parents, standard1_states, standard1_parameters)

standard2_name = :β
standard2_parents = [root1]
standard2_states = OrderedDict(
    [:y] => Normal(),
    [:n] => Normal(2, 2)
)
standard2_states = OrderedDict(
    [:y] => Normal(),
    [:n] => Normal(2, 2)
)
standard2_node = ContinuousStandardNode(standard2_name, standard2_parents, standard2_states, [[-1.1, 0], [0, 0.11]], 2)

functional1_name = :f1
functional1_parents = [root2, standard2_node]
functional1_model1 = Model(df -> (df.y .^ 2 + df.β .^ 2) ./ 2, :value1)
functional1_model2 = Model(df -> (df.y .^ 2 - df.β .^ 2) ./ 2, :value1)
functional1_models = OrderedDict(
    [:yes] => [functional1_model1],
    [:no] => [functional1_model2],
)
functional1_simulations = OrderedDict(
    [:yes] => MonteCarlo(200),
    [:no] => MonteCarlo(300),
)
functional1_performances = OrderedDict(
    [:yes] => df -> 1 .- 2 .* df.value1,
    [:no] => df -> 1 .- 2 .* df.value1,
)
functional1_node = DiscreteFunctionalNode(functional1_name, functional1_parents, functional1_models, functional1_performances, functional1_simulations)

functional2_name = :f2
functional2_parents = [standard1_node, root3]
functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
functional2_models = OrderedDict(
    [:a] => [functional2_model],
    [:b] => [functional2_model]
)
functional2_simulations = OrderedDict(
    [:a] => MonteCarlo(600),
    [:b] => MonteCarlo(800)
)
functional2_performances = OrderedDict(
    [:a] => df -> 1 .- 2 .* df.value1,
    [:b] => df -> 1 .- 2 .* df.value1
)

functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, functional2_models, functional2_performances, functional2_simulations)


nodes = [standard1_node, root1, root3, root2, functional1_node, functional2_node, standard2_node]

## Already build everything with evidence.
ebn = EnhancedBayesianNetwork(nodes)
rbn1 = reduce_ebn_standard(ebn)
rbns2 = reduce_ebn_markov_envelopes(ebn)
a = evaluate_ebn(ebn)
