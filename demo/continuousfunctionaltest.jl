using EnhancedBayesianNetworks
using Plots

root1 = DiscreteRootNode(:a, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :a)], :n => [Parameter(0, :a), Parameter(5.6, :x1)]))
root2 = DiscreteRootNode(:b, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :b)], :no => [Parameter(5.5, :b)]))
root3 = ContinuousRootNode(RandomVariable(Normal(), :c))

root4 = ContinuousRootNode(RandomVariable(Normal(), :e))

functional1_name = :f1
functional1_parents = [root2, root3]
functional1_model1 = Model(df -> (df.c .^ 2) ./ 2, :f1)
functional1_model2 = Model(df -> (df.c .^ 3) ./ 2, :f1)
functional1_models = OrderedDict(
    [:yes] => [functional1_model1],
    [:no] => [functional1_model2],
)
functional1_node = ContinuousFunctionalNode(functional1_name, functional1_parents, functional1_models)

functional3_name = :f2
functional3_parents = [root2, functional1_node]
functional3_model1 = Model(df -> (df.f1 .^ 2) ./ 2, :f2)
functional3_model2 = Model(df -> (df.f1 .^ 3) ./ 2, :f2)
functional3_models = OrderedDict(
    [:yes] => [functional3_model1],
    [:no] => [functional3_model2],
)
functional3_node = ContinuousFunctionalNode(functional3_name, functional3_parents, functional3_models)


functional2_name = :output
functional2_parents = [root1, functional3_node, root4]
functional2_model1 = Model(df -> df.a .- df.f1, :output)
functional2_model2 = Model(df -> df.a .+ df.f1, :output)

functional2_models = OrderedDict(
    [:y] => [functional2_model1],
    [:n] => [functional2_model2]
)

functional2_simulations = OrderedDict(
    [:y] => MonteCarlo(200),
    [:n] => MonteCarlo(300),
)
functional2_performances = OrderedDict(
    [:y] => df -> 1 .- 2 .* df.output,
    [:n] => df -> 1 .- 2 .* df.output,
)
functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, functional2_models, functional2_performances, functional2_simulations)

ebn = EnhancedBayesianNetwork([root1, root2, root3, root4, functional1_node, functional2_node, functional3_node])
rbns = reduce_ebn_markov_envelopes(ebn)

a = EnhancedBayesianNetworks._build_structuralreliabilityproblem_node(rbns[1], functional2_node)

evaluate_rbn(rbns[1])

rbns[1].nodes
