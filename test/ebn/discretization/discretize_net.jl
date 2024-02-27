@testset "Discretize EBN" begin
    root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
    root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
    discretization_root3 = ExactDiscretization([-Inf, 0, Inf])
    root3 = ContinuousRootNode(:z1, Normal(), discretization_root3)

    standard1_name = :α
    standard1_parents = [root1, root2]
    standard1_states = Dict(
        [:y, :yes] => Dict(:a => 0.2, :b => 0.8),
        [:n, :yes] => Dict(:a => 0.3, :b => 0.7),
        [:y, :no] => Dict(:a => 0.4, :b => 0.6),
        [:n, :no] => Dict(:a => 0.5, :b => 0.5)
    )
    standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
    standard1_node = DiscreteChildNode(standard1_name, standard1_parents, standard1_states, standard1_parameters)

    standard2_name = :β
    standard2_parents = [root1]
    standard2_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    standard2_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    discretization_standard2 = ApproximatedDiscretization([-Inf, 0.1, Inf], 1.5)
    standard2_node = ContinuousChildNode(standard2_name, standard2_parents, standard2_states, discretization_standard2)

    functional2_name = :f2
    functional2_parents = [standard1_node, root3]
    functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
    functional2_simulation = MonteCarlo(800)
    functional2_performance = df -> 1 .- 2 .* df.value1
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, [functional2_model], functional2_performance, functional2_simulation)

    nodes = [standard1_node, root1, root3, root2, standard2_node, functional2_node]
    ebn = EnhancedBayesianNetwork(nodes)
    @test discretize(ebn) == EnhancedBayesianNetwork(EnhancedBayesianNetworks._discretize!(ebn.nodes))
end