@testset "Root Node Discretization" begin

    root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
    root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
    discretization_root3 = ExactDiscretization([0, Inf])
    root3 = ContinuousRootNode(:z1, Normal(), discretization_root3)

    discretization_root4 = ExactDiscretization([-Inf, 0])
    root4 = ContinuousRootNode(:z2, Normal(), discretization_root4)

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
    standard2_node = ContinuousChildNode(standard2_name, standard2_parents, standard2_states)

    functional2_name = :f2
    functional2_parents = [standard1_node, root3]
    functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
    functional2_simulation = MonteCarlo(800)
    functional2_performance = df -> 1 .- 2 .* df.value1
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, [functional2_model], functional2_performance, functional2_simulation)

    nodes = [standard1_node, root1, root3, root2, root4, standard2_node, functional2_node]
    ebn = EnhancedBayesianNetwork(nodes)

    @test_logs (:warn, "selected minimum intervals value 0.0 ≥ support lower buond -Inf. Support lower bound will be used as intervals starting value!") EnhancedBayesianNetworks._discretize_node(ebn, root3)

    @test_logs (:warn, "selected maximum intervals value 0.0 ≤ support's upper buond Inf. Support's upper bound will be used as intervals final value!") EnhancedBayesianNetworks._discretize_node(ebn, root4)

    discretization_root3 = ExactDiscretization([-Inf, 0, Inf])
    root3 = ContinuousRootNode(:z1, Normal(), discretization_root3)
    functional2_parents = [standard1_node, root3]
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, [functional2_model], functional2_performance, functional2_simulation)

    nodes = [standard1_node, root1, root3, root2, root4, standard2_node, functional2_node]
    ebn = EnhancedBayesianNetwork(nodes)

    disc_ebn_nodes = EnhancedBayesianNetworks._discretize_node(ebn, root3)
    z_d_node = DiscreteRootNode(:z1_d,
        Dict(
            Symbol("[-Inf, 0.0]") => 0.5,
            Symbol("[0.0, Inf]") => 0.5
        )
    )
    z_c_node = ContinuousChildNode(:z1, [z_d_node],
        Dict(
            [Symbol("[-Inf, 0.0]")] => truncated(Normal(0.0, 1.0), -Inf, 0.0),
            [Symbol("[0.0, Inf]")] => truncated(Normal(0.0, 1.0), 0.0, Inf)
        )
    )
    functional2_r_node = DiscreteFunctionalNode(functional2_name, [standard1_node, z_c_node], [functional2_model], functional2_performance, functional2_simulation)

    @test issetequal(disc_ebn_nodes, [root2, root1, root4, standard2_node, standard1_node, functional2_r_node, z_c_node, z_d_node])
end

