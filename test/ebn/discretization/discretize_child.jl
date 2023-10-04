@testset "Child Node Discretization" begin
    root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
    root3 = ContinuousRootNode(:z1, Normal())

    standard1_name = :β1
    standard1_parents = [root1]
    standard1_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    standard1_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    discretization_standard1 = ApproximatedDiscretization([-Inf, 0], 1.5)
    standard1_node = ContinuousChildNode(standard1_name, standard1_parents, standard1_states, discretization_standard1)

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
    discretization_standard2 = ApproximatedDiscretization([0, Inf], 1.5)
    standard2_node = ContinuousChildNode(standard2_name, standard2_parents, standard2_states, discretization_standard2)

    functional2_name = :f2
    functional2_parents = [standard2_node, root1]
    functional2_model = Model(df -> df.z1_d .+ df.x, :value1)
    functional2_simulation = MonteCarlo(800)
    functional2_performance = df -> 2 .- df.value1
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, [functional2_model], functional2_performance, functional2_simulation)

    nodes = [root1, root3, standard1_node, standard2_node, functional2_node]
    ebn = EnhancedBayesianNetwork(nodes)

    @test_logs (:warn, "selected minimum intervals value 0.0 ≥ support lower buond -Inf. Support lower bound will be used as intervals starting value!") EnhancedBayesianNetworks._discretize_node(ebn, standard2_node)

    @test_logs (:warn, "selected maximum intervals value 0.0 ≤ support's upper buond Inf. Support's upper bound will be used as intervals final value!") EnhancedBayesianNetworks._discretize_node(ebn, standard1_node)

    discretization_standard2 = ApproximatedDiscretization([-Inf, 0, Inf], 1.5)
    standard2_node = ContinuousChildNode(standard2_name, standard2_parents, standard2_states, discretization_standard2)
    functional2_parents = [standard2_node, root1]
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, [functional2_model], functional2_performance, functional2_simulation)
    nodes = [root1, root3, standard1_node, standard2_node, functional2_node]
    ebn = EnhancedBayesianNetwork(nodes)

    disc_ebn_nodes = EnhancedBayesianNetworks._discretize_node(ebn, standard2_node)

    β_d_node = DiscreteChildNode(:β_d, [root1], Dict(
        [:y] => Dict(
            Symbol("[-Inf, 0.0]") => 0.5,
            Symbol("[0.0, Inf]") => 0.5
        ),
        [:n] => Dict(
            Symbol("[-Inf, 0.0]") => 0.15865525393145702,
            Symbol("[0.0, Inf]") => 0.841344746068543
        )
    ))

    β_c_node = ContinuousChildNode(:β, [β_d_node], Dict(
        [Symbol("[-Inf, 0.0]")] => truncated(Normal(0.0, 1.5), -Inf, 0.0),
        [Symbol("[0.0, Inf]")] => truncated(Normal(0.0, 1.5), 0.0, Inf)
    ))

    functional2_r_node = DiscreteFunctionalNode(functional2_name, [root1, β_c_node], [functional2_model], functional2_performance, functional2_simulation)

    @test issetequal(disc_ebn_nodes, [root3, root1, standard1_node, functional2_r_node, β_c_node, β_d_node])
end