@testset "Network Reduction" begin
    root1 = DiscreteNode(:x, DataFrame(:x => [:x1, :x2], :Π => [0.3, 0.7]), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
    root2 = ContinuousNode{UnivariateDistribution}(:y, DataFrame(:Π => Normal()))
    root3 = DiscreteNode(:z, DataFrame(:z => [:z1, :z2], :Π => [0.3, 0.7]), Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

    model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c1)
    cont_functional1 = ContinuousFunctionalNode(:cf1, [model1], MonteCarlo(300))

    model2 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c2)
    cont_functional2 = ContinuousFunctionalNode(:cf2, [model2], MonteCarlo(300))

    model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
    performance1 = df -> df.final1 .- 0.5
    discrete_functional1 = DiscreteFunctionalNode(:fd1, [model3], performance1, MonteCarlo(300), Dict(:fd1_fail => [Parameter(1, :fd1)], :fd1_safe => [Parameter(0, :fd1)]))

    model4 = Model(df -> df.c2 .* 0.5, :c3)
    cont_functional3 = ContinuousFunctionalNode(:c3, [model4], MonteCarlo(300))

    model5 = Model(df -> 0.5 .+ df.c3, :tot)
    performance2 = df -> 0.5 .- df.tot
    discrete_functional = DiscreteFunctionalNode(:fd, [model5], performance2, MonteCarlo(300))

    nodes = [root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, cont_functional3, discrete_functional]

    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, root1, cont_functional1)
    add_child!(ebn, root2, cont_functional1)
    add_child!(ebn, root2, cont_functional2)
    add_child!(ebn, root3, cont_functional2)
    add_child!(ebn, cont_functional1, discrete_functional1)
    add_child!(ebn, cont_functional2, discrete_functional1)
    add_child!(ebn, cont_functional2, cont_functional3)
    add_child!(ebn, discrete_functional1, discrete_functional)
    add_child!(ebn, cont_functional3, discrete_functional)
    order!(ebn)
    net1 = deepcopy(ebn)

    @test isnothing(reduce!(net1))
    @test net1.adj_matrix == [
        0.0 0.0 1.0 0.0;
        0.0 0.0 1.0 1.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0
    ]
    @test all(isa.(net1.nodes, DiscreteNode))
    @test net1.topology_dict == Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)

    evaluate!(ebn)
    net2 = deepcopy(ebn)
    reduce!(ebn)
    @test ebn.adj_matrix == net1.adj_matrix
    @test all(isa.(ebn.nodes, DiscreteNode))
    @test ebn.topology_dict == Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)

    EnhancedBayesianNetworks._eliminate_continuous_node!(net2, root2)

    @test net2.adj_matrix == [
        0.0 0.0 1.0 0.0;
        0.0 0.0 1.0 1.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0
    ]
    @test length(net2.nodes) == 4
    @test net2.topology_dict == Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)
end