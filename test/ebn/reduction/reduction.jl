@testset "Node Elimination" begin
    root1 = DiscreteRootNode(:x, Dict(:x1 => 0.3, :x2 => 0.7), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
    root2 = ContinuousRootNode(:y, Normal())
    root3 = DiscreteRootNode(:z, Dict(:z1 => 0.3, :z2 => 0.7), Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

    model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c1)
    cont_functional1 = ContinuousFunctionalNode(:cf1, [model1], MonteCarlo(300))

    model2 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c2)
    cont_functional2 = ContinuousFunctionalNode(:cf2, [model2], MonteCarlo(300))

    model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
    performance1 = df -> df.final1 .- 0.5
    discrete_functional1 = DiscreteFunctionalNode(:fd1, [model3], performance1, MonteCarlo(300), Dict(:fail_fd1 => [Parameter(1, :fd1)], :safe_fd1 => [Parameter(0, :fd1)]))

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

    @test_throws ErrorException("node elimination algorithm is for continuous nodes and x is discrete") EnhancedBayesianNetworks._is_eliminable(ebn, root1)

    @test EnhancedBayesianNetworks._is_eliminable(ebn, cont_functional2) == false
    @test EnhancedBayesianNetworks._is_eliminable(ebn, root2)
    @test EnhancedBayesianNetworks._is_eliminable(ebn, root2) == EnhancedBayesianNetworks._is_eliminable(ebn, 2)
    @test EnhancedBayesianNetworks._is_eliminable(ebn, root2) == EnhancedBayesianNetworks._is_eliminable(ebn, :y)
end