@testset "Trasmission Net" begin
    root1 = DiscreteRootNode(:x, Dict(:x1 => 0.3, :x2 => 0.7), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
    root2 = ContinuousRootNode(:y, Normal())
    root3 = DiscreteRootNode(:z, Dict(:z1 => 0.3, :z2 => 0.7), Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

    model1 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c1)
    cont_functional1 = ContinuousFunctionalNode(:cf1, [root1, root2], [model1], MonteCarlo(300))

    model2 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c2)
    cont_functional2 = ContinuousFunctionalNode(:cf2, [root2, root3], [model2], MonteCarlo(300))

    model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
    performance1 = df -> df.final1 .- 0.5
    discrete_functional1 = DiscreteFunctionalNode(:fd1, [cont_functional1, cont_functional2], [model3], performance1, MonteCarlo(300))

    model4 = Model(df -> df.c2 .* 0.5, :c3)
    continuous_functional3 = ContinuousFunctionalNode(:c3, [cont_functional2], [model4], MonteCarlo(300))

    model5 = Model(df -> df.final2 .* 0.5 .+ df.final1, :tot)
    performance2 = df -> 0.5 .- df.tot
    discrete_functional = DiscreteFunctionalNode(:fd, [discrete_functional1, continuous_functional3], [model5], performance2, MonteCarlo(300))

    nodes = [root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, continuous_functional3, discrete_functional]
    ebn = EnhancedBayesianNetwork(nodes)

    new_discrete_functional1 = DiscreteFunctionalNode(:fd1, [root3, root1, root2], [model1, model2, model3], performance1, MonteCarlo(300))

    new_discrete_functional = DiscreteFunctionalNode(:fd, [new_discrete_functional1, root2, root3], [model2, model4, model5], performance2, MonteCarlo(300))

    new_ebn = EnhancedBayesianNetworks._transfer_continuous(ebn)
    @test new_ebn == EnhancedBayesianNetwork(EnhancedBayesianNetworks._transfer_continuous!(deepcopy(ebn.nodes)))
    @test issetequal(new_ebn.nodes, [root1, root2, root3, new_discrete_functional1, new_discrete_functional])

end