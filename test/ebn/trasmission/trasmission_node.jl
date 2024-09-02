@testset "Trasmission Nodes" begin

    @testset "Transfer Single Node" begin
        root1 = DiscreteRootNode(:x, Dict(:x1 => 0.3, :x2 => 0.7), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
        root2 = ContinuousRootNode(:y, Normal())

        model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :fc)
        discretization = ApproximatedDiscretization([-2, 0, 2], 2)
        cont_functional = ContinuousFunctionalNode(:cf, [root1, root2], [model1], MonteCarlo(300), discretization)

        model2 = Model(df -> df.fc .* 0.5, :fc)
        performance = df -> df.fc .- 0.5
        discrete_functional = DiscreteFunctionalNode(:fd, [cont_functional], [model2], performance, MonteCarlo(300))

        nodes = deepcopy([root1, root2, cont_functional, discrete_functional])

        @test EnhancedBayesianNetworks._transfer_single_continuous_functional!(nodes, cont_functional) == nodes

        cont_functional = ContinuousFunctionalNode(:cf, [root1, root2], [model1], MonteCarlo(300))

        discrete_functional = DiscreteFunctionalNode(:fd, [cont_functional], [model2], performance, MonteCarlo(300))

        nodes = deepcopy([root1, root2, cont_functional, discrete_functional])

        new_discrete_functional = DiscreteFunctionalNode(:fd, [root1, root2], [model1, model2], performance, MonteCarlo(300))

        @test issetequal(EnhancedBayesianNetworks._transfer_single_continuous_functional!(nodes, cont_functional), [root1, root2, new_discrete_functional])
    end
    @testset "Transfer All" begin
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

        model4 = Model(df -> df.c2 .* 0.5, :final2)
        performance2 = df -> df.final2 .- 0.5
        discrete_functional2 = DiscreteFunctionalNode(:fd2, [cont_functional2], [model4], performance2, MonteCarlo(300))

        nodes = deepcopy([root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, discrete_functional2])
        ebn = EnhancedBayesianNetwork(nodes)

        new_discrete_functional2 = DiscreteFunctionalNode(:fd2, [root2, root3], [model2, model4], performance2, MonteCarlo(300))
        new_discrete_functional1 = DiscreteFunctionalNode(:fd1, [root1, root2, root3], [model2, model1, model3], performance1, MonteCarlo(300))

        # l = EnhancedBayesianNetworks._transfer_continuous!(nodes)
        @test issetequal(EnhancedBayesianNetworks._transfer_continuous!(nodes), [root1, root2, root3, new_discrete_functional1, new_discrete_functional2])
    end
end