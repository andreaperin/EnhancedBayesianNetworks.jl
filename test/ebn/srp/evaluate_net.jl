@testset "Evaluate eBN" begin

    @testset "Replace Nodes" begin

        root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
        root2 = ContinuousRootNode(:B, Normal())

        model = Model(df -> df.A .+ df.B, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [root1, root2], [model], sim)

        a = evaluate(cont_functional)

        list = EnhancedBayesianNetworks._replace_node([root1, root2, cont_functional], cont_functional, a)

        @test list == [root1, root2, a]

    end

    @testset "Transfer Continuous Functional Nodes" begin

    end

end