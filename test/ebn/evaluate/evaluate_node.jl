@testset "Evaluation Node" begin
    root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
    root2 = ContinuousRootNode(:B, Normal())
    @testset "Continuous Node" begin
        model = Model(df -> df.A .+ df.B, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [root1, root2], [model], sim)
        evaluated = EnhancedBayesianNetworks._evaluate(cont_functional)

        @test isa(evaluated, ContinuousChildNode)
        @test evaluated.name == :C
        @test isa(evaluated.distribution[[:a1]], EmpiricalDistribution)
        @test isa(evaluated.distribution[[:a2]], EmpiricalDistribution)
        @test issetequal(evaluated.parents, [root1])
        @test evaluated.discretization == cont_functional.discretization
        @test isa(evaluated.additional_info[[:a1]], Dict{Symbol,DataFrame})
        @test size(evaluated.additional_info[[:a1]][:samples]) == (sim.n, 3)
        @test isa(evaluated.additional_info[[:a2]], Dict{Symbol,DataFrame})
        @test size(evaluated.additional_info[[:a2]][:samples]) == (sim.n, 3)

        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [root2], [model], sim)

        evaluated = EnhancedBayesianNetworks._evaluate(cont_functional)

        @test isa(evaluated, ContinuousRootNode)
        @test evaluated.name == :C
        @test isa(evaluated.distribution, EmpiricalDistribution)
        @test isa(evaluated.distribution, EmpiricalDistribution)
        @test evaluated.discretization == ExactDiscretization(cont_functional.discretization.intervals)
    end

    @testset "Discrete Node" begin
        model = Model(df -> df.A .+ df.B, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [root1, root2], [model], performance, sim)

        evaluated = EnhancedBayesianNetworks._evaluate(disc_functional)

        @test evaluated.name == :C
        @test isapprox(evaluated.states[[:a1]][:safe_C], 0.84, atol=0.02)
        @test isapprox(evaluated.states[[:a1]][:fail_C], 0.16, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:safe_C], 0.5, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:fail_C], 0.5, atol=0.02)
        @test issetequal(evaluated.parents, [root1])
        @test evaluated.parameters == disc_functional.parameters
        @test isa(evaluated.additional_info[[:a1]], Dict{Symbol,Any})
        @test size(evaluated.additional_info[[:a1]][:samples]) == (sim.n, 3)
        @test isa(evaluated.additional_info[[:a1]][:cov], Real)
        @test isa(evaluated.additional_info[[:a2]], Dict{Symbol,Any})
        @test size(evaluated.additional_info[[:a2]][:samples]) == (sim.n, 3)
        @test isa(evaluated.additional_info[[:a1]][:cov], Real)

        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [root2], [model], performance, sim)

        evaluated = EnhancedBayesianNetworks._evaluate(disc_functional)

        @test isa(evaluated, DiscreteRootNode)
        @test evaluated.name == :C
        @test isapprox(evaluated.states[:safe_C], 0.84; atol=0.1)
        @test isapprox(evaluated.states[:fail_C], 0.15; atol=0.1)
        @test evaluated.parameters == disc_functional.parameters

        ##FORM
        model = Model(df -> 1 .+ df.B, :C)
        sim = FORM()
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [root2], [model], performance, sim)

        evaluated = EnhancedBayesianNetworks._evaluate(disc_functional)

        @test isa(evaluated, DiscreteRootNode)
        @test evaluated.name == :C
        @test isapprox(evaluated.states[:safe_C], 0.84; atol=0.1)
        @test isapprox(evaluated.states[:fail_C], 0.15; atol=0.1)
        @test evaluated.parameters == disc_functional.parameters
    end
end
