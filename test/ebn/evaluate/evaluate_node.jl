@testset "Evaluation Node" begin
    root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
    root2 = ContinuousRootNode(:B, Normal())
    model = Model(df -> df.A .+ df.B, :C)
    sim = MonteCarlo(100_000)
    performance = df -> 2 .- df.C

    @testset "Continuous Node" begin
        cont_functional = ContinuousFunctionalNode(:C, [root1, root2], [model], sim)

        evaluated = EnhancedBayesianNetworks._evaluate(cont_functional)

        @test evaluated.name == :C
        @test isa(evaluated.distribution[[:a1]], EmpiricalDistribution)
        @test isa(evaluated.distribution[[:a2]], EmpiricalDistribution)
        @test issetequal(evaluated.parents, [root1])
        @test evaluated.discretization == cont_functional.discretization
        @test isa(evaluated.samples[[:a1]], DataFrame)
        @test size(evaluated.samples[[:a1]]) == (sim.n, 3)
        @test isa(evaluated.samples[[:a2]], DataFrame)
        @test size(evaluated.samples[[:a2]]) == (sim.n, 3)
    end

    @testset "Discrete Node" begin
        disc_functional = DiscreteFunctionalNode(:C, [root1, root2], [model], performance, sim)

        evaluated = EnhancedBayesianNetworks._evaluate(disc_functional)

        @test evaluated.name == :C
        @test isapprox(evaluated.states[[:a1]][:safe_C], 0.84, atol=0.02)
        @test isapprox(evaluated.states[[:a1]][:fail_C], 0.16, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:safe_C], 0.5, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:fail_C], 0.5, atol=0.02)
        @test issetequal(evaluated.parents, [root1])
        @test evaluated.parameters == disc_functional.parameters
        @test isa(evaluated.samples[[:a1]], DataFrame)
        @test size(evaluated.samples[[:a1]]) == (sim.n, 3)
        @test isa(evaluated.samples[[:a2]], DataFrame)
        @test size(evaluated.samples[[:a2]]) == (sim.n, 3)

        root3 = ContinuousRootNode(:P, (0, 10))
        model = Model(df -> df.A .+ df.B .+ df.P, :C)
        disc_functional = DiscreteFunctionalNode(:C, [root1, root2, root3], [model], performance, sim)

        evaluated = EnhancedBayesianNetworks._evaluate(disc_functional)
        @test evaluated.name == :C

        @test isapprox(evaluated.states[[:a1]][:safe_C][1], 0.0, atol=0.02)
        @test isapprox(evaluated.states[[:a1]][:safe_C][2], 0.84183, atol=0.02)
        @test isapprox(evaluated.states[[:a1]][:fail_C][1], 0.15817, atol=0.02)
        @test isapprox(evaluated.states[[:a1]][:fail_C][2], 1.0, atol=0.02)

        @test isapprox(evaluated.states[[:a2]][:safe_C][1], 0.0, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:safe_C][2], 0.50046, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:fail_C][1], 0.49954, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:fail_C][2], 1.0, atol=0.02)

        @test issetequal(evaluated.parents, [root1])
        @test evaluated.parameters == disc_functional.parameters
        @test isa(evaluated.samples[[:a1]], DataFrame)
        @test isa(evaluated.samples[[:a2]], DataFrame)
    end
end