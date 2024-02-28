@testset "Evaluation" begin
    root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
    root2 = ContinuousRootNode(:B, Normal())
    model = Model(df -> df.A .+ df.B, :C)
    sim = MonteCarlo(100_000)
    performance = df -> 2 .- df.C

    @testset "Continuous Node" begin
        cont_functional = ContinuousFunctionalNode(:C, [root1, root2], [model], sim)

        a = EnhancedBayesianNetworks._evaluate(cont_functional)

        @test a.name == :C
        @test isa(a.distributions[[:a1]], EmpiricalDistribution)
        @test isa(a.distributions[[:a2]], EmpiricalDistribution)
        @test issetequal(a.parents, [root1])
        @test a.discretization == cont_functional.discretization
        @test isa(a.samples[[:a1]], DataFrame)
        @test size(a.samples[[:a1]]) == (sim.n, 3)
        @test isa(a.samples[[:a2]], DataFrame)
        @test size(a.samples[[:a2]]) == (sim.n, 3)
    end

    @testset "Discrete Node" begin
        disc_functional = DiscreteFunctionalNode(:C, [root1, root2], [model], performance, sim)

        a = EnhancedBayesianNetworks._evaluate(disc_functional)

        @test a.name == :C
        @test isapprox(a.states[[:a1]][:safe_C], 0.84, atol=0.02)
        @test isapprox(a.states[[:a1]][:fail_C], 0.16, atol=0.02)
        @test isapprox(a.states[[:a2]][:safe_C], 0.5, atol=0.02)
        @test isapprox(a.states[[:a2]][:fail_C], 0.5, atol=0.02)
        @test issetequal(a.parents, [root1])
        @test a.parameters == disc_functional.parameters
        @test isa(a.samples[[:a1]], DataFrame)
        @test size(a.samples[[:a1]]) == (sim.n, 3)
        @test isa(a.samples[[:a2]], DataFrame)
        @test size(a.samples[[:a2]]) == (sim.n, 3)

    end
end